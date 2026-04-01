import json
import logging
import os
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from data_service import DataService
from station_placement import pipeline_result_to_geojson, run_full_pipeline

app = FastAPI(title="Clustering", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_service = DataService()
logger = logging.getLogger(__name__)


def _dedupe_close_parallel_voronoi_edges(
    fc: dict,
    *,
    max_midpoint_distance_m: float = 90.0,
    max_angle_diff_deg: float = 18.0,
) -> dict:
    """Убирает почти дублирующиеся близкие и параллельные рёбра Вороного."""
    if not fc or not isinstance(fc, dict):
        return {"type": "FeatureCollection", "features": []}
    feats = list(fc.get("features", []) or [])
    if len(feats) < 2:
        return {"type": "FeatureCollection", "features": feats}

    angle_thr = np.radians(float(max_angle_diff_deg))

    def _xy_m(lon: float, lat: float, ref_lat: float) -> tuple[float, float]:
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.radians(ref_lat))
        return lon * m_per_deg_lon, lat * m_per_deg_lat

    def _enrich(feature: dict):
        geom = (feature or {}).get("geometry") or {}
        if geom.get("type") != "LineString":
            return None
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            return None
        p1 = coords[0]
        p2 = coords[-1]
        if len(p1) < 2 or len(p2) < 2:
            return None
        lon1, lat1 = float(p1[0]), float(p1[1])
        lon2, lat2 = float(p2[0]), float(p2[1])
        ref_lat = 0.5 * (lat1 + lat2)
        x1, y1 = _xy_m(lon1, lat1, ref_lat)
        x2, y2 = _xy_m(lon2, lat2, ref_lat)
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 0.1:
            return None
        ang = float(np.arctan2(dy, dx))
        if ang < 0:
            ang += np.pi
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        return {
            "feature": feature,
            "cluster": str(((feature.get("properties") or {}).get("cluster_id"))),
            "p1": (x1, y1),
            "p2": (x2, y2),
            "mx": mx,
            "my": my,
            "angle": ang,
            "length": length,
            "ux": dx / length,
            "uy": dy / length,
        }

    enriched = [e for e in (_enrich(f) for f in feats) if e is not None]
    if len(enriched) < 2:
        return {"type": "FeatureCollection", "features": feats}

    by_cluster: dict[str, list[dict]] = {}
    for e in enriched:
        by_cluster.setdefault(e["cluster"], []).append(e)

    kept_features: list[dict] = []
    for _, items in by_cluster.items():
        items.sort(key=lambda it: it["length"], reverse=True)
        kept: list[dict] = []
        for cur in items:
            is_dup = False
            for k in kept:
                d_ang = abs(cur["angle"] - k["angle"])
                d_ang = min(d_ang, np.pi - d_ang)
                if d_ang > angle_thr:
                    continue
                d_mid = float(np.hypot(cur["mx"] - k["mx"], cur["my"] - k["my"]))
                if d_mid > max_midpoint_distance_m:
                    continue
                # Проверка "почти одной и той же полосы": малая поперечная дистанция
                # и заметное продольное перекрытие.
                nx, ny = -k["uy"], k["ux"]  # нормаль к опорной линии
                perp_cur = abs((cur["mx"] - k["mx"]) * nx + (cur["my"] - k["my"]) * ny)
                if perp_cur > (max_midpoint_distance_m * 0.7):
                    continue

                tx, ty = k["ux"], k["uy"]  # тангенс опорной линии
                cur_min_t = min(cur["p1"][0] * tx + cur["p1"][1] * ty, cur["p2"][0] * tx + cur["p2"][1] * ty)
                cur_max_t = max(cur["p1"][0] * tx + cur["p1"][1] * ty, cur["p2"][0] * tx + cur["p2"][1] * ty)
                k_min_t = min(k["p1"][0] * tx + k["p1"][1] * ty, k["p2"][0] * tx + k["p2"][1] * ty)
                k_max_t = max(k["p1"][0] * tx + k["p1"][1] * ty, k["p2"][0] * tx + k["p2"][1] * ty)
                overlap = max(0.0, min(cur_max_t, k_max_t) - max(cur_min_t, k_min_t))
                min_len = max(1.0, min(cur["length"], k["length"]))
                if overlap >= (0.35 * min_len):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cur)
        kept_features.extend([k["feature"] for k in kept])

    return {"type": "FeatureCollection", "features": kept_features}


def _build_voronoi_local_paths_fc(
    city: str,
    network_type: str,
    simplify: bool,
    dbscan_eps_m: float,
    dbscan_min_samples: int,
    use_all_buildings: bool,
    buildings_per_centroid: int = 60,
    charging_station_features: list | None = None,
) -> dict:
    try:
        from scipy.spatial import Voronoi
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scipy Voronoi недоступен: {e}")

    data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
    buildings = data.get("buildings")
    road_graph = data.get("road_graph")
    city_boundary = data.get("city_boundary")
    no_fly_zones = data.get("no_fly_zones")

    if buildings is None or len(buildings) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    buildings_wgs = buildings.to_crs("EPSG:4326").copy()
    if not use_all_buildings:
        buildings_wgs = data_service._filter_buildings_for_demand(buildings_wgs)
    if buildings_wgs is None or len(buildings_wgs) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    hulls_gdf = data.get("demand_hulls")
    if hulls_gdf is None or len(hulls_gdf) == 0:
        result = data_service.get_demand_points_weighted(
            buildings_wgs,
            road_graph,
            city_boundary,
            method="dbscan",
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
            return_hulls=True,
            use_all_buildings=use_all_buildings,
            no_fly_zones=no_fly_zones,
        )
        _, hulls_gdf = result if isinstance(result, tuple) else (None, None)

    if hulls_gdf is None or len(hulls_gdf) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    hulls = hulls_gdf.to_crs("EPSG:4326").copy()
    if "cluster_id" not in hulls.columns:
        if "subcluster_id" in hulls.columns:
            hulls["cluster_id"] = hulls["subcluster_id"]
        elif "region_id" in hulls.columns:
            hulls["cluster_id"] = hulls["region_id"]
        else:
            return {
                "type": "FeatureCollection",
                "features": [],
                "clusters_total": 0,
                "clusters_with_paths": 0,
                "edges_total": 0,
            }
    hulls = hulls[hulls["cluster_id"].notna()].copy()
    if len(hulls) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    # Привязываем здания к cluster_id через принадлежность центроида полигонам hull.
    b_proj = buildings_wgs.to_crs(epsg=3857)
    b_centroids_wgs = gpd.GeoSeries(b_proj.geometry.centroid, crs=b_proj.crs).to_crs("EPSG:4326")
    b_points = gpd.GeoDataFrame({"geometry": b_centroids_wgs}, index=buildings_wgs.index, crs="EPSG:4326")
    try:
        assigned = gpd.sjoin(
            b_points,
            hulls[["cluster_id", "geometry"]],
            how="inner",
            predicate="within",
        )
    except Exception:
        # fallback для старых версий geopandas
        assigned = gpd.sjoin(
            b_points,
            hulls[["cluster_id", "geometry"]],
            how="inner",
            op="within",
        )

    if assigned is None or len(assigned) < 2:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }
    assigned = assigned.copy()
    if "index_left" in assigned.columns:
        assigned["_left_id"] = assigned["index_left"]
    else:
        assigned["_left_id"] = assigned.index
    assigned = assigned.drop_duplicates(subset=["_left_id"]).set_index("_left_id")
    assigned = assigned[assigned["cluster_id"].notna()].copy()
    if len(assigned) < 2:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    charging_by_cluster: dict[str, list[list[float]]] = {}
    if charging_station_features:
        for f in charging_station_features:
            if not isinstance(f, dict):
                continue
            geom = f.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            props = f.get("properties") or {}
            cid = props.get("cluster_id")
            if cid is None:
                continue
            key = str(cid)
            charging_by_cluster.setdefault(key, []).append([float(coords[0]), float(coords[1])])

    target_group_size = max(1, int(buildings_per_centroid or 20))
    features = []
    clusters_total = 0
    clusters_with_paths = 0

    for group_id, sub in assigned.groupby("cluster_id"):
        if sub is None or len(sub) < 2:
            continue
        clusters_total += 1

        pts = []
        for _, row in sub.iterrows():
            geom = row.get("geometry")
            if geom is None or geom.is_empty:
                continue
            try:
                pts.append([float(geom.x), float(geom.y)])
            except Exception:
                continue
        if len(pts) < 2:
            continue

        pts_arr = np.array(pts, dtype=float)
        if len(pts_arr) > target_group_size:
            n_groups = int(np.ceil(len(pts_arr) / float(target_group_size)))
            n_groups = max(2, min(n_groups, len(pts_arr)))
            try:
                from sklearn.cluster import MiniBatchKMeans

                km = MiniBatchKMeans(
                    n_clusters=n_groups,
                    random_state=42,
                    batch_size=min(4096, max(1024, len(pts_arr) // 3)),
                    n_init=3,
                    max_iter=200,
                )
                labels = km.fit_predict(pts_arr)
                centers = []
                for k in range(n_groups):
                    members = pts_arr[labels == k]
                    if len(members) == 0:
                        continue
                    centers.append(members.mean(axis=0))
                if len(centers) >= 2:
                    pts_arr = np.asarray(centers, dtype=float)
            except Exception:
                # Fallback без sklearn: грубая агрегация по сортировке координат.
                order = np.lexsort((pts_arr[:, 1], pts_arr[:, 0]))
                ordered = pts_arr[order]
                centers = []
                for i in range(0, len(ordered), target_group_size):
                    chunk = ordered[i : i + target_group_size]
                    if len(chunk) == 0:
                        continue
                    centers.append(chunk.mean(axis=0))
                if len(centers) >= 2:
                    pts_arr = np.asarray(centers, dtype=float)

        # Добавляем зарядные станции как отдельные центроиды (не агрегируем их по 60 зданий).
        st_points = charging_by_cluster.get(str(group_id), [])
        if st_points:
            st_arr = np.asarray(st_points, dtype=float)
            if len(st_arr) > 0:
                pts_arr = np.vstack([pts_arr, st_arr]) if len(pts_arr) > 0 else st_arr
                # Убираем дубликаты координат после объединения.
                pts_arr = np.unique(pts_arr, axis=0)

        if len(pts_arr) == 2:
            p0, p1 = pts_arr[0], pts_arr[1]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]],
                    },
                    "properties": {
                        "cluster_id": int(group_id) if isinstance(group_id, (int, np.integer)) else str(group_id),
                        "edge_type": "voronoi_local",
                        "source_i": 0,
                        "target_i": 1,
                    },
                }
            )
            clusters_with_paths += 1
            continue

        try:
            vor = Voronoi(pts_arr)
            map_idx = {i: i for i in range(len(pts_arr))}
        except Exception:
            uniq, uniq_idx = np.unique(pts_arr, axis=0, return_index=True)
            if len(uniq) < 3:
                continue
            try:
                vor = Voronoi(uniq)
                map_idx = {new_i: int(orig_i) for new_i, orig_i in enumerate(uniq_idx.tolist())}
            except Exception:
                continue

        group_edges_before = len(features)
        seen_pairs = set()
        for ridge in vor.ridge_points:
            i_raw, j_raw = int(ridge[0]), int(ridge[1])
            if i_raw not in map_idx or j_raw not in map_idx:
                continue
            i = map_idx[i_raw]
            j = map_idx[j_raw]
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            pi = pts_arr[a]
            pj = pts_arr[b]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(pi[0]), float(pi[1])], [float(pj[0]), float(pj[1])]],
                    },
                    "properties": {
                        "cluster_id": int(group_id) if isinstance(group_id, (int, np.integer)) else str(group_id),
                        "edge_type": "voronoi_local",
                        "source_i": int(a),
                        "target_i": int(b),
                    },
                }
            )
        if len(features) > group_edges_before:
            clusters_with_paths += 1

    deduped = _dedupe_close_parallel_voronoi_edges({
        "type": "FeatureCollection",
        "features": features,
    })
    deduped["clusters_total"] = int(clusters_total)
    deduped["clusters_with_paths"] = int(clusters_with_paths)
    deduped["edges_total"] = len(deduped.get("features", []) or [])
    return deduped


def _build_voronoi_edges_from_station_geojson(geo: dict) -> dict:
    """Строит локальные рёбра Вороного по точкам станций из ответа pipeline_result_to_geojson."""
    try:
        from scipy.spatial import Voronoi
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scipy Voronoi недоступен: {e}")

    station_layers = [
        geo.get("charging_type_a") or {"type": "FeatureCollection", "features": []},
        geo.get("charging_type_b") or {"type": "FeatureCollection", "features": []},
        geo.get("garages") or {"type": "FeatureCollection", "features": []},
        geo.get("to_stations") or {"type": "FeatureCollection", "features": []},
    ]

    by_cluster: dict[str, list[list[float]]] = {}
    by_region: dict[str, list[list[float]]] = {}
    all_pts: list[list[float]] = []
    for layer in station_layers:
        for f in layer.get("features", []) or []:
            geom = f.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            props = f.get("properties") or {}
            cid = props.get("cluster_id")
            pt = [float(coords[0]), float(coords[1])]
            all_pts.append(pt)
            if cid is not None:
                key = str(cid)
                by_cluster.setdefault(key, []).append(pt)
            rid = props.get("region_id")
            if rid is not None:
                rkey = str(rid)
                by_region.setdefault(rkey, []).append(pt)

    # Если по cluster_id почти нет пар для Вороного (часто 1 станция на кластер),
    # переключаемся на region_id, а затем на all stations.
    groups = by_cluster
    if sum(1 for pts in groups.values() if len(pts) >= 2) == 0:
        groups = by_region if sum(1 for pts in by_region.values() if len(pts) >= 2) > 0 else {}
    if sum(1 for pts in groups.values() if len(pts) >= 2) == 0 and len(all_pts) >= 2:
        groups = {"all_stations": all_pts}

    out_features = []
    for cluster_key, pts in groups.items():
        if len(pts) < 2:
            continue
        pts_arr = np.asarray(pts, dtype=float)
        if len(pts_arr) == 2:
            p0, p1 = pts_arr[0], pts_arr[1]
            out_features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]],
                    },
                    "properties": {
                        "cluster_id": cluster_key,
                        "edge_type": "voronoi_station_local",
                        "group_mode": "cluster_or_fallback",
                        "source_i": 0,
                        "target_i": 1,
                    },
                }
            )
            continue
        try:
            vor = Voronoi(pts_arr)
            map_idx = {i: i for i in range(len(pts_arr))}
        except Exception:
            uniq, uniq_idx = np.unique(pts_arr, axis=0, return_index=True)
            if len(uniq) < 3:
                continue
            try:
                vor = Voronoi(uniq)
                map_idx = {new_i: int(orig_i) for new_i, orig_i in enumerate(uniq_idx.tolist())}
            except Exception:
                continue

        seen = set()
        for ridge in vor.ridge_points:
            i_raw, j_raw = int(ridge[0]), int(ridge[1])
            if i_raw not in map_idx or j_raw not in map_idx:
                continue
            i, j = map_idx[i_raw], map_idx[j_raw]
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pa, pb = pts_arr[a], pts_arr[b]
            out_features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(pa[0]), float(pa[1])], [float(pb[0]), float(pb[1])]],
                    },
                    "properties": {
                        "cluster_id": cluster_key,
                        "edge_type": "voronoi_station_local",
                        "group_mode": "cluster_or_fallback",
                        "source_i": int(a),
                        "target_i": int(b),
                    },
                }
            )

    return {"type": "FeatureCollection", "features": out_features}


def _build_voronoi_edges_from_pipeline_raw(
    raw: dict,
    charging_station_features: list | None = None,
    buildings_per_centroid: int = 60,
) -> dict:
    """
    Быстрый Вороной без повторного data_service.get_city_data:
    используем demand из уже выполненного run_full_pipeline.
    """
    try:
        from scipy.spatial import Voronoi
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scipy Voronoi недоступен: {e}")

    demand = raw.get("demand")
    if demand is None or len(demand) == 0:
        return {"type": "FeatureCollection", "features": []}
    d = demand.copy()
    if "is_cluster_fill" in d.columns:
        d = d[~d["is_cluster_fill"].fillna(False).astype(bool)].copy()
    if len(d) == 0:
        return {"type": "FeatureCollection", "features": []}

    if "cluster_id" not in d.columns:
        if "subcluster_id" in d.columns:
            d["cluster_id"] = d["subcluster_id"]
        else:
            return {"type": "FeatureCollection", "features": []}
    d = d[d["cluster_id"].notna()].copy()
    if len(d) == 0:
        return {"type": "FeatureCollection", "features": []}

    # Базовые точки по кластерам из demand (часто по 1 точке на кластер).
    grouped_points: dict[str, list[list[float]]] = {}
    for cluster_id, sub in d.groupby("cluster_id"):
        pts = []
        for _, row in sub.iterrows():
            g = row.get("geometry")
            if g is None or g.is_empty:
                continue
            try:
                pts.append([float(g.x), float(g.y)])
            except Exception:
                continue
        if pts:
            grouped_points[str(cluster_id)] = pts

    # Если в demand по кластерам в основном по 1 точке, строим точки из зданий:
    # каждое здание привязываем к ближайшему demand-центроиду его кластера.
    if not any(len(pts) >= 2 for pts in grouped_points.values()):
        buildings = raw.get("buildings")
        if buildings is not None and len(buildings) > 0 and len(grouped_points) > 0:
            try:
                try:
                    from scipy.spatial import cKDTree
                except Exception:
                    cKDTree = None

                b = buildings.to_crs("EPSG:4326").copy()
                try:
                    b = data_service._filter_buildings_for_demand(b)
                except Exception:
                    pass
                if b is not None and len(b) > 0:
                    b_proj = b.to_crs(epsg=3857)
                    b_centroids_wgs = gpd.GeoSeries(b_proj.geometry.centroid, crs=b_proj.crs).to_crs("EPSG:4326")
                    b_pts = np.array([[float(g.x), float(g.y)] for g in b_centroids_wgs if g is not None and not g.is_empty], dtype=float)

                    cluster_keys = list(grouped_points.keys())
                    centroid_pts = np.array([grouped_points[k][0] for k in cluster_keys], dtype=float)
                    assigned: dict[str, list[list[float]]] = {k: [] for k in cluster_keys}
                    if len(b_pts) > 0 and len(centroid_pts) > 0:
                        if cKDTree is not None:
                            tree = cKDTree(centroid_pts)
                            _, idx = tree.query(b_pts, k=1)
                            for p, j in zip(b_pts, idx):
                                assigned[cluster_keys[int(j)]].append([float(p[0]), float(p[1])])
                        else:
                            for p in b_pts:
                                d2 = np.sum((centroid_pts - p) ** 2, axis=1)
                                j = int(np.argmin(d2))
                                assigned[cluster_keys[j]].append([float(p[0]), float(p[1])])
                        grouped_points = assigned
            except Exception:
                pass

    charging_by_cluster: dict[str, list[list[float]]] = {}
    if charging_station_features:
        for f in charging_station_features:
            geom = (f or {}).get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            props = (f or {}).get("properties") or {}
            cid = props.get("cluster_id")
            if cid is None:
                continue
            charging_by_cluster.setdefault(str(cid), []).append([float(coords[0]), float(coords[1])])

    target_group_size = max(1, int(buildings_per_centroid or 60))
    out_features = []
    for cluster_id, pts in grouped_points.items():
        if len(pts) < 2:
            continue

        pts_arr = np.asarray(pts, dtype=float)
        if len(pts_arr) > target_group_size:
            n_groups = int(np.ceil(len(pts_arr) / float(target_group_size)))
            n_groups = max(2, min(n_groups, len(pts_arr)))
            try:
                from sklearn.cluster import MiniBatchKMeans

                km = MiniBatchKMeans(
                    n_clusters=n_groups,
                    random_state=42,
                    batch_size=min(4096, max(1024, len(pts_arr) // 3)),
                    n_init=3,
                    max_iter=200,
                )
                labels = km.fit_predict(pts_arr)
                centers = []
                for k in range(n_groups):
                    members = pts_arr[labels == k]
                    if len(members) > 0:
                        centers.append(members.mean(axis=0))
                if len(centers) >= 2:
                    pts_arr = np.asarray(centers, dtype=float)
            except Exception:
                pass

        st_pts = charging_by_cluster.get(str(cluster_id), [])
        if st_pts:
            st_arr = np.asarray(st_pts, dtype=float)
            if len(st_arr) > 0:
                pts_arr = np.vstack([pts_arr, st_arr])
                pts_arr = np.unique(pts_arr, axis=0)

        if len(pts_arr) < 2:
            continue
        if len(pts_arr) == 2:
            p0, p1 = pts_arr[0], pts_arr[1]
            out_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]},
                    "properties": {"cluster_id": str(cluster_id), "edge_type": "voronoi_local", "source_i": 0, "target_i": 1},
                }
            )
            continue

        try:
            vor = Voronoi(pts_arr)
            map_idx = {i: i for i in range(len(pts_arr))}
        except Exception:
            uniq, uniq_idx = np.unique(pts_arr, axis=0, return_index=True)
            if len(uniq) < 3:
                continue
            try:
                vor = Voronoi(uniq)
                map_idx = {new_i: int(orig_i) for new_i, orig_i in enumerate(uniq_idx.tolist())}
            except Exception:
                continue

        seen = set()
        for ridge in vor.ridge_points:
            i_raw, j_raw = int(ridge[0]), int(ridge[1])
            if i_raw not in map_idx or j_raw not in map_idx:
                continue
            i, j = map_idx[i_raw], map_idx[j_raw]
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pa, pb = pts_arr[a], pts_arr[b]
            out_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[float(pa[0]), float(pa[1])], [float(pb[0]), float(pb[1])]]},
                    "properties": {"cluster_id": str(cluster_id), "edge_type": "voronoi_local", "source_i": int(a), "target_i": int(b)},
                }
            )
    return {"type": "FeatureCollection", "features": out_features}


def _placement_cache_key(
    city: str,
    network_type: str,
    simplify: bool,
    demand_method: str,
    dbscan_eps_m: float,
    dbscan_min_samples: int,
    a_by_admin_districts: bool,
    candidates_per_cluster: int,
) -> str:
    payload = {
        "city": city.strip(),
        "network_type": network_type,
        "simplify": bool(simplify),
        "demand_method": demand_method,
        "dbscan_eps_m": float(dbscan_eps_m),
        "dbscan_min_samples": int(dbscan_min_samples),
        "a_by_admin_districts": bool(a_by_admin_districts),
        "candidates_per_cluster": int(candidates_per_cluster),
        # Смена версии сбрасывает устаревший Redis-кэш (иначе после правок пайплайна
        # клиент может бесконечно получать пустой сохранённый ответ).
        "placement_schema": 13,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"drone_planner:placement:{digest}"


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Нет templates/index.html</h1>", status_code=404)


@app.get("/api/ping")
def ping():
    return {"status": "ok"}


@app.get("/api/city/map")
def get_city_map(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение"),
):
    """Границы города, no_fly_zones, bbox/center для карты."""
    try:
        data = data_service.get_city_data(
            city, network_type=network_type, simplify=simplify, load_no_fly_zones=True
        )
        city_boundary = data.get("city_boundary")
        road_graph = data.get("road_graph")
        no_fly_zones = data.get("no_fly_zones")
        no_fly_zones_geojson = None
        if no_fly_zones is not None:
            try:
                if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                    no_fly_zones_geojson = json.loads(no_fly_zones.to_json())
                elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                    zones_gdf = gpd.GeoDataFrame([{"geometry": z} for z in no_fly_zones], crs="EPSG:4326")
                    no_fly_zones_geojson = json.loads(zones_gdf.to_json())
            except Exception as e:
                logger.warning("Ошибка сериализации no_fly_zones: %s", e)
        if city_boundary is not None:
            boundary_gdf = gpd.GeoDataFrame([{"geometry": city_boundary}], crs="EPSG:4326")
            city_boundary_geojson = json.loads(boundary_gdf.to_json())
            bbox = list(city_boundary.bounds)
            center_lon = (bbox[0] + bbox[2]) / 2.0
            center_lat = (bbox[1] + bbox[3]) / 2.0
        elif road_graph is not None and len(road_graph.nodes) > 0:
            import osmnx as ox

            gdf_nodes, _ = ox.graph_to_gdfs(road_graph)
            xmin, ymin, xmax, ymax = gdf_nodes.total_bounds
            bbox = [xmin, ymin, xmax, ymax]
            center_lat = (ymin + ymax) / 2.0
            center_lon = (xmin + xmax) / 2.0
            city_boundary_geojson = None
        else:
            raise HTTPException(status_code=404, detail="Нет данных по городу")
        return JSONResponse(
            {
                "city_boundary": city_boundary_geojson,
                "no_fly_zones": no_fly_zones_geojson,
                "bbox": bbox,
                "center": {"lat": center_lat, "lon": center_lon},
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/buildings/export")
def export_buildings_with_heights(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение"),
):
    """Здания с высотами для подложки карты."""
    try:
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
        buildings = data.get("buildings")
        flight_levels = data.get("flight_levels", [])
        if buildings is None or len(buildings) == 0:
            return JSONResponse(
                {
                    "type": "FeatureCollection",
                    "features": [],
                    "flight_levels": flight_levels,
                    "building_height_stats": data.get("building_height_stats", {}),
                }
            )

        if "height_m" not in buildings.columns:
            buildings = data_service._compute_building_heights(buildings)

        if "area_m2" not in buildings.columns:
            try:
                b_proj = buildings
                if b_proj.crs is None:
                    b_proj = b_proj.set_crs("EPSG:4326", allow_override=True)
                if b_proj.crs and b_proj.crs.is_geographic:
                    b_proj = b_proj.to_crs(epsg=3857)
                buildings = buildings.copy()
                buildings["area_m2"] = b_proj.geometry.area.astype(float)
            except Exception:
                try:
                    buildings = buildings.copy()
                    buildings["area_m2"] = buildings.geometry.area.astype(float)
                except Exception:
                    buildings = buildings.copy()
                    buildings["area_m2"] = None

        levels_m = [f["altitude_m"] for f in flight_levels] if flight_levels else [40, 65, 90, 115]
        features = []
        for idx, row in buildings.iterrows():
            geom = row.geometry
            if geom is None or not getattr(geom, "is_valid", True):
                continue
            h = float(row.get("height_m", 10))
            a_raw = row.get("area_m2", None)
            try:
                a = float(a_raw) if a_raw is not None else None
            except (TypeError, ValueError):
                a = None
            rec_level = 1
            for i, alt in enumerate(levels_m):
                if alt >= h + 10:
                    rec_level = i + 1
                    break
            feat = {
                "type": "Feature",
                "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
                "properties": {
                    "height_m": round(h, 1),
                    "area_m2": round(a, 1) if a is not None else None,
                    "flight_level": rec_level,
                },
            }
            features.append(feat)
        return JSONResponse(
            {
                "type": "FeatureCollection",
                "features": features,
                "flight_levels": flight_levels,
                "building_height_stats": data.get("building_height_stats", {}),
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/buildings/clusters")
def get_building_clusters(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
    method: str = Query("dbscan", description="Только dbscan (кластеризация по зданиям)"),
    dbscan_eps_m: float = Query(180.0, description="Радиус локального уровня (м)"),
    dbscan_min_samples: int = Query(15, description="min_samples DBSCAN локального уровня"),
    use_all_buildings: bool = Query(False, description="Все здания вне no-fly"),
):
    """Кластеры спроса (двухуровневая кластеризация в data_service) + полигоны hull."""

    def round_coords(obj, precision=6):
        if isinstance(obj, dict):
            return {k: round_coords(v, precision) for k, v in obj.items()}
        if isinstance(obj, list):
            return [round_coords(item, precision) for item in obj]
        if isinstance(obj, float):
            return round(obj, precision)
        return obj

    if method != "dbscan":
        raise HTTPException(status_code=400, detail="Поддерживается только method=dbscan")

    try:
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
        buildings = data.get("buildings")
        road_graph = data.get("road_graph")
        city_boundary = data.get("city_boundary")
        if buildings is None or len(buildings) == 0:
            return JSONResponse(
                {
                    "buildings": {"type": "FeatureCollection", "features": []},
                    "clusters": {"type": "FeatureCollection", "features": []},
                    "cluster_hulls": {"type": "FeatureCollection", "features": []},
                    "total": 0,
                    "message": "Нет зданий",
                }
            )

        demand_gdf = data.get("demand_points")
        hulls_gdf = data.get("demand_hulls")
        if demand_gdf is None or len(demand_gdf) == 0:
            no_fly_zones = data.get("no_fly_zones")
            result = data_service.get_demand_points_weighted(
                buildings,
                road_graph,
                city_boundary,
                method="dbscan",
                dbscan_eps_m=dbscan_eps_m,
                dbscan_min_samples=dbscan_min_samples,
                return_hulls=True,
                use_all_buildings=use_all_buildings,
                no_fly_zones=no_fly_zones,
            )
            demand_gdf, hulls_gdf = result if isinstance(result, tuple) else (result, None)

        if demand_gdf is None or len(demand_gdf) == 0:
            clusters_fc = {"type": "FeatureCollection", "features": []}
            cluster_hulls_fc = {"type": "FeatureCollection", "features": []}
        else:
            features = []
            source_gdf = hulls_gdf if (hulls_gdf is not None and len(hulls_gdf) > 0) else demand_gdf
            for idx, row in source_gdf.iterrows():
                geom = row.get("geometry")
                if geom is None or geom.is_empty:
                    continue
                try:
                    c = getattr(geom, "centroid", None) or geom
                    lon, lat = float(c.x), float(c.y)
                except Exception:
                    continue
                weight = int(row.get("weight", 1))
                cid = row.get("cluster_id")
                if cid is None or (isinstance(cid, float) and pd.isna(cid)):
                    cid = idx
                try:
                    cid = int(cid)
                except (TypeError, ValueError):
                    cid = hash(str(idx)) % 10_000_000
                props = {"weight": weight, "cluster_id": cid}
                nb = row.get("n_buildings")
                if nb is None or (isinstance(nb, float) and pd.isna(nb)):
                    nb = 1
                props["n_buildings"] = int(nb)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                        "properties": props,
                    }
                )
            clusters_fc = round_coords({"type": "FeatureCollection", "features": features}, precision=6)
            cluster_hulls_fc = {"type": "FeatureCollection", "features": []}
            if hulls_gdf is not None and len(hulls_gdf) > 0:
                hull_features = []
                for idx, row in hulls_gdf.iterrows():
                    g = row["geometry"]
                    if g is None or g.is_empty:
                        continue
                    try:
                        geom_json = json.loads(gpd.GeoSeries([g], crs="EPSG:4326").to_json())
                        geom_j = geom_json.get("features", [{}])[0].get("geometry")
                        if geom_j:
                            w = int(row.get("weight", 1))
                            _cid = row.get("cluster_id")
                            if _cid is None or (isinstance(_cid, float) and pd.isna(_cid)):
                                _cid = idx
                            try:
                                _cid = int(_cid)
                            except (TypeError, ValueError):
                                _cid = hash(str(idx)) % 10_000_000
                            props = {"weight": w, "cluster_id": _cid}
                            nb = row.get("n_buildings")
                            if nb is None or (isinstance(nb, float) and pd.isna(nb)):
                                nb = 1
                            props["n_buildings"] = int(nb)
                            hull_features.append(
                                {
                                    "type": "Feature",
                                    "geometry": round_coords(geom_j, precision=6),
                                    "properties": props,
                                }
                            )
                    except Exception:
                        pass
                cluster_hulls_fc = {"type": "FeatureCollection", "features": hull_features}

        buildings_fc = clusters_fc
        total = len(clusters_fc.get("features", []))
        logger.info("Кластеры: %s, hulls: %s", total, len(cluster_hulls_fc.get("features", [])))

        return JSONResponse(
            {
                "buildings": buildings_fc,
                "clusters": clusters_fc,
                "total": total,
                "method": method,
                "cluster_hulls": cluster_hulls_fc,
                "dbscan_eps_m": dbscan_eps_m,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/buildings/voronoi-local-paths")
def get_voronoi_local_paths(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
    dbscan_eps_m: float = Query(180.0, description="Радиус локального уровня (м)"),
    dbscan_min_samples: int = Query(15, description="min_samples DBSCAN локального уровня"),
    use_all_buildings: bool = Query(False, description="Все здания вне no-fly"),
    buildings_per_centroid: int = Query(60, description="Сколько зданий агрегировать в 1 центроид для Вороного"),
):
    """
    Локальные пути внутри кластера на основе соседства ячеек Вороного.
    Считается на сервере по cluster-данным из кэша DataService.
    """

    def _round_coords(obj, precision=6):
        if isinstance(obj, dict):
            return {k: _round_coords(v, precision) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_coords(item, precision) for item in obj]
        if isinstance(obj, float):
            return round(obj, precision)
        return obj

    try:
        raw_fc = _build_voronoi_local_paths_fc(
            city=city,
            network_type=network_type,
            simplify=simplify,
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
            use_all_buildings=use_all_buildings,
            buildings_per_centroid=buildings_per_centroid,
        )
        fc = _round_coords(raw_fc, precision=6)
        return JSONResponse(
            fc
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("voronoi local paths")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stations/placement")
def get_stations_placement(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
    demand_method: str = Query("dbscan", description="dbscan или grid"),
    dbscan_eps_m: float = Query(180.0, description="eps DBSCAN (м)"),
    dbscan_min_samples: int = Query(15, description="min_samples DBSCAN"),
    a_by_admin_districts: bool = Query(True, description="Ставить станции A по административным районам"),
    candidates_per_cluster: int = Query(25, description="Кандидатов на кластер при выборе точек зарядки"),
    use_saved: bool = Query(True, description="Использовать сохранённую расстановку, если есть"),
    only_saved: bool = Query(False, description="Только кэш Redis: при промахе 404, без пересчёта пайплайна"),
    save_result: bool = Query(True, description="Сохранять новую расстановку в Redis"),
    buildings_per_centroid: int = Query(60, description="Сколько зданий агрегировать в 1 центроид для Вороного"),
):
    """
    Временный режим: по кнопке «Рассчитать зарядки» строим только слой Вороного.
    """
    try:
        logger.info("Этап 2/5: Отрисовка локальных маршрутов Вороного")
        vor_fc = _build_voronoi_local_paths_fc(
            city,
            network_type=network_type,
            simplify=simplify,
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
            use_all_buildings=False,
            buildings_per_centroid=buildings_per_centroid,
            charging_station_features=None,
        )
        n_vor = len((vor_fc or {}).get("features", []) or [])
        logger.info("Этап 2/5 завершён: рёбер Вороного=%s", n_vor)
        empty_fc = {"type": "FeatureCollection", "features": []}
        geo = {
            "charging_type_a": empty_fc,
            "charging_type_b": empty_fc,
            "garages": empty_fc,
            "to_stations": empty_fc,
            "trunk": empty_fc,
            "branch_edges": empty_fc,
            "local_edges": empty_fc,
            "voronoi_edges": {"type": "FeatureCollection", "features": list((vor_fc or {}).get("features", []) or [])},
            "metrics": {
                "coverage_ratio": None,
                "cluster_count": int(vor_fc.get("clusters_total", 0) if isinstance(vor_fc, dict) else 0),
                "charging_buildings_in_clusters": None,
            },
            "params": {
                "local_paths_source": "voronoi_buildings_only",
                "station_voronoi_buildings_per_centroid": int(buildings_per_centroid),
                "stages_disabled_except_voronoi": True,
            },
            "cluster_centroids": empty_fc,
            "from_saved": False,
        }
        logger.info("Этап 5/5: Конец")
        return JSONResponse(geo)
    except Exception as e:
        logger.exception("stations placement")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stations/export_saved")
def export_saved_stations(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
    demand_method: str = Query("dbscan", description="dbscan или grid"),
    dbscan_eps_m: float = Query(180.0, description="eps DBSCAN (м)"),
    dbscan_min_samples: int = Query(15, description="min_samples DBSCAN"),
    a_by_admin_districts: bool = Query(True, description="Режим размещения A по районам/группам"),
    candidates_per_cluster: int = Query(25, description="Кандидатов на кластер при выборе точек зарядки"),
):
    """Выгрузка сохраненной расстановки станций из Redis в JSON."""
    redis_client = data_service.get_redis_client()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis недоступен")
    cache_key = _placement_cache_key(
        city=city,
        network_type=network_type,
        simplify=simplify,
        demand_method=demand_method,
        dbscan_eps_m=dbscan_eps_m,
        dbscan_min_samples=dbscan_min_samples,
        a_by_admin_districts=a_by_admin_districts,
        candidates_per_cluster=candidates_per_cluster,
    )
    try:
        cached = redis_client.get(cache_key)
        if not cached:
            raise HTTPException(status_code=404, detail="Сохраненная расстановка не найдена")
        data = json.loads(cached.decode("utf-8"))
        safe_city = city.replace(" ", "_").replace(",", "").replace("/", "_")
        filename = f"stations_{safe_city}.json"
        return JSONResponse(
            data,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка выгрузки: {e}")


def _ensure_static_mount():
    static_dir = os.path.join(os.getcwd(), "static")
    if os.path.isdir(static_dir):
        from fastapi.staticfiles import StaticFiles

        app.mount("/static", StaticFiles(directory=static_dir), name="static")


_ensure_static_mount()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
