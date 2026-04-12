"""
Локальные рёбра диаграммы Вороного по кластерам спроса.
Сайты — центроиды зданий (возможно агрегированные) и точки зарядных станций в кластере.
Площадь hull (м²) до порога 3_000_000: для не-МКД в однородном малом кластере — 3 здания на сайт;
МКД в малом кластере (только МКД или в смеси с другими) — 2 здания на сайт.
При площади hull строго больше 3_000_000: МКД — 5 зданий на один сайт; не-МКД — как раньше (buildings_per_centroid / 60).
Если площадь hull не больше порога и в кластере одновременно есть МКД и прочие типы зданий,
не-МКД — по buildings_per_centroid зданий на сайт (по умолчанию 60).
Сайты (центроиды), попадающие в no-fly, отбрасываются; для эшелонов 1–3 рёбра режутся отдельно:
на эшелоне k не летаем над зданием, если height_m ≥ (высота_эшелона_k − 5 м) — см. voronoi_by_echelon.
Станции стоят на крышах: если высота здания под станцией ≥ этому же порогу, все рёбра, касающиеся станции,
на этом эшелоне удаляются (union по упрощённым контурам мог не содержать точку станции).
NFZ учитывается на всех эшелонах (как в _filter_voronoi_fc_linestrings_nfz).
"""
from __future__ import annotations

import copy
from typing import Any

import geopandas as gpd
import numpy as np

import pandas as pd
import logging

from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

from data_service import DataService

logger = logging.getLogger(__name__)


def _merge_polygon_unions_wgs84(*parts: Any) -> Any:
    """Объединяет несколько shapely-геометрий (EPSG:4326); пустые пропускает."""
    geoms = [g for g in parts if g is not None and not getattr(g, "is_empty", True)]
    if not geoms:
        return None
    if len(geoms) == 1:
        return geoms[0]
    try:
        u = unary_union(geoms)
        return u if u is not None and not getattr(u, "is_empty", True) else None
    except Exception:
        return geoms[0]


def _voronoi_fc_copy_filter_and_bridges(
    fc_base: dict,
    *,
    nfz_union: Any,
    building_clearance_union: Any,
    coords_g: np.ndarray,
    flags_g: np.ndarray,
    all_cluster: list[Any],
    max_bridge_m: float,
    buildings_wgs: gpd.GeoDataFrame | None = None,
    voronoi_obstacle_min_building_height_m: float | None = None,
) -> dict:
    """Копия FC → NFZ (прежняя логика) → строгий фильтр по высотным зданиям → мосты → крыши станций."""
    fc = {
        "type": "FeatureCollection",
        "features": copy.deepcopy(list(fc_base.get("features") or [])),
    }
    fc = _filter_voronoi_fc_linestrings_nfz(fc, nfz_union)
    fc = _filter_voronoi_fc_building_clearance(fc, building_clearance_union)
    fc = _add_intra_cluster_nfz_safe_bridges(
        fc,
        coords_g,
        flags_g,
        all_cluster,
        nfz_union,
        building_clearance_union=building_clearance_union,
        max_bridge_m=float(max_bridge_m),
    )
    if (
        buildings_wgs is not None
        and len(buildings_wgs) > 0
        and voronoi_obstacle_min_building_height_m is not None
    ):
        try:
            thr = float(voronoi_obstacle_min_building_height_m)
        except (TypeError, ValueError):
            thr = None
        if thr is not None:
            fc = _filter_voronoi_fc_station_roof_echelon(fc, buildings_wgs, thr)
    return fc


# Hull площадью до этого порога (м², EPSG:3857) — без агрегации (1 здание = 1 сайт).
_VORONOI_BPC_SMALL_CLUSTER_MAX_AREA_M2 = 3_000_000.0
# Малый кластер (<= порога), однородные не-МКД: 3 здания на 1 сайт.
_VORONOI_BPC_SMALL_CLUSTER_BUILDINGS_PER_CENTROID = 3
# Малый кластер: МКД — 2 здания на 1 центроид (только МКД или в смешанном режиме).
_VORONOI_BPC_SMALL_CLUSTER_MKD_BUILDINGS_PER_CENTROID = 2
# Дополнительное уплотнение графа для небольших кластеров.
_SMALL_CLUSTER_EXTRA_KNN = 2
_SMALL_CLUSTER_EXTRA_EDGE_MAX_M = 450.0
_SMALL_CLUSTER_EXTRA_MAX_POINTS = 1500
# Если в запросе не задан buildings_per_centroid, для крупных кластеров используется это значение.
_VORONOI_BPC_LARGE_CLUSTER_FALLBACK = 60
# Кластеры с hull > _VORONOI_BPC_SMALL_CLUSTER_MAX_AREA_M2: только для МКД — 5 зданий на один сайт Вороного.
_VORONOI_BPC_LARGE_CLUSTER_MKD_BUILDINGS_PER_CENTROID = 5
# Мосты между компонентами графа внутри кластера (если включён фильтр по препятствиям — с проверкой пересечений).
_DEFAULT_VORONOI_INTRA_COMPONENT_BRIDGE_MAX_M = 600.0
_MAX_INTRA_BRIDGE_PAIR_EVAL = 80_000


def _cluster_id_prop_from_key(cid: str) -> Any:
    try:
        return int(cid)
    except ValueError:
        return cid


def _hull_area_m2(hull_poly: Any) -> float:
    """Площадь hull в м² (EPSG:3857)."""
    if hull_poly is None or getattr(hull_poly, "is_empty", True):
        return 0.0
    try:
        hp = hull_poly.buffer(0)
        h = gpd.GeoSeries([hp], crs="EPSG:4326").to_crs(3857)
        return float(h.geometry.iloc[0].area)
    except Exception:
        return 0.0


def _voronoi_effective_buildings_per_centroid(base_bpc: int, hull_poly: Any) -> int:
    """
    При площади hull ≤ порога — 3 здания на сайт для однородного не-МКД кластера
    (однородный только МКД — см. _VORONOI_BPC_SMALL_CLUSTER_MKD_BUILDINGS_PER_CENTROID).
    При площади hull > порога — для не-МКД: `base_bpc` зданий на сайт (или fallback 60);
    для МКД на большом hull — см. _VORONOI_BPC_LARGE_CLUSTER_MKD_BUILDINGS_PER_CENTROID в _voronoi_cluster_building_centroids.
    """
    area_m2 = _hull_area_m2(hull_poly)
    if area_m2 <= _VORONOI_BPC_SMALL_CLUSTER_MAX_AREA_M2:
        return _VORONOI_BPC_SMALL_CLUSTER_BUILDINGS_PER_CENTROID
    return max(1, int(base_bpc or _VORONOI_BPC_LARGE_CLUSTER_FALLBACK))


def _aggregate_points_to_centroids(pts_arr: np.ndarray, target_group_size: int) -> np.ndarray:
    """
    Сжимает набор точек [lon, lat] в центроиды групп (~target_group_size точек в группе).
    KMeans с запасным вариантом по лексикографическим чанкам (как в build_voronoi_local_paths_fc).
    """
    pts_arr = np.asarray(pts_arr, dtype=float)
    tg = max(1, int(target_group_size))
    if tg <= 1 or len(pts_arr) <= tg:
        return pts_arr
    n_groups = int(np.ceil(len(pts_arr) / float(tg)))
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
            return np.asarray(centers, dtype=float)
    except Exception:
        pass
    order = np.lexsort((pts_arr[:, 1], pts_arr[:, 0]))
    ordered = pts_arr[order]
    centers = []
    for i in range(0, len(ordered), tg):
        chunk = ordered[i : i + tg]
        if len(chunk) == 0:
            continue
        centers.append(chunk.mean(axis=0))
    if len(centers) >= 2:
        return np.asarray(centers, dtype=float)
    return pts_arr


def _voronoi_centroids_mixed_small_cluster(
    pts_mkd: list[list[float]],
    pts_non: list[list[float]],
    *,
    bpc_mkd: int,
    bpc_non: int,
) -> np.ndarray:
    """Объединённые сайты Вороного: МКД и не-МКД агрегируются с разным размером группы."""
    parts: list[np.ndarray] = []
    for bucket, bpc in ((pts_mkd, bpc_mkd), (pts_non, bpc_non)):
        arr = np.asarray(bucket, dtype=float) if bucket else np.empty((0, 2), dtype=float)
        if len(arr) == 0:
            continue
        parts.append(_aggregate_points_to_centroids(arr, max(1, int(bpc))))
    if not parts:
        return np.empty((0, 2), dtype=float)
    if len(parts) == 1:
        return parts[0]
    return np.vstack(parts)


def _small_cluster_mixed_mkd_nonmkd(
    hull_poly: Any,
    pts_mkd: list[list[float]],
    pts_non: list[list[float]],
) -> bool:
    """Площадь hull ≤ порогу и в кластере после фильтра NFZ есть и МКД, и другие здания."""
    if _hull_area_m2(hull_poly) > _VORONOI_BPC_SMALL_CLUSTER_MAX_AREA_M2:
        return False
    return len(pts_mkd) > 0 and len(pts_non) > 0


def _voronoi_cluster_building_centroids(
    hull_poly: Any,
    pts_mkd: list[list[float]],
    pts_non: list[list[float]],
    buildings_per_centroid: int,
) -> np.ndarray:
    """
    Центроиды сайтов Вороного для одного кластера (МКД / не-МКД раздельно где нужно).
    Для hull > 3_000_000 м²: МКД — 5:1, не-МКД — прежняя крупнокластерная агрегация (BPC / 60).
    """
    if _small_cluster_mixed_mkd_nonmkd(hull_poly, pts_mkd, pts_non):
        return _voronoi_centroids_mixed_small_cluster(
            pts_mkd,
            pts_non,
            bpc_mkd=_VORONOI_BPC_SMALL_CLUSTER_MKD_BUILDINGS_PER_CENTROID,
            bpc_non=max(1, int(buildings_per_centroid or _VORONOI_BPC_LARGE_CLUSTER_FALLBACK)),
        )

    area_m2 = _hull_area_m2(hull_poly)
    n_mkd, n_non = len(pts_mkd), len(pts_non)

    if area_m2 > _VORONOI_BPC_SMALL_CLUSTER_MAX_AREA_M2:
        bpc_non = max(1, _voronoi_effective_buildings_per_centroid(buildings_per_centroid, hull_poly))
        if n_mkd > 0 and n_non > 0:
            return _voronoi_centroids_mixed_small_cluster(
                pts_mkd,
                pts_non,
                bpc_mkd=_VORONOI_BPC_LARGE_CLUSTER_MKD_BUILDINGS_PER_CENTROID,
                bpc_non=bpc_non,
            )
        if n_mkd > 0 and n_non == 0:
            return _aggregate_points_to_centroids(
                np.asarray(pts_mkd, dtype=float),
                _VORONOI_BPC_LARGE_CLUSTER_MKD_BUILDINGS_PER_CENTROID,
            )
        pts_all = np.asarray(pts_non, dtype=float)
        return _aggregate_points_to_centroids(pts_all, bpc_non)

    target_group_size = max(
        1,
        _voronoi_effective_buildings_per_centroid(buildings_per_centroid, hull_poly),
    )
    if (
        area_m2 <= _VORONOI_BPC_SMALL_CLUSTER_MAX_AREA_M2
        and n_non == 0
        and n_mkd >= 2
    ):
        target_group_size = max(1, _VORONOI_BPC_SMALL_CLUSTER_MKD_BUILDINGS_PER_CENTROID)
    pts_arr = np.asarray(pts_mkd + pts_non, dtype=float)
    return _aggregate_points_to_centroids(pts_arr, target_group_size)


def _normalize_hulls_gdf(hulls_gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    if hulls_gdf is None or len(hulls_gdf) == 0:
        return None
    h = hulls_gdf.to_crs("EPSG:4326").copy()
    if "cluster_id" not in h.columns:
        if "subcluster_id" in h.columns:
            h["cluster_id"] = h["subcluster_id"]
        else:
            return None
    h = h[h["cluster_id"].notna()].copy()
    return h if len(h) > 0 else None


def _hull_polygon_for_cluster(hulls_norm: gpd.GeoDataFrame | None, cluster_id: Any):
    if hulls_norm is None or len(hulls_norm) == 0:
        return None
    try:
        sub = hulls_norm[hulls_norm["cluster_id"] == cluster_id]
        if len(sub) == 0:
            sub = hulls_norm[hulls_norm["cluster_id"].astype(str) == str(cluster_id)]
        if len(sub) == 0:
            return None
        g = sub.geometry.iloc[0]
        if g is None or getattr(g, "is_empty", True):
            return None
        return g
    except Exception:
        return None


def _point_in_hull(lon: float, lat: float, hull_poly) -> bool:
    if hull_poly is None or getattr(hull_poly, "is_empty", True):
        return True
    try:
        p = Point(float(lon), float(lat))
        return bool(hull_poly.covers(p) or hull_poly.contains(p))
    except Exception:
        return False


def _clip_segment_to_hull_coords(
    lon1: float, lat1: float, lon2: float, lat2: float, hull_poly
) -> list[list[float]] | None:
    """Обрезает отрезок между сайтами по полигону кластера (EPSG:4326)."""
    if hull_poly is None or getattr(hull_poly, "is_empty", True):
        return [[lon1, lat1], [lon2, lat2]]
    try:
        line = LineString([(lon1, lat1), (lon2, lat2)])
        inter = line.intersection(hull_poly.buffer(0))
        if inter.is_empty:
            return None
        if inter.geom_type == "LineString":
            return [[float(x), float(y)] for x, y in inter.coords]
        if inter.geom_type == "MultiLineString":
            best = max(inter.geoms, key=lambda gg: gg.length)
            return [[float(x), float(y)] for x, y in best.coords]
        return None
    except Exception:
        return [[lon1, lat1], [lon2, lat2]]


def _filter_station_features_to_hulls(
    features: list | None,
    hulls_norm: gpd.GeoDataFrame | None,
) -> list[dict]:
    """Оставляет только точки станций, геометрически попадающие в hull своего cluster_id."""
    if not features:
        return []
    if hulls_norm is None or len(hulls_norm) == 0:
        return [f for f in features if isinstance(f, dict)]
    out: list[dict] = []
    for f in features:
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
        lon, lat = float(coords[0]), float(coords[1])
        hp = _hull_polygon_for_cluster(hulls_norm, cid)
        if hp is not None and not _point_in_hull(lon, lat, hp):
            continue
        out.append(f)
    return out


def charging_station_gdfs_to_features(*layers) -> list[dict]:
    """
    Точки зарядок / гаражей / ТО с привязкой к кластеру для построения Вороного вместе со спросом.
    Берётся cluster_id; если нет — subcluster_id.
    """
    out: list[dict] = []
    for gdf in layers:
        if gdf is None or len(gdf) == 0:
            continue
        for _, row in gdf.iterrows():
            g = row.geometry
            if g is None or getattr(g, "is_empty", True):
                continue
            cid = row.get("cluster_id")
            if cid is None or (isinstance(cid, float) and pd.isna(cid)):
                cid = row.get("subcluster_id")
            if cid is None or (isinstance(cid, float) and pd.isna(cid)):
                continue
            try:
                lon, lat = float(g.x), float(g.y)
            except Exception:
                continue
            try:
                cid_val = int(cid)
            except (TypeError, ValueError):
                cid_val = str(cid)
            out.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"cluster_id": cid_val},
                }
            )
    return out


def _no_fly_obstacles_union(no_fly_zones: Any) -> Any:
    """Единый полигон/MultiPolygon no-fly для проверок и маршрутизации (EPSG:4326)."""
    if no_fly_zones is None:
        return None
    try:
        if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
            nz = no_fly_zones.to_crs("EPSG:4326")
            geoms = [g for g in nz.geometry if g is not None and not getattr(g, "is_empty", True)]
            if not geoms:
                return None
            return unary_union(geoms)
        if isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
            geoms = [g for g in no_fly_zones if g is not None and not getattr(g, "is_empty", True)]
            if not geoms:
                return None
            return unary_union(geoms)
    except Exception:
        return None
    return None


def _point_in_nfz_union(lon: float, lat: float, nfz_union: Any) -> bool:
    if nfz_union is None or getattr(nfz_union, "is_empty", True):
        return False
    try:
        return bool(nfz_union.contains(Point(float(lon), float(lat))))
    except Exception:
        return False


def _filter_xy_rows_outside_nfz(
    pts_arr: np.ndarray,
    is_station: np.ndarray | None,
    nfz_union: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Удаляет строки сайтов, чей центроид попадает внутрь nfz_union."""
    if nfz_union is None or len(pts_arr) == 0:
        return pts_arr, is_station
    keep_rows: list[int] = []
    for i in range(len(pts_arr)):
        lo, la = float(pts_arr[i, 0]), float(pts_arr[i, 1])
        if _point_in_nfz_union(lo, la, nfz_union):
            continue
        keep_rows.append(i)
    if not keep_rows:
        return np.empty((0, 2), dtype=float), (
            np.empty(0, dtype=bool) if is_station is not None else None
        )
    idx = np.array(keep_rows, dtype=int)
    out_pts = pts_arr[idx]
    out_st = is_station[idx] if is_station is not None and len(is_station) == len(pts_arr) else is_station
    return out_pts, out_st


def _line_violates_building_clearance(line: LineString, building_union: Any, *, interior_samples: int = 7) -> bool:
    """
    Строгая проверка для «зданий под эшелон»: нельзя заканчивать ребро внутри контура,
    нельзя пересекать контур по дуге ненулевой длины, проверяются точки вдоль отрезка
    (устраняет баг, когда пересечение деградирует в Point и ребро ошибочно оставляли).
    """
    if building_union is None or getattr(building_union, "is_empty", True):
        return False
    eps = 1e-7
    try:
        prepared = prep(building_union)
    except Exception:
        prepared = None
    try:
        for xy in (line.coords[0], line.coords[-1]):
            pt = Point(float(xy[0]), float(xy[1]))
            try:
                if building_union.covers(pt):
                    return True
            except Exception:
                if prepared is not None and prepared.contains(pt):
                    return True
        n = max(2, int(interior_samples))
        for k in range(1, n):
            pt = line.interpolate(k / float(n), normalized=True)
            try:
                if building_union.contains(pt) or building_union.covers(pt):
                    return True
            except Exception:
                if prepared is not None and prepared.contains(pt):
                    return True
        inter = building_union.intersection(line)
        if inter.is_empty:
            return False
        if inter.geom_type == "LineString" and float(inter.length) > eps:
            return True
        if inter.geom_type == "MultiLineString":
            return any(float(g.length) > eps for g in inter.geoms)
        if inter.geom_type == "GeometryCollection":
            for g in getattr(inter, "geoms", []):
                gt = getattr(g, "geom_type", "")
                if gt == "LineString" and float(g.length) > eps:
                    return True
                if gt == "MultiLineString":
                    if any(float(gg.length) > eps for gg in g.geoms):
                        return True
        return False
    except Exception:
        return True


def _filter_voronoi_fc_building_clearance(fc: dict, building_union: Any) -> dict:
    """Удаляет рёбра LineString, нарушающие ограничение по высотным зданиям (см. _line_violates_building_clearance)."""
    if building_union is None or getattr(building_union, "is_empty", True):
        return fc
    feats = list(fc.get("features", []) or [])
    kept: list[dict] = []
    for feat in feats:
        if not isinstance(feat, dict):
            kept.append(feat)
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            kept.append(feat)
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        try:
            line_coords = [(float(c[0]), float(c[1])) for c in coords if len(c) >= 2]
            if len(line_coords) < 2:
                continue
            line = LineString(line_coords)
        except Exception:
            kept.append(feat)
            continue
        if _line_violates_building_clearance(line, building_union):
            continue
        kept.append(feat)
    out = dict(fc)
    out["features"] = kept
    return out


def _batch_max_building_heights_m_at_lonlat(
    lonlats: list[tuple[float, float]],
    buildings_wgs: gpd.GeoDataFrame,
) -> dict[tuple[float, float], float]:
    """
    Для каждой уникальной точки (lon, lat) в WGS84 — max height_m среди зданий, с которыми пересекается точка.
    Используется intersects (не только within), чтобы точки на границе контура крыши не терялись.
    """
    out: dict[tuple[float, float], float] = {}
    if not lonlats or buildings_wgs is None or len(buildings_wgs) == 0:
        return out
    try:
        b = buildings_wgs
        if "height_m" not in b.columns:
            return out
        bg = b[["geometry", "height_m"]].copy()
        bg = bg[bg.geometry.notna() & (~bg.geometry.is_empty)]
        if len(bg) == 0:
            return out
    except Exception:
        return out

    uniq: dict[tuple[float, float], tuple[float, float]] = {}
    for lo, la in lonlats:
        k = (round(float(lo), 6), round(float(la), 6))
        uniq.setdefault(k, (float(lo), float(la)))
    keys_list = list(uniq.keys())
    try:
        pts = gpd.GeoDataFrame(
            {"qid": range(len(keys_list))},
            geometry=[
                Point(float(uniq[k][0]), float(uniq[k][1])) for k in keys_list
            ],
            crs="EPSG:4326",
        )
        try:
            j = pts.sjoin(bg, how="left", predicate="intersects")
        except TypeError:
            j = pts.sjoin(bg, how="left", op="intersects")
    except Exception:
        return out

    if "height_m" not in j.columns:
        for k in keys_list:
            out[k] = 0.0
        return out

    j["height_m"] = pd.to_numeric(j["height_m"], errors="coerce")
    for qid, grp in j.groupby("qid"):
        qi = int(qid)
        if qi < 0 or qi >= len(keys_list):
            continue
        rk = keys_list[qi]
        hs = grp["height_m"].dropna()
        out[rk] = float(np.nanmax(hs.to_numpy(dtype=float))) if len(hs) > 0 else 0.0
    for i, k in enumerate(keys_list):
        if k not in out:
            out[k] = 0.0
    return out


def _filter_voronoi_fc_station_roof_echelon(
    fc: dict,
    buildings_wgs: gpd.GeoDataFrame,
    min_obstacle_height_m: float,
) -> dict:
    """
    Удаляет рёбра, инцидентные зарядной станции на крыше здания высотой ≥ min_obstacle_height_m
    (тот же порог, что и для препятствий на эшелоне: высота_эшелона − 5 м).
    """
    try:
        mh = float(min_obstacle_height_m)
    except (TypeError, ValueError):
        return fc
    feats = list(fc.get("features") or [])
    if not feats or buildings_wgs is None or len(buildings_wgs) == 0:
        return fc

    need_pts: list[tuple[float, float]] = []
    for feat in feats:
        if not isinstance(feat, dict):
            continue
        p = feat.get("properties") or {}
        if not p.get("connects_station"):
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            continue
        cr = geom.get("coordinates") or []
        if len(cr) < 2:
            continue
        try:
            if p.get("source_is_station"):
                need_pts.append((float(cr[0][0]), float(cr[0][1])))
            if p.get("target_is_station"):
                need_pts.append((float(cr[-1][0]), float(cr[-1][1])))
        except (TypeError, ValueError, IndexError):
            continue

    hmap = _batch_max_building_heights_m_at_lonlat(need_pts, buildings_wgs)

    def _h(lo: float, la: float) -> float:
        return float(hmap.get((round(float(lo), 6), round(float(la), 6)), 0.0))

    kept: list[dict] = []
    for feat in feats:
        if not isinstance(feat, dict):
            kept.append(feat)
            continue
        p = feat.get("properties") or {}
        if not p.get("connects_station"):
            kept.append(feat)
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            kept.append(feat)
            continue
        cr = geom.get("coordinates") or []
        if len(cr) < 2:
            continue
        try:
            drop = False
            if p.get("source_is_station"):
                if _h(float(cr[0][0]), float(cr[0][1])) >= mh:
                    drop = True
            if not drop and p.get("target_is_station"):
                if _h(float(cr[-1][0]), float(cr[-1][1])) >= mh:
                    drop = True
        except (TypeError, ValueError, IndexError):
            kept.append(feat)
            continue
        if drop:
            continue
        kept.append(feat)

    out = dict(fc)
    out["features"] = kept
    return out


def _filter_voronoi_fc_linestrings_nfz(fc: dict, nfz_union: Any) -> dict:
    """Удаляет рёбра LineString, пересекающие внутренность no-fly (is_edge_blocked)."""
    if nfz_union is None or getattr(nfz_union, "is_empty", True):
        return fc
    eps = 1e-9
    try:
        nfz_prepared = prep(nfz_union)
    except Exception:
        nfz_prepared = None

    feats = list(fc.get("features", []) or [])
    kept: list[dict] = []
    for feat in feats:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            kept.append(feat)
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        try:
            line_coords = []
            for c in coords:
                if len(c) >= 2:
                    line_coords.append((float(c[0]), float(c[1])))
            if len(line_coords) < 2:
                continue
            line = LineString(line_coords)
        except Exception:
            kept.append(feat)
            continue
        try:
            if nfz_prepared is not None and not nfz_prepared.intersects(line):
                kept.append(feat)
                continue
            inter = nfz_union.intersection(line)
            if inter.is_empty:
                kept.append(feat)
                continue
            # Касание границы точкой считаем допустимым.
            if inter.geom_type in ("Point", "MultiPoint"):
                kept.append(feat)
                continue
            if inter.geom_type == "GeometryCollection":
                parts = list(getattr(inter, "geoms", []))
                if parts and all(
                    (
                        p.geom_type in ("Point", "MultiPoint")
                        or getattr(p, "length", 0.0) <= eps
                    )
                    for p in parts
                ):
                    kept.append(feat)
                    continue
            # Любая нетривиальная линейная/площадная часть внутри NFZ — блокируем.
            if getattr(inter, "length", 0.0) > eps or getattr(inter, "area", 0.0) > eps:
                continue
            continue
        except Exception:
            # Консервативно оставляем ребро при ошибке геометрии.
            kept.append(feat)
            continue
    out = dict(fc)
    out["features"] = kept
    return out


def apply_nfz_detours_to_voronoi_fc(
    fc: dict,
    *,
    no_fly_zones: Any,
    city_boundary: Any,
) -> dict:
    """
    Для **межкластерных** рёбер (properties.inter_cluster=True), пересекающих no-fly, строит обход по контуру зоны.
    A* не используется. Внутрикластерные рёбра не трогаем — остаются прямыми отрезками Вороного, даже через зону.
    """
    if not fc or not isinstance(fc, dict):
        return fc
    feats = list(fc.get("features", []) or [])
    if not feats:
        return fc

    obstacles = _no_fly_obstacles_union(no_fly_zones)
    if obstacles is None or getattr(obstacles, "is_empty", True):
        return fc

    # Ленивый импорт: избегаем циклических зависимостей с station_placement.
    from station_placement import (
        is_edge_blocked,
        prepare_air_detour_auxiliary,
        route_lonlat_segment_with_nfz_detours,
    )

    try:
        if city_boundary is not None and not getattr(city_boundary, "is_empty", True):
            b = city_boundary.bounds
            utm_anchor = (float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0))
        else:
            b = obstacles.bounds
            utm_anchor = (float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0))
    except Exception:
        utm_anchor = (37.6173, 55.7558)

    obstacles_buf = obstacles

    # Без воздушного графа: в route_lonlat_segment_with_nfz_detours не вызывается A* (только boundary-обход).
    _nfz_u, nfz_metric_pair, _air_w, _air_c, _air_i, _air_t = prepare_air_detour_auxiliary(
        obstacles_buf,
        None,
        utm_anchor,
        no_fly_safety_buffer=0.0,
    )

    detour_counter: list[int] = [0]
    n_cross = 0
    n_ok = 0
    n_fail = 0

    for idx, feat in enumerate(feats):
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props_early = feat.get("properties") or {}
        # Локальные (внутрикластерные) рёбра: без обходов — только прямая геометрия Вороного.
        if not props_early.get("inter_cluster"):
            continue
        try:
            line = LineString([(float(c[0]), float(c[1])) for c in coords if len(c) >= 2])
        except Exception:
            continue
        if line.is_empty:
            continue
        if not is_edge_blocked(line, obstacles_buf, safety_buffer=0.0):
            continue

        n_cross += 1
        lon1, lat1 = float(coords[0][0]), float(coords[0][1])
        lon2, lat2 = float(coords[-1][0]), float(coords[-1][1])
        lbl = "vor_edge_%s" % idx
        routed = route_lonlat_segment_with_nfz_detours(
            lon1,
            lat1,
            lon2,
            lat2,
            obstacles_buf,
            air_graph=None,
            air_coords=None,
            air_ids=None,
            air_tree=None,
            no_fly_safety_buffer=0.0,
            nfz_metric=nfz_metric_pair,
            utm_anchor_lonlat=utm_anchor,
            logger=logger,
            detour_counter=detour_counter,
            log_edge_label=lbl,
        )
        props = dict(feat.get("properties") or {})
        if routed is None:
            n_fail += 1
            props["nfz_detour_failed"] = True
            feat["properties"] = props
            continue
        path_ll, _len_km, how = routed
        if how != "straight" and path_ll and len(path_ll) >= 2:
            feat["geometry"] = {
                "type": "LineString",
                "coordinates": [[float(lon), float(lat)] for lon, lat in path_ll],
            }
            props["nfz_route"] = how
            n_ok += 1
        feat["properties"] = props

    out = dict(fc)
    out["features"] = feats
    out["nfz_voronoi_edges_crossing"] = int(n_cross)
    out["nfz_voronoi_detours_ok"] = int(n_ok)
    out["nfz_voronoi_detours_failed"] = int(n_fail)
    return out


def dedupe_close_parallel_voronoi_edges(
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
                nx, ny = -k["uy"], k["ux"]
                perp_cur = abs((cur["mx"] - k["mx"]) * nx + (cur["my"] - k["my"]) * ny)
                if perp_cur > (max_midpoint_distance_m * 0.7):
                    continue

                tx, ty = k["ux"], k["uy"]
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


def _require_voronoi():
    try:
        from scipy.spatial import Voronoi

        return Voronoi
    except Exception as e:
        raise RuntimeError(f"Scipy Voronoi недоступен: {e}") from e


def merge_xy_with_station_flags(
    building_xy: np.ndarray,
    station_xy: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Объединяет точки центроидов и станций; при совпадении координат сайт считается станцией (логическое ИЛИ).
    """
    rows: list[tuple[float, float, bool]] = []
    if building_xy is not None and len(building_xy) > 0:
        b = np.asarray(building_xy, dtype=float)
        for i in range(len(b)):
            rows.append((float(b[i, 0]), float(b[i, 1]), False))
    if station_xy is not None and len(station_xy) > 0:
        s = np.asarray(station_xy, dtype=float)
        for i in range(len(s)):
            rows.append((float(s[i, 0]), float(s[i, 1]), True))
    if not rows:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=bool)
    merged: dict[tuple[float, float], bool] = {}
    order_keys: list[tuple[float, float]] = []
    for x, y, is_st in rows:
        key = (round(x, 6), round(y, 6))
        if key not in merged:
            merged[key] = is_st
            order_keys.append(key)
        else:
            merged[key] = merged[key] or is_st
    pts = np.array([[k[0], k[1]] for k in order_keys], dtype=float)
    flags = np.array([merged[k] for k in order_keys], dtype=bool)
    return pts, flags


def _station_edge_props(
    cluster_prop: Any,
    edge_type: str,
    a: int,
    b: int,
    is_station: np.ndarray,
    *,
    group_mode: str | None = None,
    source_i: int | None = None,
    target_i: int | None = None,
) -> dict:
    sa = bool(is_station[a]) if a < len(is_station) else False
    sb = bool(is_station[b]) if b < len(is_station) else False
    props: dict[str, Any] = {
        "cluster_id": cluster_prop,
        "edge_type": edge_type,
        "source_i": int(source_i if source_i is not None else a),
        "target_i": int(target_i if target_i is not None else b),
        "source_is_station": sa,
        "target_is_station": sb,
        "connects_station": sa or sb,
    }
    if group_mode is not None:
        props["group_mode"] = group_mode
    return props


def _append_voronoi_edges_for_cluster(
    features: list[dict],
    pts_arr: np.ndarray,
    is_station: np.ndarray,
    group_id: Any,
    edge_type: str,
    *,
    group_mode: str | None = None,
    hull_poly=None,
) -> bool:
    """Добавляет рёбра Вороного для одного кластера (только внутри hull_poly, если задан)."""
    Voronoi = _require_voronoi()
    n0 = len(features)
    cid = int(group_id) if isinstance(group_id, (int, np.integer)) else str(group_id)

    if len(pts_arr) < 2:
        return False

    if len(pts_arr) == 2:
        p0, p1 = pts_arr[0], pts_arr[1]
        clipped = _clip_segment_to_hull_coords(
            float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), hull_poly
        )
        if clipped is None or len(clipped) < 2:
            return False
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": clipped,
                },
                "properties": {
                    **_station_edge_props(cid, edge_type, 0, 1, is_station, group_mode=group_mode),
                    "inter_cluster": False,
                },
            }
        )
        return True

    coords = np.asarray(pts_arr, dtype=float)
    flags = np.asarray(is_station, dtype=bool)
    if len(flags) != len(coords):
        flags = np.zeros(len(coords), dtype=bool)

    map_idx: dict[int, int] = {i: i for i in range(len(coords))}
    try:
        vor = Voronoi(coords)
    except Exception:
        uniq, inv = np.unique(coords, axis=0, return_inverse=True)
        if len(uniq) < 3:
            return False
        fu = np.zeros(len(uniq), dtype=bool)
        for k in range(len(flags)):
            fu[inv[k]] |= flags[k]
        coords = uniq
        flags = fu
        map_idx = {i: i for i in range(len(coords))}
        try:
            vor = Voronoi(coords)
        except Exception:
            return False

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
        pi = coords[a]
        pj = coords[b]
        clipped = _clip_segment_to_hull_coords(
            float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1]), hull_poly
        )
        if clipped is None or len(clipped) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": clipped,
                },
                "properties": {
                    **_station_edge_props(cid, edge_type, a, b, flags, group_mode=group_mode),
                    "inter_cluster": False,
                },
            }
        )
    return len(features) > n0


def _cluster_label_for_props(cid: Any) -> Any:
    if isinstance(cid, (int, np.integer)):
        return int(cid)
    return str(cid)


def _inter_cluster_pair_label(cid_a: Any, cid_b: Any) -> str:
    sa, sb = sorted((str(cid_a), str(cid_b)))
    return f"{sa}|{sb}"


def _canonical_cluster_pair_key(cid_a: Any, cid_b: Any) -> tuple[str, str]:
    return tuple(sorted((str(cid_a), str(cid_b))))


def _segment_length_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Длина отрезка в метрах (локально equirectangular по средней широте)."""
    lat0 = 0.5 * (lat1 + lat2)
    m_lat = 111_320.0
    m_lon = 111_320.0 * float(np.cos(np.radians(lat0)))
    return float(np.hypot((lon2 - lon1) * m_lon, (lat2 - lat1) * m_lat))


def _inter_cluster_pairs_present(features: list[dict]) -> set[tuple[str, str]]:
    """Уже есть хотя бы одно межкластерное ребро между этой парой cluster_id."""
    out: set[tuple[str, str]] = set()
    for f in features:
        p = f.get("properties") or {}
        if not p.get("inter_cluster"):
            continue
        cid = p.get("cluster_id")
        if isinstance(cid, str) and "|" in cid:
            parts = cid.split("|")
            if len(parts) == 2:
                out.add(tuple(sorted((parts[0], parts[1]))))
    return out


def _add_shortest_bridges_for_allowed_pairs(
    features: list[dict],
    coords: np.ndarray,
    flags: np.ndarray,
    cluster_per_point: list[Any],
    allowed_pair_keys: set[tuple[str, str]],
    max_edge_m: float,
) -> None:
    """
    Если для пары кластеров из allowed_pair_keys Delaunay не дал ни одного ребра,
    добавляет отрезок между ближайшими сайтами (длина <= max_edge_m).
    """
    if max_edge_m <= 0 or not allowed_pair_keys:
        return
    by_c: dict[str, list[int]] = {}
    for i, c in enumerate(cluster_per_point):
        key = str(c)
        by_c.setdefault(key, []).append(i)
    have = _inter_cluster_pairs_present(features)
    for a, b in allowed_pair_keys:
        pk = tuple(sorted((a, b)))
        if pk in have:
            continue
        ia = by_c.get(a, [])
        ib = by_c.get(b, [])
        if not ia or not ib:
            continue
        best_i, best_j = -1, -1
        best_d = float("inf")
        for i in ia:
            pi = coords[i]
            for j in ib:
                pj = coords[j]
                d = _segment_length_m(float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1]))
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i < 0 or best_d > max_edge_m:
            continue
        pi = coords[best_i]
        pj = coords[best_j]
        ca, cb = cluster_per_point[best_i], cluster_per_point[best_j]
        cid_prop = _inter_cluster_pair_label(ca, cb)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(pi[0]), float(pi[1])],
                        [float(pj[0]), float(pj[1])],
                    ],
                },
                "properties": {
                    **_station_edge_props(
                        cid_prop,
                        "voronoi_local",
                        best_i,
                        best_j,
                        flags,
                        group_mode=None,
                    ),
                    "inter_cluster": True,
                    "bridge_gap_fill": True,
                },
            }
        )
        have.add(pk)


def _iter_intra_bridge_candidate_pairs(
    C1: set[int],
    C2: set[int],
    coords: np.ndarray,
) -> list[tuple[int, int]]:
    """Пары индексов между двумя компонентами; при переборе слишком большом — только k ближайших к каждой точке меньшей компоненты."""
    if not C1 or not C2:
        return []
    if len(C1) > len(C2):
        C1, C2 = C2, C1
    n1, n2 = len(C1), len(C2)
    if n1 * n2 <= _MAX_INTRA_BRIDGE_PAIR_EVAL:
        return [(i, j) for i in C1 for j in C2]
    k = max(12, _MAX_INTRA_BRIDGE_PAIR_EVAL // max(n2, 1))
    out: list[tuple[int, int]] = []
    list2 = list(C2)
    for i in C1:
        pi = coords[i]
        dists: list[tuple[float, int]] = []
        for j in list2:
            pj = coords[j]
            d = _segment_length_m(
                float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1])
            )
            dists.append((d, j))
        dists.sort(key=lambda t: t[0])
        for _, j in dists[:k]:
            out.append((i, j))
    return out


def _add_intra_cluster_nfz_safe_bridges(
    fc: dict,
    coords: np.ndarray,
    flags: np.ndarray,
    cluster_per_point: list[Any],
    nfz_union: Any,
    *,
    building_clearance_union: Any = None,
    max_bridge_m: float,
) -> dict:
    """
    Если после фильтров внутри одного кластера граф распался на компоненты,
    добавляет кратчайшие отрезки, не пересекающие NFZ (is_edge_blocked) и не нарушающие высотные здания.
    """
    if max_bridge_m <= 0:
        out = dict(fc)
        out["voronoi_intra_component_bridges_added"] = 0
        return out
    nfz_empty = nfz_union is None or getattr(nfz_union, "is_empty", True)
    bld_empty = building_clearance_union is None or getattr(building_clearance_union, "is_empty", True)
    if nfz_empty and bld_empty:
        out = dict(fc)
        out.setdefault("voronoi_intra_component_bridges_added", 0)
        return out
    from station_placement import is_edge_blocked

    feats = list(fc.get("features", []) or [])
    n = len(coords)
    if n < 2 or len(cluster_per_point) != n:
        out = dict(fc)
        out.setdefault("voronoi_intra_component_bridges_added", 0)
        return out
    fl = np.asarray(flags, dtype=bool)
    if len(fl) != n:
        fl = np.zeros(n, dtype=bool)

    intra_pairs: set[tuple[int, int]] = set()
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for feat in feats:
        p = feat.get("properties") or {}
        if p.get("inter_cluster"):
            continue
        si = p.get("source_i")
        ti = p.get("target_i")
        if si is None or ti is None:
            continue
        a, b = int(si), int(ti)
        if a < 0 or b < 0 or a >= n or b >= n or a == b:
            continue
        u, v = (a, b) if a < b else (b, a)
        intra_pairs.add((u, v))
        adj[a].add(b)
        adj[b].add(a)

    by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(cluster_per_point):
        by_cluster.setdefault(str(c), []).append(i)

    added = 0
    for cid, verts in by_cluster.items():
        if len(verts) < 2:
            continue
        verts_set = set(verts)
        visited: set[int] = set()
        components: list[set[int]] = []
        for v in verts:
            if v in visited:
                continue
            stack = [v]
            comp: set[int] = set()
            while stack:
                x = stack.pop()
                if x in visited or x not in verts_set:
                    continue
                visited.add(x)
                comp.add(x)
                for y in adj.get(x, ()):
                    if y in verts_set and y not in visited:
                        stack.append(y)
            if comp:
                components.append(comp)
        if len(components) <= 1:
            continue

        comps = [set(c) for c in components]
        cid_prop = _cluster_id_prop_from_key(cid)

        while len(comps) > 1:
            best: tuple[float, int, int, int, int] | None = None
            for ci in range(len(comps)):
                for cj in range(ci + 1, len(comps)):
                    for i, j in _iter_intra_bridge_candidate_pairs(comps[ci], comps[cj], coords):
                        u, v = (i, j) if i < j else (j, i)
                        if (u, v) in intra_pairs:
                            continue
                        pi, pj = coords[i], coords[j]
                        d = _segment_length_m(
                            float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1])
                        )
                        if d > max_bridge_m:
                            continue
                        line = LineString(
                            [
                                (float(pi[0]), float(pi[1])),
                                (float(pj[0]), float(pj[1])),
                            ]
                        )
                        if not nfz_empty and is_edge_blocked(line, nfz_union, safety_buffer=0.0):
                            continue
                        if not bld_empty and _line_violates_building_clearance(line, building_clearance_union):
                            continue
                        if best is None or d < best[0]:
                            best = (d, i, j, ci, cj)
            if best is None:
                break
            _d, i, j, ci, cj = best
            u, v = (i, j) if i < j else (j, i)
            intra_pairs.add((u, v))
            adj[i].add(j)
            adj[j].add(i)
            merged = comps[ci] | comps[cj]
            hi, lo = (ci, cj) if ci > cj else (cj, ci)
            comps.pop(hi)
            comps.pop(lo)
            comps.append(merged)

            pi, pj = coords[i], coords[j]
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [float(pi[0]), float(pi[1])],
                            [float(pj[0]), float(pj[1])],
                        ],
                    },
                    "properties": {
                        **_station_edge_props(
                            cid_prop,
                            "voronoi_local",
                            i,
                            j,
                            fl,
                            group_mode=None,
                        ),
                        "inter_cluster": False,
                        "intra_component_bridge": True,
                    },
                }
            )
            added += 1

    out = dict(fc)
    out["features"] = feats
    out["voronoi_intra_component_bridges_added"] = int(added)
    return out


def _add_extra_intra_edges_for_small_clusters(
    fc: dict,
    coords: np.ndarray,
    flags: np.ndarray,
    cluster_per_point: list[Any],
    hulls_norm: gpd.GeoDataFrame | None,
    *,
    k_neighbors: int = _SMALL_CLUSTER_EXTRA_KNN,
    max_edge_m: float = _SMALL_CLUSTER_EXTRA_EDGE_MAX_M,
) -> dict:
    """
    Добавляет короткие дополнительные рёбра внутри небольших кластеров (по k ближайшим соседям),
    чтобы повысить локальную связность там, где чистого Voronoi недостаточно.
    """
    if not fc or not isinstance(fc, dict):
        return {"type": "FeatureCollection", "features": []}
    feats = list(fc.get("features", []) or [])
    n = len(coords)
    if n < 3 or len(cluster_per_point) != n:
        return {"type": "FeatureCollection", "features": feats}
    if k_neighbors <= 0 or max_edge_m <= 0:
        return {"type": "FeatureCollection", "features": feats}

    fl = np.asarray(flags, dtype=bool)
    if len(fl) != n:
        fl = np.zeros(n, dtype=bool)

    # Уже существующие внутрикластерные рёбра, чтобы не дублировать.
    existing_pairs: set[tuple[int, int]] = set()
    for feat in feats:
        p = feat.get("properties") or {}
        if p.get("inter_cluster"):
            continue
        si, ti = p.get("source_i"), p.get("target_i")
        if si is None or ti is None:
            continue
        a, b = int(si), int(ti)
        if a < 0 or b < 0 or a >= n or b >= n or a == b:
            continue
        existing_pairs.add((a, b) if a < b else (b, a))

    by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(cluster_per_point):
        by_cluster.setdefault(str(c), []).append(i)

    added = 0
    for cid, idxs in by_cluster.items():
        if len(idxs) < 3:
            continue
        if len(idxs) > _SMALL_CLUSTER_EXTRA_MAX_POINTS:
            continue
        hp = _hull_polygon_for_cluster(hulls_norm, cid)
        area = _hull_area_m2(hp)
        if area > _VORONOI_BPC_SMALL_CLUSTER_MAX_AREA_M2:
            continue

        local = np.asarray([coords[i] for i in idxs], dtype=float)
        for li, gi in enumerate(idxs):
            pi = local[li]
            dists = np.empty(len(idxs), dtype=float)
            for lj in range(len(idxs)):
                if lj == li:
                    dists[lj] = np.inf
                    continue
                pj = local[lj]
                dists[lj] = _segment_length_m(
                    float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1])
                )
            k_eff = min(k_neighbors, max(0, len(idxs) - 1))
            if k_eff <= 0:
                continue
            nbr_local_idx = np.argpartition(dists, k_eff)[:k_eff]
            for lj in nbr_local_idx.tolist():
                gj = idxs[int(lj)]
                if gi == gj:
                    continue
                u, v = (gi, gj) if gi < gj else (gj, gi)
                if (u, v) in existing_pairs:
                    continue
                d = float(dists[int(lj)])
                if not np.isfinite(d) or d > max_edge_m:
                    continue
                p1 = coords[u]
                p2 = coords[v]
                feats.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [float(p1[0]), float(p1[1])],
                                [float(p2[0]), float(p2[1])],
                            ],
                        },
                        "properties": {
                            **_station_edge_props(
                                _cluster_id_prop_from_key(cid),
                                "voronoi_local",
                                u,
                                v,
                                fl,
                                group_mode=None,
                            ),
                            "inter_cluster": False,
                            "small_cluster_knn_extra": True,
                        },
                    }
                )
                existing_pairs.add((u, v))
                added += 1

    out = dict(fc)
    out["features"] = feats
    out["small_cluster_extra_edges_added"] = int(added)
    return out


def _bordering_cluster_pair_keys(
    hulls_norm: gpd.GeoDataFrame | None,
    *,
    max_boundary_gap_m: float = 45.0,
) -> set[tuple[str, str]]:
    """
    Пары cluster_id, у которых полигоны hull в EPSG:3857 касаются или разделены не больше max_boundary_gap_m.
    Так отсекаются длинные рёбра Делоне между далёкими кластерами (через реку и т.п.).
    """
    if hulls_norm is None or len(hulls_norm) < 2:
        return set()
    try:
        h = hulls_norm.to_crs(epsg=3857)
    except Exception:
        return set()
    out: set[tuple[str, str]] = set()
    ids: list[str] = [str(x) for x in h["cluster_id"].tolist()]
    geoms: list = []
    for g in h.geometry:
        if g is None or getattr(g, "is_empty", True):
            geoms.append(None)
            continue
        try:
            geoms.append(g.buffer(0))
        except Exception:
            geoms.append(g)
    tol = float(max_boundary_gap_m)
    n = len(ids)
    for i in range(n):
        gi = geoms[i]
        if gi is None:
            continue
        for j in range(i + 1, n):
            gj = geoms[j]
            if gj is None:
                continue
            if ids[i] == ids[j]:
                continue
            try:
                d = float(gi.distance(gj))
            except Exception:
                continue
            if d <= tol:
                out.add(tuple(sorted((ids[i], ids[j]))))
    return out


def _filter_inter_cluster_edges_through_third_cluster(
    fc: dict,
    hulls_norm: gpd.GeoDataFrame | None,
) -> dict:
    """
    Удаляет межкластерные рёбра, если они проходят через hull любого третьего кластера.
    Для скорости использует spatial index по hull bbox.
    """
    if not fc or not isinstance(fc, dict):
        return {"type": "FeatureCollection", "features": []}
    feats = list(fc.get("features", []) or [])
    if not feats or hulls_norm is None or len(hulls_norm) == 0:
        return {"type": "FeatureCollection", "features": feats}

    h = hulls_norm[["cluster_id", "geometry"]].copy()
    h = h[h["cluster_id"].notna()].copy()
    if len(h) == 0:
        return {"type": "FeatureCollection", "features": feats}
    try:
        h["geometry"] = h.geometry.buffer(0)
    except Exception:
        pass
    h = h[h.geometry.notna()].copy()
    h = h[~h.geometry.is_empty].copy()
    if len(h) == 0:
        return {"type": "FeatureCollection", "features": feats}

    try:
        sidx = h.sindex
    except Exception:
        sidx = None

    eps = 1e-9
    kept: list[dict] = []
    for feat in feats:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        if not props.get("inter_cluster"):
            kept.append(feat)
            continue
        cid_raw = props.get("cluster_id")
        if not (isinstance(cid_raw, str) and "|" in cid_raw):
            kept.append(feat)
            continue
        pair_parts = [p.strip() for p in cid_raw.split("|") if str(p).strip()]
        if len(pair_parts) != 2:
            kept.append(feat)
            continue
        endpoint_clusters = {pair_parts[0], pair_parts[1]}

        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            kept.append(feat)
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        try:
            line = LineString([(float(c[0]), float(c[1])) for c in coords if len(c) >= 2])
        except Exception:
            kept.append(feat)
            continue
        if line.is_empty:
            continue

        blocked = False
        try:
            cand_idx = list(sidx.intersection(line.bounds)) if sidx is not None else list(range(len(h)))
            for hi in cand_idx:
                row = h.iloc[int(hi)]
                other_cid = str(row["cluster_id"])
                if other_cid in endpoint_clusters:
                    continue
                g = row["geometry"]
                if g is None or getattr(g, "is_empty", True):
                    continue
                inter = line.intersection(g)
                if inter.is_empty:
                    continue
                if inter.geom_type in ("Point", "MultiPoint"):
                    continue
                if inter.geom_type == "GeometryCollection":
                    parts = list(getattr(inter, "geoms", []))
                    if parts and all(
                        (p.geom_type in ("Point", "MultiPoint") or getattr(p, "length", 0.0) <= eps)
                        for p in parts
                    ):
                        continue
                if getattr(inter, "length", 0.0) > eps or getattr(inter, "area", 0.0) > eps:
                    blocked = True
                    break
                blocked = True
                break
        except Exception:
            kept.append(feat)
            continue

        if blocked:
            continue
        kept.append(feat)

    return {"type": "FeatureCollection", "features": kept}


def _separate_coincident_sites(coords: np.ndarray) -> np.ndarray:
    """Сдвигает дубликаты координат на ~0.5 m, чтобы сайты разных кластеров не схлопывались в одном Voronoi."""
    c = np.asarray(coords, dtype=float).copy()
    if len(c) <= 1:
        return c
    ref_lat = float(np.mean(c[:, 1]))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = max(111_320.0 * np.cos(np.radians(ref_lat)), 1e-3)
    step_lat = 0.5 / m_per_deg_lat
    step_lon = 0.5 / m_per_deg_lon
    seen: dict[tuple[float, float], int] = {}
    for i in range(len(c)):
        key = (round(float(c[i, 0]), 9), round(float(c[i, 1]), 9))
        n = seen.get(key, 0)
        if n > 0:
            k = n
            c[i, 0] += (k // 4) * step_lon
            c[i, 1] += (k % 4) * step_lat
        seen[key] = n + 1
    return c


def _append_voronoi_edges_global(
    features: list[dict],
    pts_arr: np.ndarray,
    is_station: np.ndarray,
    cluster_per_point: list[Any],
    hulls_norm: gpd.GeoDataFrame | None,
    edge_type: str = "voronoi_local",
    *,
    bordering_pair_keys: set[tuple[str, str]] | None = None,
    inter_cluster_max_edge_length_m: float | None = None,
) -> bool:
    """
    Один Voronoi по всем сайтам: рёбра — пары ridge_points (как у Делоне).
    Отрезок всегда между двумя сайтами целиком (без обрезки по hull), в т.ч. внутри одного кластера —
    чтобы линии не обрывались на границе полигона.
    Межкластерные рёбра — только если пара в bordering_pair_keys (смежные hull / зазор до порога),
    и длина отрезка не больше inter_cluster_max_edge_length_m (если задано).
    """
    Voronoi = _require_voronoi()
    n0 = len(features)

    if len(pts_arr) < 2:
        return False

    c_labels = list(cluster_per_point)
    if len(c_labels) != len(pts_arr):
        return False

    if len(pts_arr) == 2:
        coords2 = _separate_coincident_sites(np.asarray(pts_arr, dtype=float))
        p0, p1 = coords2[0], coords2[1]
        c0, c1 = c_labels[0], c_labels[1]
        flags = np.asarray(is_station, dtype=bool)
        if len(flags) != 2:
            flags = np.array([False, False], dtype=bool)
        if str(c0) == str(c1):
            clipped = [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]
            cid_prop = _cluster_label_for_props(c0)
        else:
            pk = _canonical_cluster_pair_key(c0, c1)
            if bordering_pair_keys is not None and pk not in bordering_pair_keys:
                return len(features) > n0
            if inter_cluster_max_edge_length_m is not None and inter_cluster_max_edge_length_m > 0:
                sl = _segment_length_m(
                    float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])
                )
                if sl > inter_cluster_max_edge_length_m:
                    return len(features) > n0
            clipped = [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]
            cid_prop = _inter_cluster_pair_label(c0, c1)
        if clipped is None or len(clipped) < 2:
            return False
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": clipped},
                "properties": {
                    **_station_edge_props(cid_prop, edge_type, 0, 1, flags, group_mode=None),
                    "inter_cluster": str(c0) != str(c1),
                },
            }
        )
        return True

    coords = _separate_coincident_sites(np.asarray(pts_arr, dtype=float))
    flags = np.asarray(is_station, dtype=bool)
    labels = list(c_labels)
    if len(flags) != len(coords):
        flags = np.zeros(len(coords), dtype=bool)

    map_idx: dict[int, int] = {i: i for i in range(len(coords))}
    try:
        vor = Voronoi(coords)
    except Exception:
        rng = np.random.default_rng(42)
        coords = coords + (rng.random(coords.shape) - 0.5) * 1e-6
        try:
            vor = Voronoi(coords)
        except Exception:
            uniq, inv = np.unique(coords, axis=0, return_inverse=True)
            if len(uniq) < 2:
                return False
            fu = np.zeros(len(uniq), dtype=bool)
            lab_u: list[Any] = [None] * len(uniq)
            for k in range(len(flags)):
                u = int(inv[k])
                fu[u] |= bool(flags[k])
                if lab_u[u] is None:
                    lab_u[u] = labels[k]
            if any(x is None for x in lab_u):
                return False
            coords = uniq
            flags = fu
            labels = lab_u
            map_idx = {i: i for i in range(len(coords))}
            if len(coords) == 2:
                return _append_voronoi_edges_global(
                    features,
                    coords,
                    flags,
                    labels,
                    hulls_norm,
                    edge_type=edge_type,
                    bordering_pair_keys=bordering_pair_keys,
                    inter_cluster_max_edge_length_m=inter_cluster_max_edge_length_m,
                )
            try:
                vor = Voronoi(coords)
            except Exception:
                return False

    seen_pairs: set[tuple[int, int]] = set()
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
        pi = coords[a]
        pj = coords[b]
        ca, cb = labels[a], labels[b]
        same = str(ca) == str(cb)
        if same:
            clipped = [
                [float(pi[0]), float(pi[1])],
                [float(pj[0]), float(pj[1])],
            ]
            cid_prop = _cluster_label_for_props(ca)
        else:
            pk = _canonical_cluster_pair_key(ca, cb)
            if bordering_pair_keys is not None and pk not in bordering_pair_keys:
                continue
            if inter_cluster_max_edge_length_m is not None and inter_cluster_max_edge_length_m > 0:
                sl = _segment_length_m(
                    float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1])
                )
                if sl > inter_cluster_max_edge_length_m:
                    continue
            clipped = [
                [float(pi[0]), float(pi[1])],
                [float(pj[0]), float(pj[1])],
            ]
            cid_prop = _inter_cluster_pair_label(ca, cb)
        if clipped is None or len(clipped) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": clipped},
                "properties": {
                    **_station_edge_props(cid_prop, edge_type, a, b, flags, group_mode=None),
                    "inter_cluster": not same,
                },
            }
        )
    return len(features) > n0


def build_voronoi_local_paths_fc(
    data_service: DataService,
    city: str,
    network_type: str,
    simplify: bool,
    dbscan_eps_m: float,
    dbscan_min_samples: int,
    use_all_buildings: bool,
    buildings_per_centroid: int = 60,
    charging_station_features: list | None = None,
    inter_cluster_max_hull_gap_m: float = 2000.0,
    inter_cluster_max_edge_length_m: float = 2000.0,
    voronoi_intra_component_bridge_max_m: float = _DEFAULT_VORONOI_INTRA_COMPONENT_BRIDGE_MAX_M,
    *,
    city_data: dict | None = None,
) -> dict:
    _require_voronoi()

    if city_data is not None:
        data = city_data
    else:
        data = data_service.get_city_data(
            city, network_type=network_type, simplify=simplify, load_no_fly_zones=True
        )
    buildings = data.get("buildings")
    road_graph = data.get("road_graph")
    city_boundary = data.get("city_boundary")
    no_fly_zones = data.get("no_fly_zones")
    nfz_union_f = _no_fly_obstacles_union(no_fly_zones)

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

    if "height_m" not in buildings_wgs.columns:
        buildings_wgs = data_service._compute_building_heights(buildings_wgs)

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

    hulls_norm = _normalize_hulls_gdf(hulls)

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

    features: list[dict] = []
    clusters_total = 0
    all_coords: list[list[float]] = []
    all_flags: list[bool] = []
    all_cluster: list[Any] = []

    for group_id, sub in assigned.groupby("cluster_id"):
        if sub is None or len(sub) < 2:
            continue

        pts_mkd: list[list[float]] = []
        pts_non: list[list[float]] = []
        for bid, row in sub.iterrows():
            geom = row.get("geometry")
            if geom is None or geom.is_empty:
                continue
            try:
                lon, lat = float(geom.x), float(geom.y)
            except Exception:
                continue
            if nfz_union_f is not None and _point_in_nfz_union(lon, lat, nfz_union_f):
                continue
            try:
                b_row = buildings_wgs.loc[bid]
            except (KeyError, TypeError, IndexError):
                b_row = row
            if data_service.row_is_mkd_building(b_row):
                pts_mkd.append([lon, lat])
            else:
                pts_non.append([lon, lat])

        if len(pts_mkd) + len(pts_non) < 2:
            continue

        hull_poly = _hull_polygon_for_cluster(hulls_norm, group_id)
        pts_arr = _voronoi_cluster_building_centroids(
            hull_poly, pts_mkd, pts_non, buildings_per_centroid
        )

        pts_arr, _ = _filter_xy_rows_outside_nfz(pts_arr, None, nfz_union_f)

        st_points = charging_by_cluster.get(str(group_id), [])
        if st_points:
            st_points = [
                [float(x), float(y)]
                for x, y in st_points
                if (nfz_union_f is None or not _point_in_nfz_union(float(x), float(y), nfz_union_f))
                and (hull_poly is None or _point_in_hull(float(x), float(y), hull_poly))
            ]
        st_arr = np.asarray(st_points, dtype=float) if st_points else None
        pts_arr, is_station = merge_xy_with_station_flags(pts_arr, st_arr)
        pts_arr, is_station = _filter_xy_rows_outside_nfz(pts_arr, is_station, nfz_union_f)

        if len(pts_arr) < 2:
            continue
        clusters_total += 1
        for k in range(len(pts_arr)):
            all_coords.append([float(pts_arr[k, 0]), float(pts_arr[k, 1])])
            all_flags.append(bool(is_station[k]))
            all_cluster.append(group_id)

    if len(all_coords) < 2:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": int(clusters_total),
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    coords_g = np.asarray(all_coords, dtype=float)
    flags_g = np.asarray(all_flags, dtype=bool)
    bordering_keys = _bordering_cluster_pair_keys(
        hulls_norm, max_boundary_gap_m=float(inter_cluster_max_hull_gap_m)
    )
    cap_edge = float(inter_cluster_max_edge_length_m)
    edge_cap: float | None = cap_edge if cap_edge > 0 else None
    _append_voronoi_edges_global(
        features,
        coords_g,
        flags_g,
        all_cluster,
        hulls_norm,
        "voronoi_local",
        bordering_pair_keys=bordering_keys,
        inter_cluster_max_edge_length_m=edge_cap,
    )
    _add_shortest_bridges_for_allowed_pairs(
        features,
        coords_g,
        flags_g,
        all_cluster,
        bordering_keys,
        max_edge_m=cap_edge,
    )

    deduped = dedupe_close_parallel_voronoi_edges(
        {
            "type": "FeatureCollection",
            "features": features,
        }
    )
    deduped = _filter_inter_cluster_edges_through_third_cluster(deduped, hulls_norm)
    deduped = _add_extra_intra_edges_for_small_clusters(
        deduped,
        coords_g,
        flags_g,
        all_cluster,
        hulls_norm,
    )

    from station_placement import buildings_footprint_union_min_height_wgs84

    deduped_pre_barrier = deduped
    voronoi_alts = DataService.voronoi_echelon_altitudes_m(data.get("flight_levels"))
    echelon_levels = list(DataService.VORONOI_FLIGHT_ECHELON_LEVELS)
    voronoi_by_echelon: dict[str, dict[str, Any]] = {}
    bridge_m = float(voronoi_intra_component_bridge_max_m)

    for idx, alt in enumerate(voronoi_alts):
        level = int(echelon_levels[idx]) if idx < len(echelon_levels) else idx + 1
        min_h_b = float(DataService.echelon_altitude_to_building_obstacle_min_m(float(alt)))
        tall_i = buildings_footprint_union_min_height_wgs84(buildings_wgs, min_h_b)
        fc_i = _voronoi_fc_copy_filter_and_bridges(
            deduped_pre_barrier,
            nfz_union=nfz_union_f,
            building_clearance_union=tall_i,
            coords_g=coords_g,
            flags_g=flags_g,
            all_cluster=all_cluster,
            max_bridge_m=bridge_m,
            buildings_wgs=buildings_wgs,
            voronoi_obstacle_min_building_height_m=min_h_b,
        )
        voronoi_by_echelon[str(level)] = {
            "type": "FeatureCollection",
            "features": list(fc_i.get("features") or []),
            "voronoi_echelon_level": level,
            "voronoi_echelon_altitude_m": float(alt),
            "voronoi_building_obstacle_min_height_m": min_h_b,
            "voronoi_intra_component_bridges_added": int(fc_i.get("voronoi_intra_component_bridges_added") or 0),
        }

    e1 = voronoi_by_echelon.get("1") or next(iter(voronoi_by_echelon.values()))
    deduped = dict(deduped_pre_barrier)
    deduped["type"] = "FeatureCollection"
    deduped["features"] = list(e1.get("features") or [])
    deduped["voronoi_intra_component_bridges_added"] = int(
        e1.get("voronoi_intra_component_bridges_added") or 0
    )
    deduped["voronoi_by_echelon"] = voronoi_by_echelon

    # Обходы no-fly для рёбер Вороного не строим — прямые отрезки как у диаграммы (без boundary/A*).
    deduped["nfz_voronoi_edges_crossing"] = 0
    deduped["nfz_voronoi_detours_ok"] = 0
    deduped["nfz_voronoi_detours_failed"] = 0
    deduped["voronoi_echelon_levels"] = list(DataService.VORONOI_FLIGHT_ECHELON_LEVELS)
    deduped["voronoi_echelon_altitudes_m"] = [float(x) for x in voronoi_alts]
    deduped["voronoi_echelon_altitude_m"] = float(min(voronoi_alts))
    deduped["voronoi_building_obstacle_min_height_m"] = float(
        DataService.voronoi_building_obstacle_min_height_m(data.get("flight_levels"))
    )
    deduped["flight_echelon_building_obstacle_min_height_m"] = {
        int(k): float(v) for k, v in DataService.all_echelon_building_obstacle_min_heights_m(
            data.get("flight_levels")
        ).items()
    }

    touched: set[str] = set()
    for f in deduped.get("features", []) or []:
        p = f.get("properties") or {}
        cid = p.get("cluster_id")
        if p.get("inter_cluster"):
            if isinstance(cid, str) and "|" in cid:
                for part in cid.split("|"):
                    touched.add(part)
        elif cid is not None:
            touched.add(str(cid))

    deduped["clusters_total"] = int(clusters_total)
    deduped["clusters_with_paths"] = int(len(touched))
    deduped["edges_total"] = len(deduped.get("features", []) or [])
    return deduped


def build_voronoi_edges_from_station_geojson(geo: dict) -> dict:
    """Строит локальные рёбра Вороного по точкам станций из ответа pipeline_result_to_geojson (по cluster_id)."""
    _require_voronoi()

    station_layers = [
        geo.get("charging_type_a") or {"type": "FeatureCollection", "features": []},
        geo.get("charging_type_b") or {"type": "FeatureCollection", "features": []},
        geo.get("garages") or {"type": "FeatureCollection", "features": []},
        geo.get("to_stations") or {"type": "FeatureCollection", "features": []},
    ]

    by_cluster: dict[str, list[list[float]]] = {}
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
            if cid is not None:
                key = str(cid)
                by_cluster.setdefault(key, []).append(pt)

    out_features: list[dict] = []
    for cluster_key, pts in by_cluster.items():
        if len(pts) < 2:
            continue
        pts_arr = np.asarray(pts, dtype=float)
        is_station = np.ones(len(pts_arr), dtype=bool)
        _append_voronoi_edges_for_cluster(
            out_features,
            pts_arr,
            is_station,
            cluster_key,
            "voronoi_station_local",
            group_mode="cluster_or_fallback",
            hull_poly=None,
        )

    return {"type": "FeatureCollection", "features": out_features}


def build_voronoi_edges_from_pipeline_raw(
    data_service: DataService,
    raw: dict,
    charging_station_features: list | None = None,
    buildings_per_centroid: int = 60,
) -> dict:
    """
    Быстрый Вороной без повторного data_service.get_city_data:
    используем demand из уже выполненного run_full_pipeline.
    """
    _require_voronoi()

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

    hulls_norm = _normalize_hulls_gdf(raw.get("demand_hulls"))
    nfz_union_f = _no_fly_obstacles_union(raw.get("no_fly_zones"))

    grouped_points: dict[str, list[list[float]]] = {}
    mkd_split_by_cluster: dict[str, tuple[list[list[float]], list[list[float]]]] | None = None
    for cluster_id, sub in d.groupby("cluster_id"):
        pts = []
        for _, row in sub.iterrows():
            g = row.get("geometry")
            if g is None or g.is_empty:
                continue
            try:
                lo, la = float(g.x), float(g.y)
            except Exception:
                continue
            if nfz_union_f is not None and _point_in_nfz_union(lo, la, nfz_union_f):
                continue
            pts.append([lo, la])
        if pts:
            grouped_points[str(cluster_id)] = pts

    # Добавляем только здания, чей центроид попадает в hull соответствующего кластера (не по ближайшему центроиду).
    if not any(len(pts) >= 2 for pts in grouped_points.values()):
        buildings = raw.get("buildings")
        if (
            buildings is not None
            and len(buildings) > 0
            and hulls_norm is not None
            and len(hulls_norm) > 0
        ):
            try:
                b = buildings.to_crs("EPSG:4326").copy()
                try:
                    b = data_service._filter_buildings_for_demand(b)
                except Exception:
                    pass
                if b is not None and len(b) > 0:
                    b_proj = b.to_crs(epsg=3857)
                    b_centroids_wgs = gpd.GeoSeries(b_proj.geometry.centroid, crs=b_proj.crs).to_crs("EPSG:4326")
                    b_points = gpd.GeoDataFrame(
                        {"geometry": b_centroids_wgs},
                        index=b.index,
                        crs="EPSG:4326",
                    )
                    h_join = hulls_norm[["cluster_id", "geometry"]].copy()
                    try:
                        assigned = gpd.sjoin(
                            b_points,
                            h_join,
                            how="inner",
                            predicate="within",
                        )
                    except Exception:
                        assigned = gpd.sjoin(b_points, h_join, how="inner", op="within")
                    if assigned is not None and len(assigned) > 0:
                        assigned = assigned.copy()
                        if "index_left" in assigned.columns:
                            assigned["_bid"] = assigned["index_left"]
                        else:
                            assigned["_bid"] = assigned.index
                        assigned = assigned.drop_duplicates(subset=["_bid"])
                        gp: dict[str, list[list[float]]] = {}
                        gp_mkd: dict[str, list[list[float]]] = {}
                        gp_non: dict[str, list[list[float]]] = {}
                        for _, row in assigned.iterrows():
                            g = row.geometry
                            cid = row.get("cluster_id")
                            if g is None or g.is_empty or cid is None:
                                continue
                            try:
                                lo, la = float(g.x), float(g.y)
                            except Exception:
                                continue
                            if nfz_union_f is not None and _point_in_nfz_union(lo, la, nfz_union_f):
                                continue
                            sk = str(cid)
                            gp.setdefault(sk, []).append([lo, la])
                            bid = row.get("_bid")
                            try:
                                b_row = b.loc[bid] if bid is not None else row
                            except (KeyError, TypeError, IndexError):
                                b_row = row
                            if data_service.row_is_mkd_building(b_row):
                                gp_mkd.setdefault(sk, []).append([lo, la])
                            else:
                                gp_non.setdefault(sk, []).append([lo, la])
                        if gp:
                            grouped_points = gp
                            mkd_split_by_cluster = {
                                sk: (gp_mkd.get(sk, []), gp_non.get(sk, [])) for sk in gp
                            }
            except Exception:
                pass

    charging_station_features = _filter_station_features_to_hulls(
        charging_station_features,
        hulls_norm,
    )

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
            lo, la = float(coords[0]), float(coords[1])
            if nfz_union_f is not None and _point_in_nfz_union(lo, la, nfz_union_f):
                continue
            charging_by_cluster.setdefault(str(cid), []).append([lo, la])

    out_features: list[dict] = []
    for cluster_id, pts in grouped_points.items():
        if len(pts) < 2:
            continue

        hull_poly = _hull_polygon_for_cluster(hulls_norm, cluster_id)
        sk = str(cluster_id)
        split = mkd_split_by_cluster.get(sk) if mkd_split_by_cluster else None
        if split is not None:
            pts_arr = _voronoi_cluster_building_centroids(
                hull_poly, split[0], split[1], buildings_per_centroid
            )
        else:
            target_group_size = max(
                1,
                _voronoi_effective_buildings_per_centroid(buildings_per_centroid, hull_poly),
            )
            pts_arr = np.asarray(pts, dtype=float)
            pts_arr = _aggregate_points_to_centroids(pts_arr, target_group_size)

        pts_arr, _ = _filter_xy_rows_outside_nfz(pts_arr, None, nfz_union_f)

        st_pts = charging_by_cluster.get(str(cluster_id), [])
        if st_pts:
            st_pts = [
                [float(x), float(y)]
                for x, y in st_pts
                if (nfz_union_f is None or not _point_in_nfz_union(float(x), float(y), nfz_union_f))
                and (hull_poly is None or _point_in_hull(float(x), float(y), hull_poly))
            ]
        st_arr = np.asarray(st_pts, dtype=float) if st_pts else None
        pts_arr, is_station = merge_xy_with_station_flags(pts_arr, st_arr)
        pts_arr, is_station = _filter_xy_rows_outside_nfz(pts_arr, is_station, nfz_union_f)

        if len(pts_arr) < 2:
            continue
        _append_voronoi_edges_for_cluster(
            out_features,
            pts_arr,
            is_station,
            cluster_id,
            "voronoi_local",
            hull_poly=hull_poly,
        )
    fc_out = {"type": "FeatureCollection", "features": out_features}
    fc_pre = {"type": "FeatureCollection", "features": list(out_features)}
    try:
        b_raw = raw.get("buildings")
        if b_raw is not None and len(b_raw) > 0:
            b_w = b_raw.to_crs("EPSG:4326").copy()
            if "height_m" not in b_w.columns:
                b_w = data_service._compute_building_heights(b_w)
            from station_placement import buildings_footprint_union_min_height_wgs84

            v_alts = DataService.voronoi_echelon_altitudes_m(raw.get("flight_levels"))
            levels = list(DataService.VORONOI_FLIGHT_ECHELON_LEVELS)
            vbe: dict[str, dict[str, Any]] = {}
            for idx, alt in enumerate(v_alts):
                level = int(levels[idx]) if idx < len(levels) else idx + 1
                min_h_b = float(DataService.echelon_altitude_to_building_obstacle_min_m(float(alt)))
                tall_u = buildings_footprint_union_min_height_wgs84(b_w, min_h_b)
                fc_i = {
                    "type": "FeatureCollection",
                    "features": copy.deepcopy(list(fc_pre.get("features") or [])),
                }
                fc_i = _filter_voronoi_fc_linestrings_nfz(fc_i, nfz_union_f)
                fc_i = _filter_voronoi_fc_building_clearance(fc_i, tall_u)
                fc_i = _filter_voronoi_fc_station_roof_echelon(fc_i, b_w, min_h_b)
                vbe[str(level)] = {
                    "type": "FeatureCollection",
                    "features": list(fc_i.get("features") or []),
                    "voronoi_echelon_level": level,
                    "voronoi_echelon_altitude_m": float(alt),
                    "voronoi_building_obstacle_min_height_m": min_h_b,
                }
            e1 = vbe.get("1")
            if e1:
                fc_out["features"] = list(e1["features"])
            fc_out["voronoi_by_echelon"] = vbe
            fc_out["voronoi_echelon_levels"] = list(DataService.VORONOI_FLIGHT_ECHELON_LEVELS)
            fc_out["voronoi_echelon_altitudes_m"] = [float(x) for x in v_alts]
            fc_out["voronoi_echelon_altitude_m"] = float(min(v_alts))
            fc_out["voronoi_building_obstacle_min_height_m"] = float(
                DataService.voronoi_building_obstacle_min_height_m(raw.get("flight_levels"))
            )
            fc_out["flight_echelon_building_obstacle_min_height_m"] = {
                int(k): float(v)
                for k, v in DataService.all_echelon_building_obstacle_min_heights_m(
                    raw.get("flight_levels")
                ).items()
            }
            return fc_out
    except Exception:
        pass
    fc_out = _filter_voronoi_fc_linestrings_nfz(fc_out, nfz_union_f)
    return fc_out
