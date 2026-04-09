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
from voronoi_paths import build_voronoi_local_paths_fc


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


def _empty_feature_collection() -> dict:
    return {"type": "FeatureCollection", "features": []}


def _round_geojson_coords(obj, precision: int = 6):
    if isinstance(obj, dict):
        return {k: _round_geojson_coords(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_geojson_coords(item, precision) for item in obj]
    if isinstance(obj, float):
        return round(obj, precision)
    return obj


def _voronoi_only_placement_payload(
    *,
    city: str,
    network_type: str,
    simplify: bool,
    dbscan_eps_m: float,
    dbscan_min_samples: int,
    buildings_per_centroid: int,
    inter_cluster_max_hull_gap_m: float,
    inter_cluster_max_edge_length_m: float,
    use_all_buildings: bool,
    voronoi_intra_bridge_max_m: float,
    from_saved: bool,
) -> dict:
    """
    Только маршруты Вороного (как /api/buildings/voronoi-local-paths): без расстановки станций и без магистрали.
    Формат ответа совместим с прежним `/api/stations/placement` (пустые слои станций/trunk, заполнен voronoi_edges).
    """
    raw_fc = build_voronoi_local_paths_fc(
        data_service,
        city=city.strip(),
        network_type=network_type,
        simplify=simplify,
        dbscan_eps_m=dbscan_eps_m,
        dbscan_min_samples=dbscan_min_samples,
        use_all_buildings=use_all_buildings,
        buildings_per_centroid=buildings_per_centroid,
        charging_station_features=None,
        inter_cluster_max_hull_gap_m=inter_cluster_max_hull_gap_m,
        inter_cluster_max_edge_length_m=inter_cluster_max_edge_length_m,
        voronoi_intra_component_bridge_max_m=float(voronoi_intra_bridge_max_m),
    )
    empty_fc = _empty_feature_collection()
    vor_fc = {
        "type": "FeatureCollection",
        "features": list(raw_fc.get("features") or []),
    }
    metrics = {
        "voronoi_edges_total": int(raw_fc.get("edges_total", len(vor_fc["features"]))),
        "voronoi_clusters_with_paths": int(raw_fc.get("clusters_with_paths", 0)),
        "voronoi_clusters_total": int(raw_fc.get("clusters_total", 0)),
        "voronoi_intra_component_bridges_added": int(
            raw_fc.get("voronoi_intra_component_bridges_added", 0)
        ),
    }
    geo = {
        "charging_type_a": empty_fc,
        "charging_type_b": empty_fc,
        "garages": empty_fc,
        "to_stations": empty_fc,
        "trunk": empty_fc,
        "branch_edges": empty_fc,
        "local_edges": empty_fc,
        "voronoi_edges": _round_geojson_coords(vor_fc, precision=6),
        "metrics": metrics,
        "cluster_centroids": empty_fc,
        "params": {
            "city": city.strip(),
            "network_type": network_type,
            "simplify": simplify,
            "pipeline_mode": "voronoi_only",
            "buildings_per_centroid": buildings_per_centroid,
            "inter_cluster_max_hull_gap_m": inter_cluster_max_hull_gap_m,
            "inter_cluster_max_edge_length_m": inter_cluster_max_edge_length_m,
            "dbscan_eps_m": dbscan_eps_m,
            "dbscan_min_samples": dbscan_min_samples,
            "use_all_buildings": use_all_buildings,
            "voronoi_intra_bridge_max_m": float(voronoi_intra_bridge_max_m),
        },
    }
    geo["from_saved"] = from_saved
    return geo


def _placement_cache_key(
    city: str,
    network_type: str,
    simplify: bool,
    dbscan_eps_m: float,
    dbscan_min_samples: int,
    buildings_per_centroid: int,
    inter_cluster_max_hull_gap_m: float,
    inter_cluster_max_edge_length_m: float,
    use_all_buildings: bool,
    voronoi_intra_bridge_max_m: float,
) -> str:
    payload = {
        "city": city.strip(),
        "network_type": network_type,
        "simplify": bool(simplify),
        "dbscan_eps_m": float(dbscan_eps_m),
        "dbscan_min_samples": int(dbscan_min_samples),
        "buildings_per_centroid": int(buildings_per_centroid),
        "inter_cluster_max_hull_gap_m": float(inter_cluster_max_hull_gap_m),
        "inter_cluster_max_edge_length_m": float(inter_cluster_max_edge_length_m),
        "use_all_buildings": bool(use_all_buildings),
        "voronoi_intra_bridge_max_m": float(voronoi_intra_bridge_max_m),
        # Смена версии сбрасывает устаревший Redis-кэш (иначе после правок пайплайна
        # клиент может бесконечно получать пустой сохранённый ответ).
        "placement_schema": 36,
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
    buildings_per_centroid: int = Query(
        60,
        description="Для кластера с hull > 2 000 000 м² — зданий на 1 сайт Вороного (по умолчанию 60); при hull ≤ 2 000 000 м² — 3 здания на 1 сайт; no-fly для Вороного не применяется",
    ),
    inter_cluster_max_hull_gap_m: float = Query(
        2000.0,
        description="Макс. зазор между hull кластеров (м), при котором разрешены межкластерные связи (дорога, пустырь)",
    ),
    inter_cluster_max_edge_length_m: float = Query(
        2000.0,
        description="Макс. длина одного межкластерного отрезка (м); защита от длинных перебросов",
    ),
    voronoi_intra_bridge_max_m: float = Query(
        600.0,
        description="После вырезания рёбер через NFZ: макс. длина моста внутри кластера между компонентами (м); 0 — не добавлять",
    ),
):
    """
    Локальные пути внутри кластера на основе соседства ячеек Вороного.
    Считается на сервере по cluster-данным из кэша DataService.
    """

    try:
        raw_fc = build_voronoi_local_paths_fc(
            data_service,
            city=city,
            network_type=network_type,
            simplify=simplify,
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
            use_all_buildings=use_all_buildings,
            buildings_per_centroid=buildings_per_centroid,
            inter_cluster_max_hull_gap_m=inter_cluster_max_hull_gap_m,
            inter_cluster_max_edge_length_m=inter_cluster_max_edge_length_m,
            voronoi_intra_component_bridge_max_m=float(voronoi_intra_bridge_max_m),
        )
        fc = _round_geojson_coords(raw_fc, precision=6)
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
    buildings_per_centroid: int = Query(
        60,
        description="Для hull > 2 000 000 м² — зданий на сайт (по умолчанию 60); при hull ≤ 2 000 000 м² — 3 здания на 1 сайт; no-fly для Вороного не применяется",
    ),
    inter_cluster_max_hull_gap_m: float = Query(
        2000.0,
        description="Макс. зазор между hull кластеров (м) для межкластерных рёбер",
    ),
    inter_cluster_max_edge_length_m: float = Query(
        2000.0,
        description="Макс. длина межкластерного отрезка (м)",
    ),
    use_all_buildings: bool = Query(
        False,
        description="Учитывать все здания вне no-fly (как у /api/buildings/voronoi-local-paths)",
    ),
    voronoi_intra_bridge_max_m: float = Query(
        600.0,
        description="Мосты связности внутри кластера после NFZ (м); 0 — выключить",
    ),
):
    """
    Пайплайн карты: только маршруты Вороного (меж- и внутрикластерные), без расстановки станций и без магистрали.
    Параметры demand_method / a_by_admin_districts / candidates_per_cluster оставлены для совместимости запросов и не используются.
    Кэш в Redis — JSON ответа для карты.
    """
    cache_key = _placement_cache_key(
        city=city,
        network_type=network_type,
        simplify=simplify,
        dbscan_eps_m=dbscan_eps_m,
        dbscan_min_samples=dbscan_min_samples,
        buildings_per_centroid=buildings_per_centroid,
        inter_cluster_max_hull_gap_m=inter_cluster_max_hull_gap_m,
        inter_cluster_max_edge_length_m=inter_cluster_max_edge_length_m,
        use_all_buildings=use_all_buildings,
        voronoi_intra_bridge_max_m=float(voronoi_intra_bridge_max_m),
    )
    redis_client = data_service.get_redis_client()

    if only_saved:
        if redis_client is None:
            raise HTTPException(status_code=503, detail="Redis недоступен")
        cached = redis_client.get(cache_key)
        if not cached:
            raise HTTPException(status_code=404, detail="Нет сохранённой расстановки")
        return JSONResponse(json.loads(cached.decode("utf-8")))

    if use_saved and redis_client is not None:
        cached = redis_client.get(cache_key)
        if cached:
            data = json.loads(cached.decode("utf-8"))
            data["from_saved"] = True
            return JSONResponse(data)

    try:
        logger.info("Пайплайн: только маршруты Вороного (станции и магистраль отключены)")
        geo = _voronoi_only_placement_payload(
            city=city.strip(),
            network_type=network_type,
            simplify=simplify,
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
            buildings_per_centroid=buildings_per_centroid,
            inter_cluster_max_hull_gap_m=inter_cluster_max_hull_gap_m,
            inter_cluster_max_edge_length_m=inter_cluster_max_edge_length_m,
            use_all_buildings=use_all_buildings,
            voronoi_intra_bridge_max_m=float(voronoi_intra_bridge_max_m),
            from_saved=False,
        )
        n_vor = len((geo.get("voronoi_edges") or {}).get("features", []) or [])
        logger.info("Пайплайн завершён: рёбер Вороного=%s", n_vor)
        if save_result and redis_client is not None:
            try:
                redis_client.set(
                    cache_key,
                    json.dumps(geo, ensure_ascii=False).encode("utf-8"),
                )
            except Exception as e:
                logger.warning("Не удалось сохранить расстановку в Redis: %s", e)
        return JSONResponse(geo)
    except Exception as e:
        logger.exception("stations placement")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stations/export_saved")
def export_saved_stations(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
    dbscan_eps_m: float = Query(180.0, description="eps DBSCAN (м)"),
    dbscan_min_samples: int = Query(15, description="min_samples DBSCAN"),
    buildings_per_centroid: int = Query(60, description="Должен совпадать с параметром при сохранении"),
    inter_cluster_max_hull_gap_m: float = Query(
        2000.0,
        description="Должен совпадать с параметром при сохранении",
    ),
    inter_cluster_max_edge_length_m: float = Query(
        2000.0,
        description="Должен совпадать с параметром при сохранении",
    ),
    use_all_buildings: bool = Query(False, description="Должен совпадать с параметром при сохранении"),
    voronoi_intra_bridge_max_m: float = Query(
        600.0,
        description="Должен совпадать с параметром при сохранении",
    ),
):
    """Выгрузка сохранённого JSON карты из Redis (режим только Вороного)."""
    redis_client = data_service.get_redis_client()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis недоступен")
    cache_key = _placement_cache_key(
        city=city,
        network_type=network_type,
        simplify=simplify,
        dbscan_eps_m=dbscan_eps_m,
        dbscan_min_samples=dbscan_min_samples,
        buildings_per_centroid=buildings_per_centroid,
        inter_cluster_max_hull_gap_m=inter_cluster_max_hull_gap_m,
        inter_cluster_max_edge_length_m=inter_cluster_max_edge_length_m,
        use_all_buildings=use_all_buildings,
        voronoi_intra_bridge_max_m=float(voronoi_intra_bridge_max_m),
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
