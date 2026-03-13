import os
import re
from typing import Optional
import logging
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from data_service import DataService
from station_placement import run_full_pipeline, R_CHARGE_KM, R_GARAGE_TO_KM, D_MAX_KM, StationPlacement
import geopandas as gpd
import osmnx as ox


app = FastAPI(title="Graph Service UI", version="0.1.0")

# CORS (на случай локального фронтенда)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_service = DataService()
logger = logging.getLogger(__name__)

def _placement_slug(city: str) -> str:
    """Нормализует название города в имя файла."""
    s = city.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s or "default"


def save_placement(city: str, data: dict) -> None:
    """Сохраняет результат размещения в Redis (без использования локального диска)."""
    redis_client = data_service.get_redis_client()
    if redis_client is None:
        logger.warning("Redis недоступен, размещение станций не будет закэшировано")
        return
    try:
        key = f"drone_planner:placement:{_placement_slug(city)}"
        redis_client.set(key, json.dumps(data, ensure_ascii=False))
        logger.info("Сохранено размещение в Redis: %s", key)
    except Exception as e:
        logger.warning("Не удалось сохранить размещение в Redis: %s", e)


def load_placement(city: str) -> Optional[dict]:
    """Загружает сохранённое размещение из Redis. None если записи нет или Redis недоступен."""
    redis_client = data_service.get_redis_client()
    if redis_client is None:
        logger.warning("Redis недоступен, нет кэша размещений")
        return None
    try:
        key = f"drone_planner:placement:{_placement_slug(city)}"
        blob = redis_client.get(key)
        if not blob:
            return None
        # Redis возвращает bytes — декодируем и парсим JSON
        if isinstance(blob, bytes):
            blob = blob.decode("utf-8", errors="ignore")
        return json.loads(blob)
    except Exception as e:
        logger.warning("Не удалось загрузить размещение из Redis: %s", e)
        return None


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>UI не найден. Создайте templates/index.html</h1>", status_code=200)


@app.get("/api/ping")
def ping():
    return {"status": "ok"}


@app.get("/api/buildings/clusters")
def get_building_clusters(
    city: str = Query(..., description="Название города, как в /api/city"),
    network_type: str = Query('drive', description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
    method: str = Query("dbscan", description="Метод кластеризации: dbscan или grid"),
    dbscan_eps_m: float = Query(180.0, description="Радиус окрестности DBSCAN (м), 120–180 — кварталы, 200–300 — микрорайоны"),
    dbscan_min_samples: int = Query(15, description="Мин. точек в кластере DBSCAN (10–25, 15 — компромисс)"),
    use_all_buildings: bool = Query(False, description="Если True — кластеризация по всем зданиям (исключая здания в беспилотных зонах)"),
):
    """
    Возвращает кластеры спроса: DBSCAN (центроиды кластеров с весом) или точки высадки у дороги (buildings).
    Для отображения на карте используйте method=dbscan.
    """
    import logging
    logger = logging.getLogger(__name__)

    def round_coords(obj, precision=6):
        if isinstance(obj, dict):
            return {k: round_coords(v, precision) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_coords(item, precision) for item in obj]
        elif isinstance(obj, float):
            return round(obj, precision)
        return obj

    try:
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
        buildings = data.get("buildings")
        road_graph = data.get("road_graph")
        city_boundary = data.get("city_boundary")
        if buildings is None or len(buildings) == 0:
            logger.warning("Нет зданий")
            return JSONResponse({
                "buildings": {"type": "FeatureCollection", "features": []},
                "clusters": {"type": "FeatureCollection", "features": []},
                "total": 0,
                "message": "Нет зданий",
            })

        if method == "dbscan":
            # Кластеры DBSCAN: сначала пытаемся взять предвычисленные кластеры,
            # выделенные на этапе get_city_data (и уже нарезанные до <=100 точек).
            demand_gdf = data.get("demand_points")
            hulls_gdf = data.get("demand_hulls")
            if demand_gdf is None or len(demand_gdf) == 0:
                # Если предвычисленных кластеров нет (старый кэш или ошибка) — считаем как раньше
                no_fly_zones = data.get("no_fly_zones")
                result = data_service.get_demand_points_weighted(
                    buildings, road_graph, city_boundary,
                    method="dbscan",
                    dbscan_eps_m=dbscan_eps_m,
                    dbscan_min_samples=dbscan_min_samples,
                    return_hulls=True,
                    use_all_buildings=use_all_buildings,
                    no_fly_zones=no_fly_zones,
                )
                demand_gdf, hulls_gdf = result if isinstance(result, tuple) else (result, None)
                # И сразу режем крупные кластеры, чтобы в одном итоговом кластере
                # было не более 100 точек/зданий.
                if demand_gdf is not None and len(demand_gdf) > 0:
                    try:
                        placement = StationPlacement(data_service, logger=logger)
                        if "weight" in demand_gdf.columns:
                            total_weight = float(demand_gdf["weight"].sum())
                            # Целимся примерно в такие кластеры, чтобы их было хотя бы len(demand_gdf)/300,
                            # но не делаем cap слишком маленьким.
                            approx_clusters = max(1, len(demand_gdf) // 300)
                            max_weight_per_cluster = max(total_weight / approx_clusters, 1.0)
                        else:
                            max_weight_per_cluster = float("inf")
                        demand_gdf = placement.split_demand_by_capacity(
                            demand_gdf,
                            max_weight_per_cluster=max_weight_per_cluster,
                            max_points_per_cluster=300,
                            cluster_id_column=None,
                            k_neighbors=16,
                        )
                    except Exception:
                        pass
            if demand_gdf is None or len(demand_gdf) == 0:
                clusters_fc = {"type": "FeatureCollection", "features": []}
                cluster_hulls_fc = {"type": "FeatureCollection", "features": []}
            else:
                features = []
                for idx, row in demand_gdf.iterrows():
                    g = row["geometry"]
                    if g is None:
                        continue
                    lon, lat = g.x, g.y
                    weight = int(row.get("weight", 1))
                    props = {"weight": weight, "cluster_id": idx}
                    if "demand_type" in row and row.get("demand_type") is not None:
                        props["demand_type"] = str(row["demand_type"])
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                        "properties": props,
                    })
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
                            geom = geom_json.get("features", [{}])[0].get("geometry")
                            if geom:
                                # Вес области спроса не должен превышать 300
                                w = min(300, int(row.get("weight", 1)))
                                props = {
                                    "weight": w,
                                    "cluster_id": idx,
                                }
                                # Пробрасываем тип спроса (тип зданий) для стилизации на фронтенде
                                if "demand_type" in row:
                                    props["demand_type"] = row.get("demand_type")
                                hull_features.append({
                                    "type": "Feature",
                                    "geometry": round_coords(geom, precision=6),
                                    "properties": props,
                                })
                        except Exception:
                            pass
                    cluster_hulls_fc = {"type": "FeatureCollection", "features": hull_features}
            buildings_fc = clusters_fc  # для совместимости
            total = len(clusters_fc.get("features", []))
            logger.info(f"Кластеры DBSCAN: {total}, областей (hulls): {len(cluster_hulls_fc.get('features', []))}")
        else:
            # Точки высадки у дороги (как раньше)
            if road_graph is None or len(road_graph.nodes) == 0:
                logger.warning("Нет дорожного графа для расчёта точек высадки")
            delivery_gdf = data_service.get_delivery_points(buildings, road_graph, city_boundary)
            if len(delivery_gdf) == 0:
                return JSONResponse({
                    "buildings": {"type": "FeatureCollection", "features": []},
                    "clusters": {"type": "FeatureCollection", "features": []},
                    "total": 0,
                    "message": "Нет зданий в границах города",
                })
            features = []
            for idx, row in delivery_gdf.iterrows():
                lon_d, lat_d = row["delivery_lon"], row["delivery_lat"]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon_d, lat_d]},
                    "properties": {"delivery_lon": round(lon_d, 6), "delivery_lat": round(lat_d, 6)},
                })
            buildings_fc = round_coords({"type": "FeatureCollection", "features": features}, precision=6)
            clusters_fc = buildings_fc
            total = len(features)
            logger.info(f"Точек высадки (grid): {total}")

        resp = {
            "buildings": buildings_fc,
            "clusters": clusters_fc,
            "total": total,
            "method": method,
        }
        if method == "dbscan":
            resp["cluster_hulls"] = cluster_hulls_fc
            resp["dbscan_eps_m"] = dbscan_eps_m
        else:
            resp["cluster_hulls"] = {"type": "FeatureCollection", "features": []}
        return JSONResponse(resp)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/buildings/export")
def export_buildings_with_heights(
    city: str = Query(..., description="Название города, как в /api/city"),
    network_type: str = Query('drive', description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение"),
):
    """
    Выгрузка зданий с высотами, площадями и эшелонами полётов.
    GeoJSON с полигонами зданий и properties: height_m, area_m2, flight_level (рекомендуемый эшелон).
    """
    try:
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
        buildings = data.get("buildings")
        flight_levels = data.get("flight_levels", [])
        if buildings is None or len(buildings) == 0:
            return JSONResponse({
                "type": "FeatureCollection",
                "features": [],
                "flight_levels": flight_levels,
                "building_height_stats": data.get("building_height_stats", {}),
            })

        if "height_m" not in buildings.columns:
            buildings = data_service._compute_building_heights(buildings)

        # Площадь зданий в м² (по возможности в метрической проекции)
        if "area_m2" not in buildings.columns:
            try:
                b_proj = buildings
                if b_proj.crs is None:
                    b_proj = b_proj.set_crs("EPSG:4326", allow_override=True)
                if b_proj.crs and b_proj.crs.is_geographic:
                    b_proj = b_proj.to_crs(epsg=3857)
                areas = b_proj.geometry.area.astype(float)
                buildings["area_m2"] = areas
            except Exception:
                try:
                    buildings["area_m2"] = buildings.geometry.area.astype(float)
                except Exception:
                    buildings["area_m2"] = None

        # GeoJSON: полигоны зданий + height_m, area_m2, рекомендованный flight_level
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
            # Рекомендуемый эшелон: первый, где altitude > h + 10м
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
        return JSONResponse({
            "type": "FeatureCollection",
            "features": features,
            "flight_levels": flight_levels,
            "building_height_stats": data.get("building_height_stats", {}),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/city")
def get_city(
    city: str = Query(..., description="Название города, например 'Volgograd, Russia'"),
    network_type: str = Query('drive', description="Тип сети OSM: drive, walk, bike, all, all_private"),
    simplify: bool = Query(True, description="Упрощать ли граф"),
):
    try:
        progress_events = []

        def _progress(stage: str, percentage: int, message: str = ""):
            progress_events.append({
                "stage": stage,
                "percentage": percentage,
                "message": message,
            })

        data_service.add_progress_callback(_progress)
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)

        # Для передачи в JSON: убираем тяжёлые объекты и возвращаем краткую сводку
        stats = data.get("stats", {})
        result = {
            "city_name": data.get("city_name"),
            "stats": stats,
            "params": data.get("params", {}),
            "flight_levels": data.get("flight_levels", []),
            "building_height_stats": data.get("building_height_stats", {}),
            "progress": progress_events,
        }
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/city/map")
def get_city_map(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение"),
):
    """Возвращает границы города, no_fly_zones и bbox/center для отрисовки карты (без графов)."""
    try:
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify, load_no_fly_zones=True)
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
                logger.warning(f"Ошибка сериализации no_fly_zones: {e}")
        if city_boundary is not None:
            boundary_gdf = gpd.GeoDataFrame([{"geometry": city_boundary}], crs="EPSG:4326")
            city_boundary_geojson = json.loads(boundary_gdf.to_json())
            bbox = list(city_boundary.bounds)
            center_lon = (bbox[0] + bbox[2]) / 2.0
            center_lat = (bbox[1] + bbox[3]) / 2.0
        elif road_graph is not None and len(road_graph.nodes) > 0:
            gdf_nodes, _ = ox.graph_to_gdfs(road_graph)
            xmin, ymin, xmax, ymax = gdf_nodes.total_bounds
            bbox = [xmin, ymin, xmax, ymax]
            center_lat = (ymin + ymax) / 2.0
            center_lon = (xmin + xmax) / 2.0
            city_boundary_geojson = None
        else:
            raise HTTPException(status_code=404, detail="Нет данных по городу")
        return JSONResponse({
            "city_boundary": city_boundary_geojson,
            "no_fly_zones": no_fly_zones_geojson,
            "bbox": bbox,
            "center": {"lat": center_lat, "lon": center_lon},
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stations/placement")
def get_stations_placement(
    city: str = Query(..., description="Название города"),
    load_saved: bool = Query(False, description="Загрузить сохранённое размещение с диска (без пересчёта)"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение"),
    demand_method: str = Query("dbscan", description="Метод спроса: dbscan или grid"),
    demand_cell_m: float = Query(250.0, description="Размер ячейки спроса при method=grid (м)"),
    dbscan_eps_m: float = Query(180.0, description="Радиус окрестности DBSCAN (м)"),
    dbscan_min_samples: int = Query(15, description="Мин. точек в кластере DBSCAN"),
    use_all_buildings: bool = Query(False, description="Кластеризация по всем зданиям вне беспилотных зон"),
    max_charge_stations: Optional[int] = Query(None, description="Макс. число зарядок (по умолчанию — до покрытия)"),
    num_garages: int = Query(1, ge=0, description="Число гаражей (0 = не размещать)"),
    num_to: int = Query(1, ge=0, description="Число станций ТО (0 = не размещать, 1 = одна станция)"),
):
    """
    Размещение станций: спрос по DBSCAN (по умолчанию) или по сетке.
    При load_saved=True возвращает последнее сохранённое размещение для города (без пересчёта).
    Иначе считает размещение и сохраняет его на диск (переживает перезапуск сервера).
    """
    try:
        if load_saved:
            cached = load_placement(city)
            if cached is not None:
                return JSONResponse(cached)
            return JSONResponse(
                {"error": "Нет сохранённого размещения для этого города. Нажмите «Разместить станции»."},
                status_code=404,
            )

        result = run_full_pipeline(
            data_service,
            city,
            network_type=network_type,
            simplify=simplify,
            demand_method=demand_method,
            demand_cell_m=demand_cell_m,
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
            use_all_buildings=use_all_buildings,
            max_charge_stations=max_charge_stations,
            num_garages=num_garages,
            num_to=num_to,
        )
        if result.get("error"):
            return JSONResponse({"error": result["error"]}, status_code=400)

        def gdf_to_geojson(gdf, default_props=None):
            if gdf is None or len(gdf) == 0:
                return {"type": "FeatureCollection", "features": []}
            return json.loads(gdf.to_json())

        def round_coords(obj, precision=6):
            if isinstance(obj, dict):
                return {k: round_coords(v, precision) for k, v in obj.items()}
            if isinstance(obj, list):
                return [round_coords(x, precision) for x in obj]
            if isinstance(obj, float):
                return round(obj, precision)
            return obj

        demand_geojson = gdf_to_geojson(result.get("demand"))
        charge_geojson = gdf_to_geojson(result.get("charge_stations"))
        garages_geojson = gdf_to_geojson(result.get("garages"))
        to_geojson = gdf_to_geojson(result.get("to_stations"))

        # Рёбра магистрали (LineString) из trunk_graph
        trunk = result.get("trunk_graph")
        trunk_edges = []
        if trunk is not None and trunk.number_of_edges() > 0:
            for u, v, data in trunk.edges(data=True):
                # Если в данных ребра уже есть полилиния (geometry_coords) — используем её,
                # иначе строим простой отрезок между узлами (как раньше).
                coords = data.get("geometry_coords")
                if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    line_coords = [[float(lon), float(lat)] for lon, lat in coords]
                    weight_val = data.get("weight") or data.get("length")
                    trunk_edges.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": line_coords},
                        "properties": {"weight_km": weight_val},
                    })
                else:
                    nu, nv = trunk.nodes[u], trunk.nodes[v]
                    lon1 = nu.get("lon") or nu.get("x")
                    lat1 = nu.get("lat") or nu.get("y")
                    lon2 = nv.get("lon") or nv.get("x")
                    lat2 = nv.get("lat") or nv.get("y")
                    if lon1 is not None and lat1 is not None and lon2 is not None and lat2 is not None:
                        trunk_edges.append({
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
                            "properties": {"weight_km": data.get("weight") or data.get("length")},
                        })
        trunk_fc = round_coords({"type": "FeatureCollection", "features": trunk_edges})

        # Ветки от станций типа Б к станциям типа А (связь, не магистраль)
        branch_edges = result.get("branch_edges") or []
        branch_fc = round_coords({"type": "FeatureCollection", "features": branch_edges})
        # Локальные рёбра между вспомогательными станциями (Б↔Б)
        local_edges = result.get("local_edges") or []
        local_fc = round_coords({"type": "FeatureCollection", "features": local_edges})

        # Граница города для карты
        city_boundary_geojson = None
        if result.get("city_boundary") is not None:
            try:
                boundary_gdf = gpd.GeoDataFrame([{"geometry": result["city_boundary"]}], crs="EPSG:4326")
                city_boundary_geojson = json.loads(boundary_gdf.to_json())
            except Exception:
                pass

        response_data = {
            "demand": round_coords(demand_geojson),
            "charge_stations": round_coords(charge_geojson),
            "garages": round_coords(garages_geojson),
            "to_stations": round_coords(to_geojson),
            "trunk_edges": trunk_fc,
            "branch_edges": branch_fc,
            "local_edges": local_fc,
            "metrics": result.get("metrics", {}),
            "params": {
                "R_charge_km": R_CHARGE_KM,
                "R_garage_to_km": R_GARAGE_TO_KM,
                "D_max_km": D_MAX_KM,
            },
            "city_boundary": city_boundary_geojson,
        }
        save_placement(city, response_data)
        return JSONResponse(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _ensure_static_mount():
    static_dir = os.path.join(os.getcwd(), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")


_ensure_static_mount()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)