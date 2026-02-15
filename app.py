import os
from typing import Optional
import pickle
import logging

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from data_service import DataService
from graph_service import GraphService
import json
import osmnx as ox
import geopandas as gpd
import networkx as nx


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
graph_service = GraphService()
logger = logging.getLogger(__name__)


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
):
    """
    Возвращает точки на зданиях (центроиды) и точку высадки у дороги для каждого здания.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
        buildings = data.get("buildings")
        road_graph = data.get("road_graph")
        city_boundary = data.get("city_boundary")
        if buildings is None or len(buildings) == 0:
            logger.warning("Нет зданий")
            return JSONResponse({
                "buildings": {"type": "FeatureCollection", "features": []},
                "total": 0,
                "message": "Нет зданий"
            })
        if road_graph is None or len(road_graph.nodes) == 0:
            logger.warning("Нет дорожного графа для расчёта точек высадки")
        
        # Точки высадки — рядом с домами на дороге
        delivery_gdf = data_service.get_delivery_points(buildings, road_graph, city_boundary)
        if len(delivery_gdf) == 0:
            return JSONResponse({
                "buildings": {"type": "FeatureCollection", "features": []},
                "total": 0,
                "message": "Нет зданий в границах города"
            })
        
        # GeoJSON: geometry = точка у дороги (высадка), не центроид здания
        features = []
        for idx, row in delivery_gdf.iterrows():
            lon_d = row["delivery_lon"]
            lat_d = row["delivery_lat"]
            feat = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon_d, lat_d]},
                "properties": {
                    "delivery_lon": round(lon_d, 6),
                    "delivery_lat": round(lat_d, 6),
                },
            }
            features.append(feat)
        
        def round_coords(obj, precision=6):
            if isinstance(obj, dict):
                return {k: round_coords(v, precision) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [round_coords(item, precision) for item in obj]
            elif isinstance(obj, float):
                return round(obj, precision)
            return obj
        
        geojson_data = round_coords({
            "type": "FeatureCollection",
            "features": features,
        }, precision=6)
        n_features = len(features)
        # Один слой для фронта (clusters ожидается кодом карты)
        clusters = {"buildings": geojson_data}
        logger.info(f"Точек на зданиях (высадка у дороги): {n_features}")
        
        return JSONResponse({
            "buildings": geojson_data,
            "clusters": clusters,
            "total": n_features
        })
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
    Выгрузка зданий с высотами и эшелонами полётов.
    GeoJSON с полигонами зданий и properties: height_m, flight_level (рекомендуемый эшелон).
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
        # GeoJSON: полигоны зданий + height_m, рекомендованный flight_level
        levels_m = [f["altitude_m"] for f in flight_levels] if flight_levels else [40, 65, 90, 115]
        features = []
        for idx, row in buildings.iterrows():
            geom = row.geometry
            if geom is None or not getattr(geom, "is_valid", True):
                continue
            h = float(row.get("height_m", 10))
            # Рекомендуемый эшелон: первый, где altitude > h + 10м
            rec_level = 1
            for i, alt in enumerate(levels_m):
                if alt >= h + 10:
                    rec_level = i + 1
                    break
            feat = {
                "type": "Feature",
                "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
                "properties": {"height_m": round(h, 1), "flight_level": rec_level},
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


@app.get("/api/graph")
def graph_geojson(
    city: str = Query(..., description="Название города, как в /api/city"),
    network_type: str = Query('drive', description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
    graph_type: str = Query('road', description="Тип графа: road, roads_only, delaunay, merged"),
    grid_spacing: float = Query(0.001, description="Размер ячейки сетки для Delaunay (в градусах)"),
):
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Загружаем (или берём из кэша) данные города. Для "Только дороги" не грузим беспилотные зоны.
        load_no_fly = (graph_type != 'roads_only')
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify, load_no_fly_zones=load_no_fly)
        road_graph = data.get("road_graph")
        if road_graph is None:
            raise HTTPException(status_code=404, detail="Граф не найден")

        buildings = data.get("buildings")
        no_fly_zones = data.get("no_fly_zones", [])
        city_boundary = data.get("city_boundary")

        # ОПТИМИЗАЦИЯ: получаем границы только если нужно
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(road_graph)
        xmin, ymin, xmax, ymax = gdf_nodes.total_bounds
        center_lat = (ymin + ymax) / 2.0
        center_lon = (xmin + xmax) / 2.0
        
        # Если есть границы города, используем их для bounds и центра (только для не-road графов)
        if graph_type not in ('road', 'roads_only') and city_boundary is not None:
            try:
                boundary_bounds = city_boundary.bounds
                # Используем границы города вместо bounding box дорожного графа
                xmin, ymin, xmax, ymax = boundary_bounds[0], boundary_bounds[1], boundary_bounds[2], boundary_bounds[3]
                center_lon = (xmin + xmax) / 2.0
                center_lat = (ymin + ymax) / 2.0
                logger.info(f"✓ Используются границы города для bounds: {city_boundary.geom_type}")
                logger.info(f"  Границы: lon=[{xmin:.6f}, {xmax:.6f}], lat=[{ymin:.6f}, {ymax:.6f}]")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Ошибка использования границ города: {e}")
        elif graph_type not in ('road', 'roads_only'):
            import logging
            logging.getLogger(__name__).warning("⚠ Границы города не найдены! Используется bounding box")
        
        bounds = (xmin, ymin, xmax, ymax)

        # ОПТИМИЗАЦИЯ: Проверяем кэш графа в Redis
        redis_client = data_service.get_redis_client()
        cache_key = data_service.generate_graph_cache_key(
            city, network_type, simplify, graph_type, grid_spacing, False
        )
        
        # Пытаемся загрузить из кэша
        cached_graph_data = None
        if redis_client is not None:
            try:
                cached_blob = redis_client.get(cache_key)
                if cached_blob:
                    cached_graph_data = pickle.loads(cached_blob)
                    logger.info(f"✓ Граф загружен из Redis кэша: {cache_key}")
            except Exception as e:
                logger.warning(f"Ошибка чтения графа из Redis: {e}")

        # Генерируем нужный тип графа (roads_only — только дороги, без загрузки/фильтра по беспилотным зонам)
        if graph_type in ('road', 'roads_only'):
            # Фильтруем рёбра по беспилотным зонам только для типа "Дороги (OSM)"; для "Только дороги" — не трогаем
            if graph_type == 'road' and no_fly_zones is not None:
                try:
                    if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                        no_fly_union = no_fly_zones.geometry.unary_union
                    elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                        from shapely.ops import unary_union
                        no_fly_union = unary_union(no_fly_zones)
                    else:
                        no_fly_union = None
                    if no_fly_union is not None:
                        original_count = len(gdf_edges)
                        mask = ~gdf_edges.geometry.intersects(no_fly_union)
                        gdf_edges = gdf_edges[mask]
                        logger.info(f"Фильтрация дорожного графа: {original_count} -> {len(gdf_edges)} рёбер (удалено в беспилотных зонах)")
                except Exception as e:
                    logger.warning(f"Ошибка фильтрации дорожного графа по беспилотным зонам: {e}")
            try:
                edges_geojson = json.loads(gdf_edges.to_json())
                for f in edges_geojson.get("features", []):
                    p = f.get("properties") or {}
                    p["edge_type"] = "road"
                    f["properties"] = p
                logger.info(f"Отправлено {len(gdf_edges)} рёбер дорожного графа")
            except Exception as e:
                logger.error(f"Ошибка конвертации рёбер в GeoJSON: {e}")
                raise
            stats = data.get("stats", {})
            
        elif graph_type == 'delaunay':
            has_no_fly = no_fly_zones is not None and (
                (isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0) or
                (isinstance(no_fly_zones, list) and len(no_fly_zones) > 0)
            )
            if cached_graph_data is not None and not has_no_fly:
                edges_geojson = cached_graph_data.get('edges_geojson')
                stats = cached_graph_data.get('stats')
                logger.info("✓ Использован кэшированный delaunay граф")
            else:
                delivery_gdf = data_service.get_delivery_points(buildings, road_graph, city_boundary)
                delivery_points = []
                if len(delivery_gdf) > 0:
                    delivery_points = [(float(row["delivery_lon"]), float(row["delivery_lat"])) for _, row in delivery_gdf.iterrows()]
                flyable_points = graph_service.get_flyable_points(
                    city_boundary=city_boundary,
                    buildings=buildings,
                    no_fly_zones=no_fly_zones,
                    min_distance_deg=0.0003,
                    max_points=2500,
                )
                all_points = graph_service.merge_points_min_distance(
                    delivery_points, flyable_points, min_distance_deg=0.0002
                )
                delaunay_graph = graph_service.create_delaunay_graph(
                    bounds=bounds,
                    num_points=0,
                    buildings=buildings if len(all_points) < 3 else None,
                    no_fly_zones=no_fly_zones,
                    city_boundary=city_boundary,
                    points=all_points if len(all_points) >= 3 else None,
                )
                obstacles = graph_service._prepare_obstacles(None, no_fly_zones, 0.0001)
                edges_geojson = graph_service.graph_to_geojson(
                    delaunay_graph, obstacles=obstacles, include_node_ids=True
                )
                stats = {
                    "nodes": len(delaunay_graph.nodes),
                    "edges": len(delaunay_graph.edges),
                    "type": "delaunay"
                }
                if redis_client is not None and not has_no_fly:
                    try:
                        cache_data = {
                            'edges_geojson': edges_geojson,
                            'stats': stats,
                            'graph_type': graph_type
                        }
                        redis_client.set(cache_key, pickle.dumps(cache_data), ex=86400 * 7)
                        logger.info(f"✓ Delaunay граф сохранен в Redis: {cache_key}")
                    except Exception as e:
                        logger.warning(f"Ошибка сохранения delaunay графа в Redis: {e}")
            # Аннотируем рёбра допустимыми эшелонами (выполняется всегда)
            flight_levels = data.get("flight_levels", [])
            buildings_with_heights = data.get("buildings")
            if buildings_with_heights is not None and "height_m" not in buildings_with_heights.columns:
                buildings_with_heights = data_service._compute_building_heights(buildings_with_heights)
            edges_geojson = graph_service.assign_flight_levels_to_edges(
                edges_geojson,
                buildings_with_heights if buildings_with_heights is not None and len(buildings_with_heights) > 0 else gpd.GeoDataFrame(),
                flight_levels
            )
        elif graph_type == 'merged':
            # Дорожный граф: фильтруем по беспилотным зонам так же, как для road
            gdf_edges_for_road = gdf_edges
            if no_fly_zones is not None:
                try:
                    if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                        no_fly_union = no_fly_zones.geometry.unary_union
                    elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                        from shapely.ops import unary_union
                        no_fly_union = unary_union(no_fly_zones)
                    else:
                        no_fly_union = None
                    if no_fly_union is not None:
                        mask = ~gdf_edges.geometry.intersects(no_fly_union)
                        gdf_edges_for_road = gdf_edges[mask]
                except Exception as e:
                    logger.warning(f"Ошибка фильтрации дорог по беспилотным зонам: {e}")
            # Собираем граф дорог из gdf (для совместимости с merge: узлы с x, y)
            G_road = nx.Graph()
            for n, row in gdf_nodes.iterrows():
                try:
                    G_road.add_node(n, x=float(row.geometry.x), y=float(row.geometry.y))
                except Exception:
                    continue
            seen_road = set()
            for idx, row in gdf_edges_for_road.iterrows():
                u, v = (idx[0], idx[1]) if isinstance(idx, tuple) else (row.get('u'), row.get('v'))
                if u is None or v is None:
                    continue
                key = (min(u, v), max(u, v))
                if key in seen_road:
                    continue
                seen_road.add(key)
                length = float(row.get('length') or row.get('weight') or 0)
                G_road.add_edge(u, v, length=length, weight=length)
            # Delaunay граф (delivery + flyable точки в лесах/полях)
            delivery_gdf = data_service.get_delivery_points(buildings, road_graph, city_boundary)
            delivery_points = []
            if len(delivery_gdf) > 0:
                delivery_points = [(float(row["delivery_lon"]), float(row["delivery_lat"])) for _, row in delivery_gdf.iterrows()]
            flyable_points = graph_service.get_flyable_points(
                city_boundary=city_boundary,
                buildings=buildings,
                no_fly_zones=no_fly_zones,
                min_distance_deg=0.0003,
                max_points=2500,
            )
            all_points = graph_service.merge_points_min_distance(
                delivery_points, flyable_points, min_distance_deg=0.0002
            )
            delaunay_graph = graph_service.create_delaunay_graph(
                bounds=bounds,
                num_points=0,
                buildings=buildings if len(all_points) < 3 else None,
                no_fly_zones=no_fly_zones,
                city_boundary=city_boundary,
                points=all_points if len(all_points) >= 3 else None,
            )
            obstacles = graph_service._prepare_obstacles(None, no_fly_zones, 0.0001)
            merged_graph = graph_service.merge_road_and_delaunay(
                G_road, delaunay_graph, obstacles=obstacles
            )
            # Дороги рисуем по полной геометрии из OSM (gdf_edges), а не по отрезкам между узлами
            road_geojson = json.loads(gdf_edges_for_road.to_json())
            road_features = []
            for f in road_geojson.get("features", []):
                if f.get("geometry", {}).get("type") not in ("LineString", "MultiLineString"):
                    continue
                props = f.get("properties") or {}
                length_m = props.get("length")
                length_km = float(length_m) / 1000.0 if length_m is not None else 0.0
                f["properties"] = {
                    "edge_type": "road",
                    "weight": length_km,
                    "length": length_km,
                    **{k: v for k, v in props.items() if k not in ("edge_type", "weight", "length")},
                }
                road_features.append(f)
            free_fc = graph_service.graph_to_geojson(
                merged_graph, edge_type_filter="free", obstacles=obstacles, include_node_ids=True
            )
            conn_fc = graph_service.graph_to_geojson(
                merged_graph, edge_type_filter="connection", obstacles=obstacles, include_node_ids=True
            )
            edges_geojson = {
                "type": "FeatureCollection",
                "features": road_features + free_fc["features"] + conn_fc["features"],
            }
            if free_fc.get("nodes"):
                edges_geojson["nodes"] = free_fc["nodes"]
            n_road = sum(1 for n in merged_graph.nodes() if not str(n).startswith("d_"))
            n_free = merged_graph.number_of_nodes() - n_road
            n_conn = sum(1 for _u, _v, d in merged_graph.edges(data=True) if d.get("edge_type") == "connection")
            stats = {
                "nodes": merged_graph.number_of_nodes(),
                "edges": merged_graph.number_of_edges(),
                "type": "merged",
                "road_nodes": n_road,
                "free_nodes": n_free,
                "connection_edges": n_conn,
            }
            flight_levels = data.get("flight_levels", [])
            buildings_with_heights = data.get("buildings")
            if buildings_with_heights is not None and "height_m" not in buildings_with_heights.columns:
                buildings_with_heights = data_service._compute_building_heights(buildings_with_heights)
            edges_geojson = graph_service.assign_flight_levels_to_edges(
                edges_geojson,
                buildings_with_heights if buildings_with_heights is not None and len(buildings_with_heights) > 0 else gpd.GeoDataFrame(),
                flight_levels
            )
        else:
            raise HTTPException(status_code=400, detail=f"Неизвестный тип графа: {graph_type}")

        # Используем границы города для bbox, если они есть
        if city_boundary is not None:
            try:
                boundary_bounds = city_boundary.bounds
                bbox = [boundary_bounds[0], boundary_bounds[1], boundary_bounds[2], boundary_bounds[3]]
            except Exception:
                bbox = [xmin, ymin, xmax, ymax]
        else:
            bbox = [xmin, ymin, xmax, ymax]
        
        # Добавляем границы города в ответ для визуализации
        city_boundary_geojson = None
        if city_boundary is not None:
            try:
                boundary_gdf = gpd.GeoDataFrame([{'geometry': city_boundary}], crs='EPSG:4326')
                city_boundary_geojson = json.loads(boundary_gdf.to_json())
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Ошибка создания GeoJSON границ: {e}")
        
        # Добавляем беспилотные зоны в ответ для визуализации
        no_fly_zones_geojson = None
        if no_fly_zones is not None:
            try:
                # Преобразуем беспилотные зоны в GeoJSON
                if isinstance(no_fly_zones, gpd.GeoDataFrame):
                    if len(no_fly_zones) > 0:
                        no_fly_zones_geojson = json.loads(no_fly_zones.to_json())
                        logger.info(f"Добавлено {len(no_fly_zones)} беспилотных зон для визуализации")
                elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                    # Если это список геометрий, создаем GeoDataFrame
                    zones_gdf = gpd.GeoDataFrame([{'geometry': zone} for zone in no_fly_zones], crs='EPSG:4326')
                    no_fly_zones_geojson = json.loads(zones_gdf.to_json())
                    logger.info(f"Добавлено {len(no_fly_zones)} беспилотных зон для визуализации")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Ошибка создания GeoJSON беспилотных зон: {e}")
        
        # Тип "Только дороги" — не отдаём беспилотные зоны; границы города оставляем для ориентира
        if graph_type == 'roads_only':
            no_fly_zones_geojson = None
        
        nodes = edges_geojson.get("nodes", [])
        return JSONResponse({
            "bbox": bbox,
            "center": {"lat": center_lat, "lon": center_lon},
            "edges": edges_geojson,
            "stats": stats,
            "graph_type": graph_type,
            "city_boundary": city_boundary_geojson,
            "no_fly_zones": no_fly_zones_geojson,
            "flight_levels": data.get("flight_levels", []),
            "building_height_stats": data.get("building_height_stats", {}),
            "nodes": nodes,
            "multilevel_planning": bool(nodes),
        })
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/path")
def compute_path_multilevel(
    edges: dict = Body(..., description="edges GeoJSON с nodes и allowed_flight_levels"),
    flight_levels: list = Body(..., description="Список эшелонов"),
    source_node: str = Body(..., description="ID узла старта"),
    target_node: str = Body(..., description="ID узла цели"),
    source_level: Optional[int] = Body(None, description="Эшелон старта (опционально)"),
    target_level: Optional[int] = Body(None, description="Эшелон цели (опционально)"),
):
    """
    Строит маршрут с возможностью перехода между эшелонами (например, спуск с 2 на 1 для экономии батареи).
    Возвращает путь как список (node_id, level) и 3D-кривую: высота плавно меняется вдоль сегментов.
    """
    if not edges.get("nodes"):
        raise HTTPException(status_code=400, detail="Для расчёта маршрута нужны nodes в edges")
    try:
        G = graph_service.graph_from_geojson_with_nodes(edges)
        path, curve = graph_service.shortest_path_multilevel(
            G, edges, flight_levels,
            source_node=source_node, target_node=target_node,
            source_level=source_level, target_level=target_level,
        )
        return {"path": path, "path_3d_curve": curve}
    except Exception as e:
        logger.warning(f"Ошибка расчёта маршрута: {e}")
        raise HTTPException(status_code=400, detail=str(e))


def _ensure_static_mount():
    static_dir = os.path.join(os.getcwd(), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")


_ensure_static_mount()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)