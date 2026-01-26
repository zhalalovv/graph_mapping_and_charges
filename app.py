import os
from typing import Optional
import pickle
import logging

from fastapi import FastAPI, HTTPException, Query
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
    Возвращает кластеризованные здания по типам
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Загружаем данные города
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
        classified_buildings = data.get("classified_buildings")
        
        # Если классифицированных зданий нет, но есть обычные здания - классифицируем их
        if classified_buildings is None or len(classified_buildings) == 0:
            buildings = data.get("buildings")
            if buildings is not None and len(buildings) > 0:
                logger.info("Классифицируем здания на лету")
                classified_buildings = data_service.classify_buildings(buildings)
            else:
                logger.warning("Нет зданий для кластеризации")
                return JSONResponse({
                    "clusters": {},
                    "total": 0,
                    "message": "Нет зданий для кластеризации"
                })
        
        if len(classified_buildings) == 0:
            logger.warning("Классифицированные здания пусты")
            return JSONResponse({
                "clusters": {},
                "total": 0,
                "message": "Нет зданий для кластеризации"
            })
        
        total_buildings = len(classified_buildings)
        logger.info(f"Всего зданий для кластеризации: {total_buildings}")
        
        # ФИЛЬТРАЦИЯ: отфильтровываем здания только в границах города
        city_boundary = data.get("city_boundary")
        if city_boundary is not None:
            try:
                logger.info("Фильтрация зданий по границам города...")
                # Проверяем, находятся ли центроиды зданий внутри границ города
                buildings_centroids = classified_buildings.geometry.centroid
                mask = buildings_centroids.within(city_boundary)
                classified_buildings = classified_buildings[mask]
                filtered_count = len(classified_buildings)
                logger.info(f"Отфильтровано зданий: {total_buildings} -> {filtered_count} (в границах города)")
            except Exception as e:
                logger.warning(f"Ошибка фильтрации зданий по границам города: {e}")
                # Продолжаем без фильтрации, если произошла ошибка
        
        if len(classified_buildings) == 0:
            logger.warning("После фильтрации зданий не осталось")
            return JSONResponse({
                "clusters": {},
                "total": 0,
                "message": "Нет зданий в границах города"
            })
        
        # Группируем здания по типам и конвертируем в точки (центроиды)
        # ОПТИМИЗАЦИЯ: используем точки вместо полигонов для минимального размера данных
        clusters = {}
        for building_type in classified_buildings['building_type'].unique():
            type_buildings = classified_buildings[classified_buildings['building_type'] == building_type]
            
            if len(type_buildings) > 0:
                try:
                    logger.info(f"Конвертация типа {building_type}: {len(type_buildings)} зданий в точки...")
                    
                    # Создаем точки из центроидов зданий
                    from shapely.geometry import Point
                    points_gdf = type_buildings.copy()
                    points_gdf['geometry'] = type_buildings.geometry.centroid
                    
                    # Конвертируем в GeoJSON
                    geojson_str = points_gdf.to_json(na='null', show_bbox=False)
                    geojson_data = json.loads(geojson_str)
                    
                    # Округляем координаты до 6 знаков после запятой
                    def round_coords(obj, precision=6):
                        if isinstance(obj, dict):
                            return {k: round_coords(v, precision) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [round_coords(item, precision) for item in obj]
                        elif isinstance(obj, float):
                            return round(obj, precision)
                        return obj
                    
                    geojson_data = round_coords(geojson_data, precision=6)
                    
                    geojson_size_mb = len(json.dumps(geojson_data)) / (1024 * 1024)
                    logger.info(f"  Размер GeoJSON (точки): {geojson_size_mb:.2f} MB")
                    
                    clusters[building_type] = geojson_data
                    logger.info(f"✓ Тип {building_type}: {len(type_buildings)} точек успешно создано")
                    
                except Exception as e:
                    logger.error(f"Критическая ошибка для {building_type}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Пропускаем этот тип, но продолжаем обработку остальных
        
        # Подсчитываем статистику
        type_counts = classified_buildings['building_type'].value_counts().to_dict()
        
        # Логируем статистику в терминал
        logger.info("=" * 60)
        logger.info("СТАТИСТИКА ЗДАНИЙ ПО ТИПАМ:")
        logger.info(f"Всего зданий: {len(classified_buildings)}")
        for btype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            type_names = {
                'apartment': 'Многоквартирные дома',
                'house': 'Частные дома',
                'industrial': 'Промышленные здания',
                'commercial': 'Коммерческие здания',
                'office': 'Офисные здания',
                'public': 'Общественные здания',
                'other': 'Прочие здания'
            }
            type_name = type_names.get(btype, btype)
            percentage = (count / len(classified_buildings) * 100) if len(classified_buildings) > 0 else 0
            logger.info(f"  {type_name:30} {count:6} ({percentage:5.1f}%)")
        logger.info("=" * 60)
        
        # Подсчитываем общий размер данных
        total_clusters = len(clusters)
        total_features = sum(len(cluster.get('features', [])) for cluster in clusters.values())
        logger.info(f"Итого: {total_clusters} типов зданий, {total_features} features в кластерах")
        
        response_data = {
            "clusters": clusters,
            "statistics": type_counts,
            "total": len(classified_buildings),
            "types": list(clusters.keys())
        }
        
        # Логируем размер ответа
        try:
            import sys
            response_size = sys.getsizeof(str(response_data)) / (1024 * 1024)
            logger.info(f"Приблизительный размер ответа: {response_size:.2f} MB")
        except:
            pass
        
        return JSONResponse(response_data)
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
    graph_type: str = Query('road', description="Тип графа: road, delaunay"),
    grid_spacing: float = Query(0.001, description="Размер ячейки сетки для Delaunay (в градусах)"),
):
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Загружаем (или берём из кэша) данные города
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
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
        if graph_type != 'road' and city_boundary is not None:
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
        elif graph_type != 'road':
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

        # Генерируем нужный тип графа
        if graph_type == 'road':
            # Только дорожный граф
            # Для road графа используем данные из кэша города (не кэшируем отдельно)
            # ОПТИМИЗАЦИЯ: фильтруем рёбра дорожного графа, которые пересекают беспилотные зоны
            if no_fly_zones is not None:
                try:
                    # Создаем объединенную геометрию беспилотных зон
                    if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                        no_fly_union = no_fly_zones.geometry.unary_union
                    elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                        from shapely.ops import unary_union
                        no_fly_union = unary_union(no_fly_zones)
                    else:
                        no_fly_union = None
                    
                    if no_fly_union is not None:
                        # Фильтруем рёбра, которые пересекают беспилотные зоны
                        original_count = len(gdf_edges)
                        # Проверяем пересечение каждого ребра с беспилотными зонами
                        mask = ~gdf_edges.geometry.intersects(no_fly_union)
                        gdf_edges_filtered = gdf_edges[mask]
                        filtered_count = len(gdf_edges_filtered)
                        logger.info(f"Фильтрация дорожного графа: {original_count} -> {filtered_count} рёбер (удалено {original_count - filtered_count} в беспилотных зонах)")
                        gdf_edges = gdf_edges_filtered
                    else:
                        pass  # Используем gdf_edges как есть
                except Exception as e:
                    logger.warning(f"Ошибка фильтрации дорожного графа по беспилотным зонам: {e}")
                    # Продолжаем с исходными рёбрами
                
                # Отправляем ВСЕ рёбра без ограничений
                try:
                    edges_geojson = json.loads(gdf_edges.to_json())
                    logger.info(f"Отправлено {len(gdf_edges)} рёбер дорожного графа")
                except Exception as e:
                    logger.error(f"Ошибка конвертации рёбер в GeoJSON: {e}")
                    raise
            else:
                # Отправляем ВСЕ рёбра без ограничений
                try:
                    edges_geojson = json.loads(gdf_edges.to_json())
                    logger.info(f"Отправлено {len(gdf_edges)} рёбер дорожного графа")
                except Exception as e:
                    logger.error(f"Ошибка конвертации рёбер в GeoJSON: {e}")
                    raise
            stats = data.get("stats", {})
            
        elif graph_type == 'delaunay':
            # Проверяем кэш
            if cached_graph_data is not None:
                edges_geojson = cached_graph_data.get('edges_geojson')
                stats = cached_graph_data.get('stats')
                logger.info("✓ Использован кэшированный delaunay граф")
            else:
                # Delaunay триангуляция
                num_points = int((xmax - xmin) * (ymax - ymin) / (grid_spacing ** 2))
                num_points = min(max(num_points, 100), 1000)  # ограничиваем 100-1000
                
                delaunay_graph = graph_service.create_delaunay_graph(
                    bounds=bounds,
                    num_points=num_points,
                    buildings=buildings,
                    no_fly_zones=no_fly_zones,
                    city_boundary=city_boundary
                )
                edges_geojson = graph_service.graph_to_geojson(delaunay_graph)
                stats = {
                    "nodes": len(delaunay_graph.nodes),
                    "edges": len(delaunay_graph.edges),
                    "type": "delaunay"
                }
                
                # Сохраняем в Redis
                if redis_client is not None:
                    try:
                        cache_data = {
                            'edges_geojson': edges_geojson,
                            'stats': stats,
                            'graph_type': graph_type
                        }
                        redis_client.set(cache_key, pickle.dumps(cache_data), ex=86400 * 7)  # 7 дней
                        logger.info(f"✓ Delaunay граф сохранен в Redis: {cache_key}")
                    except Exception as e:
                        logger.warning(f"Ошибка сохранения delaunay графа в Redis: {e}")
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
        
        return JSONResponse({
            "bbox": bbox,
            "center": {"lat": center_lat, "lon": center_lon},
            "edges": edges_geojson,
            "stats": stats,
            "graph_type": graph_type,
            "city_boundary": city_boundary_geojson,  # Границы города для визуализации
            "no_fly_zones": no_fly_zones_geojson,  # Беспилотные зоны для визуализации
        })
    except HTTPException:
        raise
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