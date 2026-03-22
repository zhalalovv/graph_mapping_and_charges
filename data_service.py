import os
import warnings
from typing import Optional, Union, List

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import pickle
import requests
from geopy.geocoders import Nominatim
import logging
import json
from redis import Redis
from shapely.geometry import box, MultiPoint, Point, Polygon
from shapely.ops import unary_union

# Подавляем DeprecationWarning из OSMnx (unary_union и др. — внутренние вызовы библиотеки)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="osmnx")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"osmnx\.features")


class DataService:
    def __init__(self, cache_dir="cache_data"):
        # Локальный кэш на диске больше не используется как хранилище данных.
        # Параметр оставлен для совместимости, но директории мы не создаём и файлы не пишем.
        self.cache_dir = cache_dir
        self.progress_callbacks = []
        self._nominatim_geocoder = Nominatim(user_agent="city_cluster_app", timeout=10)
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Redis — основное хранилище кэша тяжёлых данных города
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            self._redis: Redis | None = Redis.from_url(redis_url)
            # do not decode responses here; we store bytes for pickle blobs
            self._redis.ping()
        except Exception:
            self._redis = None
            # Локальный диск для кэша данных больше не используется
            self.logger.info("Redis not available for DataService cache; диск как кэш не используется")
    
    def add_progress_callback(self, callback):
        self.progress_callbacks.append(callback)
    
    def _update_progress(self, stage, percentage, message=""):
        for callback in self.progress_callbacks:
            callback(stage, percentage, message)
    
    def get_city_data(self, city_name: str, network_type: str = 'drive', simplify: bool = True, load_no_fly_zones: bool = True):
        """Получение данных города (универсально для любой страны). load_no_fly_zones=False — только дороги, без загрузки беспилотных зон."""
        normalized_name = city_name.strip()
        # Для совместимости: при load_no_fly_zones=True ключ без суффикса (старый кэш); при False — __nofly0
        base_suffix = f"{self._sanitize_name(normalized_name)}__{self._sanitize_name(network_type)}__{int(bool(simplify))}"
        key_suffix = base_suffix if load_no_fly_zones else f"{base_suffix}__nofly0"
        redis_key = f"drone_planner:city:{key_suffix}"
        redis_clusters_key = f"drone_planner:city_clusters:{key_suffix}"
        
        # Try Redis first
        if self._redis is not None:
            try:
                blob = self._redis.get(redis_key)
                if blob:
                    base_data = pickle.loads(blob)
                    # Проверяем, есть ли границы города в кэше
                    if 'city_boundary' not in base_data:
                        self.logger.info("В Redis кэше нет границ города, перезагружаем данные")
                        # Удаляем из Redis и загружаем заново
                        self._redis.delete(redis_key)
                        self._redis.delete(redis_clusters_key)
                    else:
                        # Пытаемся подгрузить предвычисленные кластеры из отдельного ключа
                        have_clusters = False
                        if self._redis is not None:
                            try:
                                clusters_blob = self._redis.get(redis_clusters_key)
                                if clusters_blob:
                                    clusters_data = pickle.loads(clusters_blob)
                                    if isinstance(clusters_data, dict):
                                        if 'demand_points' in clusters_data:
                                            base_data['demand_points'] = clusters_data['demand_points']
                                        if 'demand_hulls' in clusters_data:
                                            base_data['demand_hulls'] = clusters_data['demand_hulls']
                                        have_clusters = True
                            except Exception as e:
                                self.logger.warning(f"Ошибка чтения кластеров города из Redis: {e}")

                        # Если кластеров в отдельном ключе ещё нет (старый кэш) — считаем и сохраняем их отдельно
                        if not have_clusters:
                            try:
                                buildings = base_data.get('buildings')
                                road_graph = base_data.get('road_graph')
                                city_boundary = base_data.get('city_boundary')
                                no_fly_zones = base_data.get('no_fly_zones')
                                demand_result = self.get_demand_points_weighted(
                                    buildings,
                                    road_graph,
                                    city_boundary,
                                    method="dbscan",
                                    dbscan_eps_m=180.0,
                                    dbscan_min_samples=15,
                                    return_hulls=True,
                                    use_all_buildings=False,
                                    no_fly_zones=no_fly_zones,
                                )
                                demand_gdf, hulls_gdf = (
                                    demand_result if isinstance(demand_result, tuple) else (demand_result, None)
                                )
                                base_data['demand_points'] = demand_gdf
                                base_data['demand_hulls'] = hulls_gdf
                                if self._redis is not None:
                                    try:
                                        self._redis.set(
                                            redis_clusters_key,
                                            pickle.dumps(
                                                {
                                                    "demand_points": demand_gdf,
                                                    "demand_hulls": hulls_gdf,
                                                }
                                            ),
                                        )
                                    except Exception as e:
                                        self.logger.warning(f"Не удалось сохранить кластеры города в Redis: {e}")
                            except Exception as e:
                                self.logger.warning(f"Не удалось пересчитать кластеры города из кэша: {e}")

                        self._update_progress("cache", 100, "Загрузка из Redis")
                        return self.ensure_flight_levels(base_data)
            except Exception as e:
                self.logger.warning(f"Ошибка чтения из Redis: {e}")

        # Локальный диск как кэш больше не используется: при промахе в Redis загружаем данные заново
        return self._download_city_data(
            normalized_name,
            redis_key=redis_key,
            clusters_redis_key=redis_clusters_key,
            network_type=network_type,
            simplify=simplify,
            load_no_fly_zones=load_no_fly_zones,
        )
    
    def _download_city_data(self, city_name, redis_key: str | None = None, clusters_redis_key: str | None = None, *, network_type: str = 'drive', simplify: bool = True, load_no_fly_zones: bool = True):
        self._update_progress("download", 0, "Начало загрузки данных")
        
        try:
            self._update_progress("download", 15, "Получение границ города для загрузки дорог")
            # Сначала получаем границу города, чтобы грузить дороги по ней (все дороги, пересекающие границу, в т.ч. в отдалённых участках и слегка выходящие за границу)
            boundary_for_roads = None
            try:
                gdf_place = ox.geocode_to_gdf(city_name)
                if len(gdf_place) > 0 and gdf_place.geometry.iloc[0] is not None:
                    boundary_for_roads = gdf_place.geometry.iloc[0]
                    if boundary_for_roads.is_valid and boundary_for_roads.geom_type in ['Polygon', 'MultiPolygon']:
                        self.logger.info(f"Граница для загрузки дорог: {boundary_for_roads.geom_type} (geocode_to_gdf)")
            except Exception as e:
                self.logger.debug(f"geocode_to_gdf для границы: {e}")
            if boundary_for_roads is None:
                try:
                    boundary_gdf = ox.features_from_place(city_name, tags={"boundary": "administrative"})
                    if len(boundary_gdf) > 0:
                        largest_idx = boundary_gdf.geometry.area.idxmax()
                        boundary_for_roads = boundary_gdf.geometry.iloc[largest_idx]
                        if boundary_for_roads.is_valid and boundary_for_roads.geom_type in ['Polygon', 'MultiPolygon']:
                            self.logger.info(f"Граница для загрузки дорог: {boundary_for_roads.geom_type} (administrative)")
                except Exception as e:
                    self.logger.debug(f"features_from_place boundary для границы: {e}")
            
            self._update_progress("download", 20, "Загрузка дорожной сети")
            road_filter = '["highway"~"trunk|primary|secondary|tertiary"]'
            road_graph = None
            # 1) Загрузка по полигону границы с буфером — чтобы дороги, чуть выходящие за границу, не обрезались OSMnx
            # Буфер ~0.002° (~200–250 м) даёт запас: граф грузится по расширенной области, отрисовка границы — по исходной
            if boundary_for_roads is not None:
                try:
                    buffer_deg = 0.002
                    boundary_buffered = boundary_for_roads.buffer(buffer_deg) if boundary_for_roads.is_valid else boundary_for_roads
                    road_graph = ox.graph_from_polygon(boundary_buffered, custom_filter=road_filter, simplify=simplify)
                    if road_graph is not None and len(road_graph.nodes) > 0:
                        self.logger.info(f"Успешно загружена дорожная сеть по границе города (с буфером {buffer_deg}°), узлов: {len(road_graph.nodes)}")
                except Exception as e:
                    self.logger.warning(f"Загрузка дорог по полигону не удалась: {e}, пробуем по названию места")
            
            # 2) Fallback: по названию места
            if road_graph is None or len(road_graph.nodes) == 0:
                query_variants = [
                    city_name,
                    city_name.replace(", ", ","),
                    city_name.split(",")[0].strip() if "," in city_name else None,
                ]
                for q in query_variants:
                    if not q:
                        continue
                    for which in [1, 2, 3]:
                        try:
                            road_graph = ox.graph_from_place(q, custom_filter=road_filter, simplify=simplify, which_result=which)
                            if road_graph is not None and len(road_graph.nodes) > 0:
                                self.logger.info(f"Успешно загружена дорожная сеть для: {q} (which_result={which})")
                                break
                        except Exception as e:
                            self.logger.warning(f"Не удалось загрузить для '{q}' which_result={which}: {e}")
                            continue
                    if road_graph is not None and len(road_graph.nodes) > 0:
                        break
            
            # Fallback 2: геокодируем и загружаем по bbox
            if road_graph is None or len(road_graph.nodes) == 0:
                try:
                    gdf_place = ox.geocode_to_gdf(city_name)
                    if len(gdf_place) > 0 and gdf_place.geometry.iloc[0] is not None:
                        geom = gdf_place.geometry.iloc[0]
                        bbox = geom.bounds
                        if len(bbox) >= 4:
                            north, south = bbox[3], bbox[1]
                            east, west = bbox[2], bbox[0]
                            road_graph = ox.graph_from_bbox(north, south, east, west, custom_filter=road_filter, simplify=simplify)
                            if road_graph is not None and len(road_graph.nodes) > 0:
                                self.logger.info(f"Успешно загружена дорожная сеть по bbox геокодинга: {city_name}")
                except Exception as e:
                    self.logger.warning(f"Fallback geocode+bbox не сработал: {e}")
            
            # Fallback 3: геокодируем точку и загружаем по радиусу (~10 км)
            if road_graph is None or len(road_graph.nodes) == 0:
                try:
                    loc = self._nominatim_geocoder.geocode(city_name)
                    if loc and loc.latitude and loc.longitude:
                        road_graph = ox.graph_from_point((loc.latitude, loc.longitude), dist=10000, custom_filter=road_filter, simplify=simplify)
                        if road_graph is not None and len(road_graph.nodes) > 0:
                            self.logger.info(f"Успешно загружена дорожная сеть по точке+радиусу: {city_name}")
                except Exception as e:
                    self.logger.warning(f"Fallback geocode+point не сработал: {e}")
            
            if road_graph is None or len(road_graph.nodes) == 0:
                raise Exception(f"Не удалось загрузить дорожную сеть для города: {city_name}")
            
            self._update_progress("download", 40, "Загрузка границ города")
            gdf_nodes, _ = ox.graph_to_gdfs(road_graph)
            # Используем границу, по которой грузили дороги (чтобы отрисовка границы совпадала с областью загрузки)
            city_boundary = boundary_for_roads if boundary_for_roads is not None else None
            
            try:
                if city_boundary is None:
                    # Метод 1: Пробуем получить границы через geocode_to_gdf (самый точный)
                    try:
                        gdf_place = ox.geocode_to_gdf(city_name)
                        if len(gdf_place) > 0 and gdf_place.geometry.iloc[0] is not None:
                            city_boundary = gdf_place.geometry.iloc[0]
                            if city_boundary.is_valid and city_boundary.geom_type in ['Polygon', 'MultiPolygon']:
                                self.logger.info(f"✓ Загружены границы города через geocode_to_gdf: {city_boundary.geom_type}")
                            else:
                                city_boundary = None
                                self.logger.warning("Границы из geocode_to_gdf невалидны")
                    except Exception as e:
                        self.logger.warning(f"geocode_to_gdf не сработал: {e}")
                
                # Метод 2: Если не получилось, пробуем через features_from_place
                if city_boundary is None:
                    try:
                        city_boundary_gdf = ox.features_from_place(
                            city_name,
                            tags={"boundary": "administrative"}
                        )
                        if len(city_boundary_gdf) > 0:
                            # Берем самую большую границу (обычно это город)
                            largest_idx = city_boundary_gdf.geometry.area.idxmax()
                            city_boundary = city_boundary_gdf.geometry.iloc[largest_idx]
                            if city_boundary.is_valid and city_boundary.geom_type in ['Polygon', 'MultiPolygon']:
                                self.logger.info(f"✓ Загружены границы города через features_from_place: {city_boundary.geom_type}")
                            else:
                                city_boundary = None
                                self.logger.warning("Границы из features_from_place невалидны")
                    except Exception as e:
                        self.logger.warning(f"features_from_place не сработал: {e}")
                
                # Метод 3: Используем выпуклую оболочку дорожного графа (всегда работает)
                if city_boundary is None:
                    try:
                        # Создаем выпуклую оболочку (convex hull) из узлов дорожного графа
                        # Это дает более точные границы, чем bounding box
                        all_points = gdf_nodes.geometry.tolist()
                        multipoint = MultiPoint(all_points)
                        city_boundary = multipoint.convex_hull
                        self.logger.info(f"✓ Используется выпуклая оболочка дорожного графа: {city_boundary.geom_type}")
                    except Exception as e:
                        self.logger.warning(f"Не удалось создать выпуклую оболочку: {e}")
                        # Последний fallback: bounding box (но это нежелательно)
                        xmin, ymin, xmax, ymax = gdf_nodes.total_bounds
                        city_boundary = box(xmin, ymin, xmax, ymax)
                        self.logger.warning("⚠ Используется bounding box (не рекомендуется)")
                        
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки границ города: {e}")
                # Fallback: используем выпуклую оболочку
                try:
                    all_points = gdf_nodes.geometry.tolist()
                    multipoint = MultiPoint(all_points)
                    city_boundary = multipoint.convex_hull
                    self.logger.info("Используется выпуклая оболочка (fallback)")
                except:
                    xmin, ymin, xmax, ymax = gdf_nodes.total_bounds
                    city_boundary = box(xmin, ymin, xmax, ymax)
                    self.logger.warning("Используется bounding box (последний fallback)")
            
            self._update_progress("download", 50, "Загрузка зданий")
            buildings = gpd.GeoDataFrame()  # Пустой GeoDataFrame по умолчанию
            try:
                # ОПТИМИЗАЦИЯ: ограничиваем загрузку зданий по bbox для ускорения
                if city_boundary is not None:
                    try:
                        b = city_boundary.bounds  # minx, miny, maxx, maxy
                        # OSMnx v2: bbox=(north, south, east, west)
                        buildings = ox.features_from_bbox(
                            bbox=(b[3], b[1], b[2], b[0]),
                            tags={"building": True}
                        )
                        self.logger.info(f"Загружено {len(buildings)} зданий по bbox границ города")
                    except Exception:
                        # Fallback на старый метод
                        buildings = ox.features_from_place(city_name, tags={"building": True})
                else:
                    buildings = ox.features_from_place(city_name, tags={"building": True})
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить здания: {e}")
            
            self._update_progress("download", 80, "Загрузка запретных зон" if load_no_fly_zones else "Пропуск беспилотных зон (только дороги)")
            no_fly_zones = self._get_no_fly_zones(city_name, buildings) if load_no_fly_zones else []

            demand_points = None
            demand_hulls = None
            try:
                demand_result = self.get_demand_points_weighted(
                    buildings,
                    road_graph,
                    city_boundary,
                    method="dbscan",
                    dbscan_eps_m=180.0,
                    dbscan_min_samples=15,
                    return_hulls=True,
                    use_all_buildings=False,
                    no_fly_zones=no_fly_zones,
                )
                demand_gdf, hulls_gdf = (
                    demand_result if isinstance(demand_result, tuple) else (demand_result, None)
                )
                if demand_gdf is not None and len(demand_gdf) > 0:
                    demand_points = demand_gdf
                    demand_hulls = hulls_gdf
            except Exception as e:
                self.logger.warning(f"Не удалось предварительно выделить кластеры спроса: {e}")

            # Базовые данные города (без кластеров)
            data = {
                'road_graph': road_graph,
                'buildings': buildings,
                'no_fly_zones': no_fly_zones,
                'city_boundary': city_boundary,  # Добавляем границы города
                'city_name': city_name,
                'timestamp': time.time(),
                'params': {
                    'network_type': network_type,
                    'simplify': simplify,
                },
                'stats': {
                    'nodes': len(road_graph.nodes),
                    'edges': len(road_graph.edges),
                    'buildings': len(buildings)
                },
            }

            # Полные данные (для возврата вызывающему коду) включают также предвычисленные кластеры
            full_data = dict(data)
            full_data['demand_points'] = demand_points
            full_data['demand_hulls'] = demand_hulls
            
            self._update_progress("download", 85, "Расчёт эшелонов полётов")
            full_data = self.ensure_flight_levels(full_data)
        
            # Кэшируем только в Redis, локальные файлы больше не используем
            self._update_progress("download", 90, "Сохранение в Redis кэш")
            if self._redis is not None and redis_key:
                try:
                    # В ключе города храним только базовые данные (без кластеров)
                    self._redis.set(redis_key, pickle.dumps(data))
                except Exception as e:
                    self.logger.warning(f"Не удалось сохранить данные города в Redis: {e}")
            if self._redis is not None and clusters_redis_key:
                try:
                    clusters_payload = {
                        'demand_points': demand_points,
                        'demand_hulls': demand_hulls,
                    }
                    self._redis.set(clusters_redis_key, pickle.dumps(clusters_payload))
                except Exception as e:
                    self.logger.warning(f"Не удалось сохранить кластеры города в Redis: {e}")
            
            self._update_progress("download", 100, "Данные загружены")
            return full_data
            
        except Exception as e:
            self._update_progress("error", 0, f"Ошибка: {str(e)}")
            self.logger.error(f"Ошибка загрузки данных для {city_name}: {e}")
            raise
    
    def _get_no_fly_zones(self, city_name, buildings: Optional[gpd.GeoDataFrame] = None):
        """
        Загружает беспилотные зоны из OpenStreetMap.
        Только чувствительные объекты: аэропорты, военные, парки, школы, детсады, универы,
        колледжи, техникумы, поликлиники, больницы и т.п.
        Обычные жилые дома в no_fly не попадают.
        
        Признаки:
        1. Аэропорты и аэродромы
        2. Военные объекты
        3. Атомные и правительственные объекты
        4. Электростанции (ГЭС, ТЭЦ, АЭС, power=plant/station, man_made=hydroelectric_plant)
        5. Явные запретные зоны (restriction:drone=no)
        6. Парки и зоны отдыха (leisure=park, landuse=recreation_ground)
        7. Школы, детсады, университеты, колледжи, техникумы (amenity + building из OSM)
        8. Больницы (по границам из OSM)
        9. Поликлиники и клиники (amenity=clinic, healthcare=clinic)
        10. Заправки (amenity=fuel)
        11. Вокзалы (railway=station)
        
        Returns:
            GeoDataFrame или список беспилотных зон
        """
        no_fly_zones = []
        # Радиус буфера для точек (когда в OSM нет контура и не найден nearby building)
        BUFFER_RADIUS_DEGREES = 0.0002  # ~39 метров (урезано с ~55 м)
        # Радиус поиска здания рядом с точкой (~55 м)
        POINT_SEARCH_BUFFER_DEG = 0.0005
        
        def zone_from_geometry(geom, use_building_perimeter: bool = True):
            """
            Полигоны берём по границам из OSM. Для точек — ищем здание поблизости
            и используем его периметр; если не найдено — круг ~39 м.
            """
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                return geom
            if geom.geom_type != "Point":
                return geom.buffer(BUFFER_RADIUS_DEGREES)
            # Точка: пытаемся найти здание по периметру
            if use_building_perimeter and buildings is not None and len(buildings) > 0:
                try:
                    if buildings.crs is None:
                        buildings_4326 = buildings.set_crs("EPSG:4326", allow_override=True)
                    else:
                        buildings_4326 = buildings.to_crs("EPSG:4326")
                    search_area = geom.buffer(POINT_SEARCH_BUFFER_DEG)
                    # Здания, пересекающие область поиска
                    mask = buildings_4326.geometry.intersects(search_area)
                    candidates = buildings_4326[mask]
                    if len(candidates) > 0:
                        # Берём здание, центр которого ближе всего к точке (или самое большое в зоне)
                        best = None
                        best_dist = float("inf")
                        for _, row in candidates.iterrows():
                            g = row.geometry
                            if g is None or not getattr(g, "is_valid", True):
                                continue
                            if g.geom_type in ("Polygon", "MultiPolygon"):
                                cent = g.centroid
                                d = geom.distance(cent)
                                if d < best_dist:
                                    best_dist = d
                                    best = g
                        if best is not None:
                            return best
                except Exception as e:
                    self.logger.debug(f"Поиск здания для точки: {e}")
            return geom.buffer(BUFFER_RADIUS_DEGREES)
        
        try:
            self.logger.info(f"Загрузка беспилотных зон для {city_name}")
            
            # 1. Аэропорты и аэродромы (самое важное!)
            try:
                airports = ox.features_from_place(
                    city_name,
                    tags={
                        "aeroway": ["aerodrome", "airport", "heliport", "helipad"],
                    }
                )
                if len(airports) > 0:
                    self.logger.info(f"Найдено {len(airports)} аэропортов/аэродромов")
                    for idx, airport in airports.iterrows():
                        if airport.geometry is not None and airport.geometry.is_valid:
                            # Добавляем буфер 200м вокруг аэропорта (ограничение до 200м от границы)
                            buffer_zone = airport.geometry.buffer(BUFFER_RADIUS_DEGREES)
                            no_fly_zones.append(buffer_zone)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки аэропортов: {e}")
            
            # 2. Военные объекты
            try:
                military = ox.features_from_place(
                    city_name,
                    tags={
                        "military": True,
                        "landuse": "military",
                    }
                )
                if len(military) > 0:
                    self.logger.info(f"Найдено {len(military)} военных объектов")
                    for idx, obj in military.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            # Добавляем буфер 200м вокруг военного объекта
                            buffer_zone = obj.geometry.buffer(BUFFER_RADIUS_DEGREES)
                            no_fly_zones.append(buffer_zone)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки военных объектов: {e}")
            
            # 3. Атомные станции и опасные объекты
            try:
                nuclear = ox.features_from_place(
                    city_name,
                    tags={
                        "power": "nuclear",
                        "nuclear": True,
                    }
                )
                if len(nuclear) > 0:
                    self.logger.info(f"Найдено {len(nuclear)} атомных объектов")
                    for idx, obj in nuclear.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            # Добавляем буфер 200м вокруг атомного объекта
                            buffer_zone = obj.geometry.buffer(BUFFER_RADIUS_DEGREES)
                            no_fly_zones.append(buffer_zone)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки атомных объектов: {e}")
            
            # 4. Электростанции: ГЭС, ТЭЦ, АЭС и подобные (power=plant, power=station, гидро)
            for tag_dict, label in [
                ({"power": "plant"}, "power=plant (ТЭЦ, ГЭС и др.)"),
                ({"power": "station"}, "power=station"),
                ({"man_made": "hydroelectric_plant"}, "man_made=hydroelectric_plant (ГЭС)"),
            ]:
                try:
                    power_objs = ox.features_from_place(city_name, tags=tag_dict)
                    if len(power_objs) > 0:
                        self.logger.info(f"Найдено {len(power_objs)} объектов: {label}")
                        for idx, obj in power_objs.iterrows():
                            if obj.geometry is not None and obj.geometry.is_valid:
                                no_fly_zones.append(zone_from_geometry(obj.geometry))
                except Exception as e:
                    if "No data elements" not in str(e):
                        self.logger.warning(f"Ошибка загрузки {label}: {e}")
            
            # 5. Правительственные объекты (опционально, меньший буфер)
            try:
                government = ox.features_from_place(
                    city_name,
                    tags={
                        "office": "government",
                        "government": True,
                    }
                )
                if len(government) > 0:
                    self.logger.info(f"Найдено {len(government)} правительственных объектов")
                    for idx, obj in government.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            # Добавляем буфер 200м вокруг правительственного объекта
                            buffer_zone = obj.geometry.buffer(BUFFER_RADIUS_DEGREES)
                            no_fly_zones.append(buffer_zone)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки правительственных объектов: {e}")
            
            # 6. Явные запретные зоны из OSM (если есть теги)
            try:
                restricted = ox.features_from_place(
                    city_name,
                    tags={
                        "restriction:drone": "no",
                        "drone": "no",
                    }
                )
                if len(restricted) > 0:
                    self.logger.info(f"Найдено {len(restricted)} явных запретных зон")
                    for idx, zone in restricted.iterrows():
                        if zone.geometry is not None and zone.geometry.is_valid:
                            # Добавляем буфер 200м вокруг явной запретной зоны
                            buffer_zone = zone.geometry.buffer(BUFFER_RADIUS_DEGREES)
                            no_fly_zones.append(buffer_zone)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки запретных зон: {e}")
            
            # 7. Парки и зоны отдыха
            try:
                parks = ox.features_from_place(
                    city_name,
                    tags={"leisure": "park"}
                )
                if len(parks) > 0:
                    self.logger.info(f"Найдено {len(parks)} парков (leisure=park)")
                    for idx, obj in parks.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            no_fly_zones.append(zone_from_geometry(obj.geometry))
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки парков: {e}")
            try:
                recreation = ox.features_from_place(
                    city_name,
                    tags={"landuse": "recreation_ground"}
                )
                if len(recreation) > 0:
                    self.logger.info(f"Найдено {len(recreation)} зон отдыха (landuse=recreation_ground)")
                    for idx, obj in recreation.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            no_fly_zones.append(zone_from_geometry(obj.geometry))
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки зон отдыха: {e}")
            
            # 8. Школы, детсады, университеты, колледжи, техникумы
            try:
                education = ox.features_from_place(
                    city_name,
                    tags={"amenity": ["school", "kindergarten", "university", "college"]}
                )
                if len(education) > 0:
                    self.logger.info(f"Найдено {len(education)} объектов образования (школы, детсады, универы, колледжи)")
                    for idx, obj in education.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            no_fly_zones.append(zone_from_geometry(obj.geometry))
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки объектов образования: {e}")
            # 8b. Здания колледжей/школ по тегу building (техникумы, училища и т.д.)
            try:
                edu_buildings = ox.features_from_place(
                    city_name,
                    tags={"building": ["college", "school"]}
                )
                if len(edu_buildings) > 0:
                    self.logger.info(f"Найдено {len(edu_buildings)} зданий образования (building=college/school)")
                    for idx, obj in edu_buildings.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            no_fly_zones.append(zone_from_geometry(obj.geometry))
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки зданий образования: {e}")
            
            # 9. Больницы
            try:
                hospitals = ox.features_from_place(
                    city_name,
                    tags={"amenity": "hospital"}
                )
                if len(hospitals) > 0:
                    self.logger.info(f"Найдено {len(hospitals)} больниц")
                    for idx, obj in hospitals.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            no_fly_zones.append(zone_from_geometry(obj.geometry))
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки больниц: {e}")
            
            # 9b. Поликлиники и клиники (включая детские)
            for tag_dict, label in [
                ({"amenity": "clinic"}, "amenity=clinic"),
                ({"healthcare": "clinic"}, "healthcare=clinic"),
            ]:
                try:
                    clinics = ox.features_from_place(city_name, tags=tag_dict)
                    if len(clinics) > 0:
                        self.logger.info(f"Найдено {len(clinics)} поликлиник/клиник ({label})")
                        for idx, obj in clinics.iterrows():
                            if obj.geometry is not None and obj.geometry.is_valid:
                                no_fly_zones.append(zone_from_geometry(obj.geometry))
                except Exception as e:
                    if "No data elements" not in str(e):
                        self.logger.warning(f"Ошибка загрузки {label}: {e}")
            
            # 10. Заправки (АЗС)
            try:
                fuel = ox.features_from_place(
                    city_name,
                    tags={"amenity": "fuel"}
                )
                if len(fuel) > 0:
                    self.logger.info(f"Найдено {len(fuel)} заправок (amenity=fuel)")
                    for idx, obj in fuel.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            no_fly_zones.append(zone_from_geometry(obj.geometry))
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки заправок: {e}")
            
            # 11. Вокзалы (железнодорожные станции)
            try:
                stations = ox.features_from_place(
                    city_name,
                    tags={"railway": "station"}
                )
                if len(stations) > 0:
                    self.logger.info(f"Найдено {len(stations)} вокзалов/станций (railway=station)")
                    for idx, obj in stations.iterrows():
                        if obj.geometry is not None and obj.geometry.is_valid:
                            no_fly_zones.append(zone_from_geometry(obj.geometry))
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки вокзалов: {e}")
            
            if len(no_fly_zones) > 0:
                # Объединяем все зоны в один GeoDataFrame
                from shapely.geometry import Polygon, MultiPolygon
                
                try:
                    # Объединяем все зоны
                    union_zones = unary_union(no_fly_zones)
                    
                    # Создаем GeoDataFrame
                    if isinstance(union_zones, Polygon):
                        zones_gdf = gpd.GeoDataFrame([{'geometry': union_zones}], crs='EPSG:4326')
                    elif isinstance(union_zones, MultiPolygon):
                        zones_gdf = gpd.GeoDataFrame(
                            [{'geometry': geom} for geom in union_zones.geoms],
                            crs='EPSG:4326'
                        )
                    else:
                        zones_gdf = gpd.GeoDataFrame([{'geometry': union_zones}], crs='EPSG:4326')
                    
                    self.logger.info(f"Создано {len(zones_gdf)} беспилотных зон")
                    return zones_gdf
                except Exception as e:
                    self.logger.warning(f"Ошибка объединения беспилотных зон: {e}")
                    # Возвращаем как список геометрий
                    return no_fly_zones
            
        except Exception as e:
            self.logger.warning(f"Ошибка загрузки беспилотных зон: {e}")
        
        return []
    
    # --- Кластеризация спроса по зданиям ---
    DEMAND_BUILDING_TAGS = frozenset((
        'house', 'residential', 'apartments', 'apartment', 'apartment_block', 'multistory', 'block', 'flats',
        'semidetached_house', 'dormitory', 'detached', 'terrace', 'hut', 'cabin', 'bungalow',
        'retail', 'office', 'commercial', 'supermarket',
        'yes', 'true', '1',  # в OSM жилые часто без уточнения
    ))
    EXCLUDE_FROM_DEMAND_TAGS = frozenset((
        'industrial', 'warehouse', 'factory', 'manufacture', 'garage', 'shed', 'garages',
        'school', 'university', 'college', 'hospital', 'kindergarten', 'civic', 'government',
        'train_station', 'service',
    ))
    # Вес спроса: многоквартирники дают повышенный спрос (больше людей)
    APARTMENT_DEMAND_WEIGHT = 4

    def _filter_buildings_for_demand(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Оставляет только здания, релевантные для спроса: жилые и часть коммерции (retail/office).
        Пром/склады исключены.
        """
        if buildings is None or len(buildings) == 0:
            return buildings
        def keep(row):
            tag = row.get("building") if hasattr(row, "get") else None
            if tag is None or (isinstance(tag, float) and pd.isna(tag)) or str(tag).strip() == "":
                tag = "yes"
            tag = str(tag).lower().strip()
            if tag in self.EXCLUDE_FROM_DEMAND_TAGS:
                return False
            return tag in self.DEMAND_BUILDING_TAGS
        mask = buildings.apply(keep, axis=1)
        out = buildings.loc[mask].copy()
        if len(out) > 0 and len(out) < len(buildings):
            self.logger.info(f"Спрос доставки: оставлено {len(out)} зданий из {len(buildings)} (жилые + retail/office, без пром/складов)")
        return out

    def _building_footprint_area_m2(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Добавляет колонку area_m2 — площадь footprint здания в м² (в UTM)."""
        if buildings is None or len(buildings) == 0:
            return buildings
        try:
            crs_utm = buildings.estimate_utm_crs()
            b_utm = buildings.to_crs(crs_utm)
            areas = [float(g.area) if g is not None and g.is_valid else 0.0 for g in b_utm.geometry]
            out = buildings.copy()
            out["area_m2"] = areas
            return out
        except Exception as e:
            self.logger.warning(f"Расчёт площади зданий: {e}")
            out = buildings.copy()
            out["area_m2"] = 0.0
            return out

    def get_demand_points_weighted(
        self,
        buildings: gpd.GeoDataFrame,
        road_graph,
        city_boundary,
        *,
        method: str = "grid",
        cell_size_m: float = 250.0,
        dbscan_eps_m: float = 180.0,
        dbscan_min_samples: int = 15,
        return_hulls: bool = False,
        use_all_buildings: bool = False,
        no_fly_zones: Optional[Union[gpd.GeoDataFrame, List]] = None,
        min_buildings_for_zone: int = 2,
        # При method="dbscan": дополнительно заполнять полигоны кластеров сеткой точек спроса,
        # чтобы размещение станций стремилось покрыть ВСЮ область кластера, а не только центроид.
        fill_clusters: bool = False,
        # Шаг сетки внутри кластера (в метрах). По умолчанию = dbscan_eps_m.
        cluster_fill_step_m: Optional[float] = None,
        # Радиус кругов при построении полигона кластера (hull): eps_m * hull_radius_factor.
        # Чем меньше коэффициент, тем ближе граница к зданиям (меньше заходит на соседние зоны).
        hull_radius_factor: float = 0.3,
    ):
        """
        Шаг B: точки спроса для размещения — только по зданиям (центроиды), дороги не используются.
        При use_all_buildings=False — только жилые и часть коммерции; при True — все здания
        (кроме попавших в беспилотные зоны, если заданы no_fly_zones).
        method: 'grid' — ячейки 200–300 м с весом, 'dbscan' — кластеры DBSCAN.
        dbscan_eps_m: 120–180 м — кварталы, 200–300 м — микрорайоны (при большом min_samples).
        dbscan_min_samples: 10–25 для городской застройки (15 — компромисс).
        return_hulls: при method=dbscan можно вернуть (gdf, hulls_gdf) — области кластеров.
        use_all_buildings: если True — по всем зданиям, иначе только жилые/коммерция.
        no_fly_zones: при заданных зонах из спроса исключаются здания, пересекающие эти зоны.
        min_buildings_for_zone: кластер/ячейка с числом зданий меньше этого — не считается зоной спроса (одиночки = выбросы).
        """
        if buildings is None or len(buildings) == 0:
            return gpd.GeoDataFrame()
        buildings = buildings.to_crs("EPSG:4326")
        if not use_all_buildings:
            buildings = self._filter_buildings_for_demand(buildings)
        if buildings is None or len(buildings) == 0:
            return (gpd.GeoDataFrame(), gpd.GeoDataFrame()) if return_hulls else gpd.GeoDataFrame()
        no_fly_union = None
        # Исключаем здания, попадающие в беспилотные зоны
        if no_fly_zones is not None:
            try:
                if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                    no_fly_zones = no_fly_zones.to_crs("EPSG:4326")
                    no_fly_union = no_fly_zones.geometry.unary_union
                elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                    no_fly_union = unary_union(no_fly_zones)
                if no_fly_union is not None and not no_fly_union.is_empty:
                    mask = ~buildings.geometry.intersects(no_fly_union)
                    buildings = buildings[mask].copy()
                    self.logger.info(f"После исключения зданий в беспилотных зонах: {len(buildings)} зданий")
                    if len(buildings) == 0:
                        return (gpd.GeoDataFrame(), gpd.GeoDataFrame()) if return_hulls else gpd.GeoDataFrame()
            except Exception as e:
                self.logger.warning(f"Ошибка фильтрации по беспилотным зонам: {e}")
                no_fly_union = None
        # Только здания внутри границ города
        if city_boundary is not None:
            try:
                boundary_geom = city_boundary if hasattr(city_boundary, "geom_type") else None
                if boundary_geom is not None and boundary_geom.is_valid and boundary_geom.geom_type in ("Polygon", "MultiPolygon"):
                    crs_utm = buildings.estimate_utm_crs()
                    buildings_proj = buildings.to_crs(crs_utm)
                    boundary_proj = gpd.GeoSeries([boundary_geom], crs="EPSG:4326").to_crs(crs_utm).iloc[0]
                    mask = buildings_proj.geometry.centroid.within(boundary_proj)
                    n_before = len(buildings)
                    buildings = buildings[mask].copy()
                    if len(buildings) < n_before:
                        self.logger.info(f"Здания только внутри границы города: {len(buildings)} из {n_before}")
                    if len(buildings) == 0:
                        return (gpd.GeoDataFrame(), gpd.GeoDataFrame()) if return_hulls else gpd.GeoDataFrame()
            except Exception as e:
                self.logger.warning(f"Фильтрация зданий по границе города: {e}")
        # Площадь footprint для весов (МКД: подъезды; коммерция: area × levels).
        buildings = self._building_footprint_area_m2(buildings)
        # Зоны спроса — только по домам (центроиды зданий). Дороги не используются.
        # Дополнительно помечаем тип здания и вес спроса по типу.
        def _get_building_levels(row) -> Optional[int]:
            """Число надземных этажей из building:levels или из height (примерно). None если неизвестно."""
            for key in ("building:levels", "levels"):
                val = row.get(key) if hasattr(row, "get") else None
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                try:
                    s = str(val).strip().split()[0]  # "3" или "12 m" -> "12"
                    n = int(float(s.replace(",", ".")))
                    if 0 < n <= 200:
                        return n
                except (ValueError, TypeError):
                    pass
            val = row.get("height") if hasattr(row, "get") else None
            if val is not None and str(val).strip():
                try:
                    s = str(val).strip().lower().replace("m", "").replace("м", "").strip().split()[0]
                    h = float(s.replace(",", "."))
                    if 2 < h < 500:
                        return max(1, int(round(h / 3.0)))
                except (ValueError, TypeError):
                    pass
            return None

        def _apartments_per_floor_mkd(levels: int) -> int:
            """Квартир на этаж по этажности МКД (надземные этажи)."""
            if levels <= 5:
                return 3
            if levels <= 9:
                return 4
            if levels <= 16:
                return 5
            return 6

        def _entrances_count_mkd(row, area_m2: float) -> int:
            """Число подъездов: тег OSM или max(1, round(area_m² / 300))."""
            for key in ("building:entrances", "entrances"):
                val = row.get(key) if hasattr(row, "get") else None
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                try:
                    s = str(val).strip().split()[0]
                    n = int(float(s.replace(",", ".")))
                    if n > 0:
                        return n
                except (ValueError, TypeError):
                    continue
            if area_m2 > 0:
                return max(1, int(round(area_m2 / 300.0)))
            return 1

        def _building_demand_type(row) -> str:
            """
            Тип спроса по зданию (максимально используя доступные теги и этажность):
            - apartment: многоквартирные дома (приоритетные по спросу)
            - office: офисы, коммерция
            - retail: крупная розница / ТЦ
            - industrial: промзона, склады, заводы
            - public: школы, больницы, госучреждения и пр.
            - private: частный сектор (индивидуальные дома)
            - other: всё остальное.
            """
            levels = _get_building_levels(row)

            # 1) МКД — явные теги или residential/yes при этажности >= 3
            if self._is_apartment(row):
                return "apartment"
            btag_raw = row.get("building") if hasattr(row, "get") else None
            btag = str(btag_raw).lower().strip() if btag_raw is not None and not (isinstance(btag_raw, float) and pd.isna(btag_raw)) else ""
            if btag in ("residential", "yes", "true", "1") and levels is not None and levels >= 3:
                return "apartment"

            amenity_raw = row.get("amenity") if hasattr(row, "get") else None
            amenity = str(amenity_raw).lower().strip() if amenity_raw is not None and not (isinstance(amenity_raw, float) and pd.isna(amenity_raw)) else ""
            shop_raw = row.get("shop") if hasattr(row, "get") else None
            shop = str(shop_raw).lower().strip() if shop_raw is not None and not (isinstance(shop_raw, float) and pd.isna(shop_raw)) else ""

            # 2) Общественные объекты (школы, больницы и т.п.)
            if amenity in self._PUBLIC_AMENITY_VALUES:
                return "public"
            if btag in ("school", "university", "college", "kindergarten", "hospital", "clinic", "civic", "government", "public"):
                return "public"

            # 3) Коммерция / офисы / розница
            if btag in ("office", "commercial"):
                return "office"
            if btag in ("retail", "supermarket", "mall", "shop", "kiosk", "market"):
                return "retail"
            if shop:
                return "retail"

            # 4) Промзона / индустриальные объекты
            if btag in ("industrial", "warehouse", "factory", "manufacture", "plant", "depot", "hangar", "storage"):
                return "industrial"

            # 5) Отели / общежития
            if btag in ("hotel", "hostel", "dormitory", "motel"):
                return "public"

            # 6) Частный сектор: явные теги дома или yes/1 при этажности <= 2; residential только при этажности <= 2
            if self._is_private_house(row, is_apartment=False):
                return "private"
            if btag in ("yes", "true", "1") and (levels is None or levels <= 2) and not amenity and not shop:
                return "private"
            if btag == "residential" and levels is not None and levels <= 2:
                return "private"

            # 7) Без тега building или пустой тег — при отсутствии amenity/shop считаем частный сектор
            if (not btag or (isinstance(btag_raw, float) and pd.isna(btag_raw))) and not amenity and not shop:
                return "private"

            # 8) Остальное (residential без этажности и т.п. — не приписываем ни к МКД, ни к частному)
            return "other"

        def _building_demand_weight(row) -> float:
            """
            Вес спроса по типу здания:
            - apartment (МКД): кв_на_этаж(levels) * levels * entrances;
              entrances из тега или max(1, round(area_m² / 300)).
            - private: 1 этаж — 1.2; 2+ этажей — 1.5.
            - retail / office / industrial / public: area_m² * levels * 0.01.
            - other: 1.
            """
            t = _building_demand_type(row)
            area_m2 = float(row.get("area_m2", 0) or 0)
            levels = _get_building_levels(row)
            if t == "apartment":
                lv = max(1, levels if levels is not None else 5)
                apf = _apartments_per_floor_mkd(lv)
                ent = _entrances_count_mkd(row, area_m2)
                return float(apf * lv * ent)
            if t == "private":
                if levels is None:
                    return 1.2
                lv = max(1, levels)
                return 1.2 if lv <= 1 else 1.5
            if t in ("retail", "office", "industrial", "public"):
                lv = max(1, levels if levels is not None else 1)
                w = area_m2 * lv * 0.01
                return float(w) if w > 0 else 1.0
            return 1.0

        pts_list = []
        weights_list = []
        types_list = []
        for idx, b in buildings.iterrows():
            g = b.geometry
            if g is None or not getattr(g, "is_valid", True):
                continue
            c = g.centroid
            pts_list.append([c.x, c.y])
            # Вес спроса по типу здания
            w = _building_demand_weight(b)
            weights_list.append(w)
            types_list.append(_building_demand_type(b))
        base_points = np.array(pts_list) if pts_list else np.empty((0, 2))
        base_weights = np.array(weights_list, dtype=float) if weights_list else np.ones(len(base_points))
        base_types = np.array(types_list, dtype=object) if types_list else np.array([], dtype=object)
        if len(base_points) > 0:
            self.logger.info(
                f"Точки для кластеризации: только по зданиям ({len(base_points)} центроидов), "
                f"типы спроса: {set(base_types.tolist()) if base_types.size > 0 else set()}"
            )
        if len(base_points) < 2:
            return gpd.GeoDataFrame()
        use_utm = True
        crs_utm = None
        try:
            crs_utm = gpd.GeoSeries([Point(base_points[0][0], base_points[0][1])], crs="EPSG:4326").estimate_utm_crs()
            from pyproj import Transformer
            trans = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
            pts_utm = np.array([trans.transform(x, y) for x, y in base_points])
        except Exception:
            use_utm = False
            pts_utm = base_points
            crs_utm = None
            if method == "grid":
                cell_size_m = 0.0025  # ~250 m в градусах
        rows = []
        hull_rows = [] if (method == "dbscan" and return_hulls) else None
        if method == "dbscan" and use_utm and crs_utm is not None:
            try:
                from sklearn.cluster import DBSCAN
                from pyproj import Transformer
                eps_m = dbscan_eps_m
                inv = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
                # Радиус круга вокруг центроида здания при сборке hull (тот же порядок, что hull_radius_m ниже).
                hull_buf_m = max(30.0, float(eps_m) * float(hull_radius_factor))
                # Шаг сетки внутри кластера (в метрах) для fill_clusters.
                # По умолчанию делаем сетку существенно реже eps, чтобы не взрывать число точек.
                if cluster_fill_step_m is not None:
                    fill_step_m = float(cluster_fill_step_m)
                else:
                    # Минимум 400 м, но не меньше, чем 2 * eps.
                    fill_step_m = max(400.0, float(eps_m) * 2.0)
                if fill_step_m <= 0:
                    fill_step_m = max(400.0, float(eps_m) * 2.0)
                from shapely.ops import transform as sh_transform

                from sklearn.cluster import KMeans, MiniBatchKMeans
                MAX_WEIGHT_PER_HULL = 10000.0
                # За один вызов KMeans не дробим на тысячи кластеров — только порциями, иначе 80k+ точек «висит» часами.
                MAX_K_PER_SPLIT = 48
                # Доп. порог по разбросу центроидов зданий (без буфера hull): только при очень вытянутых группах.
                MAX_RADIUS_PER_HULL_M = max(500.0, float(eps_m) * 4.0)
                REGION_EPS_M = max(600.0, float(eps_m) * 6.0)
                REGION_MIN_SAMPLES = max(6, int(dbscan_min_samples) * 2)

                def _cluster_radius_m(sub_pts: np.ndarray) -> float:
                    """Макс. расстояние от среднего центроида зданий до самой дальней точки (м)."""
                    if len(sub_pts) <= 1:
                        return 0.0
                    c = np.mean(sub_pts, axis=0)
                    d2 = np.sum((sub_pts - c) ** 2, axis=1)
                    return float(np.sqrt(float(np.max(d2)))) if len(d2) > 0 else 0.0

                def _spatial_kmeans_labels(sub_pts: np.ndarray, k: int) -> np.ndarray:
                    """KMeans / MiniBatchKMeans по координатам в метрах; устойчиво на десятках тысяч точек."""
                    n = len(sub_pts)
                    k = int(max(1, min(k, n)))
                    if k <= 1 or n <= 1:
                        return np.zeros(n, dtype=int)
                    if n > 3500 or k > 24:
                        bs = min(4096, max(1024, n // 5))
                        mb = MiniBatchKMeans(
                            n_clusters=k,
                            random_state=42,
                            batch_size=bs,
                            n_init=3,
                            max_iter=200,
                            reassignment_ratio=0.02,
                        )
                        return np.asarray(mb.fit(sub_pts).labels_, dtype=int)
                    n_init = 5 if n < 8000 else 3
                    km = KMeans(n_clusters=k, random_state=42, n_init=n_init, max_iter=300)
                    return np.asarray(km.fit(sub_pts).labels_, dtype=int)

                def _partition_by_target_weight(indices):
                    """
                    Рекурсивно режем набор зданий так, чтобы итоговые кластеры:
                    - по весу не превышали 10000 (сумма весов зданий);
                    - по возможности были ~ ceil(W/10000) штук на район (KMeans в метрах).
                    """
                    if len(indices) == 0:
                        return []
                    sub_pts = pts_utm[indices]
                    sub_w = base_weights[indices]
                    total_w = float(sub_w.sum())
                    radius_m = _cluster_radius_m(sub_pts)
                    n_pts = len(indices)
                    need_w = total_w > MAX_WEIGHT_PER_HULL
                    need_r = radius_m > MAX_RADIUS_PER_HULL_M
                    if (not need_w and not need_r) or n_pts <= 1:
                        return [indices]
                    k_w = int(np.ceil(total_w / MAX_WEIGHT_PER_HULL))
                    k_w = max(1, k_w)
                    k = k_w
                    if need_r:
                        k = max(k, 2)
                    if need_w:
                        k = max(k, 2)
                    k = min(k, n_pts, MAX_K_PER_SPLIT)
                    if k < 2 and n_pts > 1 and (need_w or need_r):
                        k = 2
                    labels = _spatial_kmeans_labels(sub_pts, k)
                    out = []
                    for j in sorted(set(np.asarray(labels, dtype=int).tolist())):
                        m = labels == j
                        sub_idx = indices[m]
                        if len(sub_idx) > 0:
                            out.extend(_partition_by_target_weight(sub_idx))
                    return out

                def _merge_neighboring_weight_groups(groups):
                    """Сливаем ближайшие по центроиду группы, если суммарный вес ≤ 10000 (меньше кластеров)."""
                    if len(groups) <= 1:
                        return groups
                    groups = [np.asarray(g, dtype=int) for g in groups if len(g) > 0]
                    try:
                        from scipy.spatial import cKDTree
                    except Exception:
                        cKDTree = None
                    max_merge_iters = min(5000, max(200, len(groups) * 3))
                    it = 0
                    while len(groups) > 1 and it < max_merge_iters:
                        it += 1
                        weights = np.array([float(base_weights[gi].sum()) for gi in groups], dtype=float)
                        cents = np.stack([pts_utm[gi].mean(axis=0) for gi in groups], axis=0)
                        best_i, best_j = None, None
                        best_d = None
                        if cKDTree is not None and len(groups) > 80:
                            tree = cKDTree(cents)
                            kn = min(12, len(groups))
                            dists, nbrs = tree.query(cents, k=kn)
                            for i in range(len(groups)):
                                for jj in range(kn):
                                    j = int(nbrs[i, jj])
                                    if j == i or j < i:
                                        continue
                                    if weights[i] + weights[j] > MAX_WEIGHT_PER_HULL + 1e-9:
                                        continue
                                    d = float(dists[i, jj]) ** 2 if dists.ndim == 2 else float("inf")
                                    if best_d is None or d < best_d:
                                        best_d = d
                                        best_i, best_j = i, j
                        else:
                            for i in range(len(groups)):
                                for j in range(i + 1, len(groups)):
                                    if weights[i] + weights[j] > MAX_WEIGHT_PER_HULL + 1e-9:
                                        continue
                                    d = float(np.sum((cents[i] - cents[j]) ** 2))
                                    if best_d is None or d < best_d:
                                        best_d = d
                                        best_i, best_j = i, j
                        if best_i is None:
                            break
                        merged = np.concatenate([groups[best_i], groups[best_j]])
                        groups = [g for t, g in enumerate(groups) if t not in (best_i, best_j)]
                        groups.append(merged)
                    return groups

                def _build_one_region(indices, region_id, subcluster_id, do_fill):
                    if len(indices) < 1:
                        return
                    sub_pts = pts_utm[indices]
                    sub_weights = base_weights[indices]
                    sub_w = float(sub_weights.sum())
                    n_b = len(sub_pts)
                    cx, cy = sub_pts.mean(axis=0)
                    lon_c, lat_c = inv.transform(cx, cy)
                    rows.append({
                        "geometry": Point(lon_c, lat_c),
                        "weight": min(MAX_WEIGHT_PER_HULL, int(round(sub_w))),
                        "n_buildings": int(n_b),
                        "region_id": int(region_id),
                        "subcluster_id": int(subcluster_id),
                        "cluster_id": int(subcluster_id),
                        "is_cluster_fill": False,
                    })
                    try:
                        hull_radius_m = float(hull_buf_m)
                        circles_utm = [Point(float(x), float(y)).buffer(hull_radius_m) for x, y in sub_pts]
                        region_utm = unary_union(circles_utm)
                        if region_utm is None or region_utm.is_empty:
                            return
                        region_utm = region_utm.simplify(2.0)
                        region_4326 = sh_transform(lambda x, y: inv.transform(x, y), region_utm)
                        if (city_boundary is not None and hasattr(city_boundary, "is_valid")
                                and getattr(city_boundary, "is_valid", True)
                                and region_4326.is_valid and not region_4326.is_empty):
                            try:
                                clipped = region_4326.intersection(city_boundary)
                                if clipped is not None and not clipped.is_empty and clipped.is_valid:
                                    region_4326 = clipped
                            except Exception:
                                pass
                        if (no_fly_union is not None and not no_fly_union.is_empty
                                and region_4326.is_valid and not region_4326.is_empty):
                            try:
                                region_4326 = region_4326.difference(no_fly_union)
                                if region_4326.is_empty:
                                    return
                            except Exception:
                                pass
                        if region_4326.is_valid and not region_4326.is_empty:
                            if return_hulls and hull_rows is not None:
                                hull_rows.append({
                                    "geometry": region_4326,
                                    "weight": min(MAX_WEIGHT_PER_HULL, int(round(sub_w))),
                                    "n_buildings": int(n_b),
                                    "region_id": int(region_id),
                                    "subcluster_id": int(subcluster_id),
                                    "cluster_id": int(subcluster_id),
                                })
                            if do_fill and fill_clusters:
                                try:
                                    minx, miny, maxx, maxy = region_4326.bounds
                                    if maxx > minx and maxy > miny:
                                        lat_center = (miny + maxy) / 2.0
                                        step_km = fill_step_m / 1000.0 or eps_m / 1000.0
                                        deg_lat = step_km / 111.0
                                        deg_lon = step_km / (111.0 * max(0.2, np.cos(np.radians(lat_center))))
                                        step_x, step_y = max(deg_lon, 1e-5), max(deg_lat, 1e-5)
                                        fill_count, max_fill_points = 0, 500
                                        x = float(minx)
                                        while x <= maxx:
                                            y = float(miny)
                                            while y <= maxy:
                                                if region_4326.contains(Point(x, y)):
                                                    rows.append({"geometry": Point(x, y), "weight": 1, "n_buildings": 0, "region_id": int(region_id), "subcluster_id": int(subcluster_id), "cluster_id": int(subcluster_id), "is_cluster_fill": True})
                                                    fill_count += 1
                                                    if fill_count >= max_fill_points:
                                                        break
                                                y += step_y
                                            x += step_x
                                            if fill_count >= max_fill_points:
                                                break
                                except Exception as e_fill:
                                    self.logger.debug(f"fill_clusters: {e_fill}")
                    except Exception as e:
                        self.logger.debug(f"Область кластера: {e}")

                # Уровень 1: крупные «районные» кластеры только по пространственной близости.
                self.logger.info(
                    f"DBSCAN районов: {len(pts_utm)} зданий, eps={REGION_EPS_M:.0f} м, min_samples={REGION_MIN_SAMPLES}…"
                )
                _dbscan_kw = {"eps": REGION_EPS_M, "min_samples": REGION_MIN_SAMPLES, "metric": "euclidean"}
                _dbscan_kw["n_jobs"] = -1
                try:
                    region_labels = np.array(DBSCAN(**_dbscan_kw).fit(pts_utm).labels_, dtype=int)
                except TypeError:
                    _dbscan_kw.pop("n_jobs", None)
                    region_labels = np.array(DBSCAN(**_dbscan_kw).fit(pts_utm).labels_, dtype=int)
                self.logger.info("DBSCAN районов завершён, разбиение по весу 10000…")
                region_ids = sorted(set(region_labels) - {-1})
                if not region_ids:
                    # Если город разрежен — считаем весь набор одним районом и далее режем вторым уровнем.
                    region_labels = np.zeros(len(pts_utm), dtype=int)
                    region_ids = [0]

                # Шум первого уровня привязываем к ближайшему району; если районов нет — отдельно как шум.
                if np.any(region_labels == -1) and region_ids:
                    region_centers = {}
                    for rid in region_ids:
                        m = region_labels == rid
                        region_centers[rid] = np.array(pts_utm[m].mean(axis=0))
                    for i_noise in np.where(region_labels == -1)[0]:
                        pt = pts_utm[i_noise]
                        best_rid = min(region_ids, key=lambda rid: float(np.sum((pt - region_centers[rid]) ** 2)))
                        region_labels[i_noise] = best_rid

                subcluster_seq = 0
                for rid in sorted(set(region_labels) - {-1}):
                    rid_mask = region_labels == rid
                    region_indices = np.where(rid_mask)[0]
                    if len(region_indices) < 1:
                        continue

                    # Уровень 2: без DBSCAN — иначе получается слишком много мелких кластеров.
                    # Целимся в ~10000 веса на подкластер: k ≈ ceil(W/10000), KMeans в метрах + рекурсия + слияние соседей.
                    groups = _partition_by_target_weight(region_indices)
                    groups = _merge_neighboring_weight_groups(groups)
                    for g_idx in groups:
                        if len(g_idx) < 1:
                            continue
                        subcluster_seq += 1
                        _build_one_region(
                            g_idx,
                            region_id=int(rid),
                            subcluster_id=subcluster_seq,
                            do_fill=(len(groups) == 1),
                        )
            except Exception:
                method = "grid"
                hull_rows = None
        elif method == "dbscan":
            method = "grid"
        if method == "grid":
            cell = cell_size_m
            cell_centers = {}
            cell_counts = {}
            for i in range(len(pts_utm)):
                x, y = pts_utm[i, 0], pts_utm[i, 1]
                w = base_weights[i] if i < len(base_weights) else 1.0
                gx = int(x // cell) * cell + cell / 2
                gy = int(y // cell) * cell + cell / 2
                key = (gx, gy)
                cell_centers[key] = cell_centers.get(key, 0) + w
                cell_counts[key] = cell_counts.get(key, 0) + 1
            trans_inv = None
            if use_utm:
                try:
                    from pyproj import Transformer
                    trans_inv = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
                except Exception:
                    pass
            _grid_cluster_seq = 0
            for (gx, gy), weight in cell_centers.items():
                if cell_counts.get((gx, gy), 0) < min_buildings_for_zone:
                    continue  # одиночки в ячейке — выбросы, зону спроса не создаём
                if weight == 0:
                    continue
                if trans_inv:
                    lon, lat = trans_inv.transform(gx, gy)
                else:
                    lon, lat = float(gx), float(gy)
                rows.append({
                    "geometry": Point(lon, lat),
                    "weight": weight,
                    "n_buildings": int(cell_counts.get((gx, gy), 0)),
                    "cluster_id": int(_grid_cluster_seq),
                })
                _grid_cluster_seq += 1
        if not rows:
            if return_hulls:
                return (gpd.GeoDataFrame(), gpd.GeoDataFrame())
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        gdf["weight"] = gdf.get("weight", 1)
        self.logger.info(f"Точки спроса ({method}): {len(gdf)}, сумма весов {gdf['weight'].sum()}")

        if return_hulls:
            # Устраняем пересечения глобально для всех кластеров,
            # чтобы полигоны кластеров не накладывались друг на друга.
            # Без упрощения геометрии повторяющиеся difference + нарастающий unary_union
            # при сотнях полигонов могут «висеть» на минуты и дольше (GEOS).
            if hull_rows:
                try:
                    from shapely.ops import unary_union as shp_union

                    hulls_gdf = gpd.GeoDataFrame(hull_rows, crs="EPSG:4326")
                    # Сортируем все кластеры по весу (от более тяжёлых к более лёгким)
                    sub = hulls_gdf.sort_values(by="weight", ascending=False).reset_index(drop=True)
                    n_hulls = len(sub)
                    # ~5–8 м в градусах (середина широты РФ); сильно уменьшает число вершин у buffer-union кругов.
                    _hull_tol_deg = 6e-5

                    def _prep_hull_geom(g):
                        if g is None or g.is_empty:
                            return g
                        try:
                            g = g.simplify(_hull_tol_deg, preserve_topology=True)
                            if g is None or g.is_empty:
                                return g
                            return g.buffer(0)
                        except Exception:
                            return g

                    cleaned_parts = []
                    used_geom = None
                    if n_hulls > 0:
                        self.logger.info(
                            f"Устранение пересечений hull-кластеров: {n_hulls} полигонов (упрощение ~{_hull_tol_deg:.0e}°)…"
                        )
                    for i, (idx, row) in enumerate(sub.iterrows()):
                        if n_hulls > 400 and i > 0 and i % 250 == 0:
                            self.logger.info(f"Пересечения hulls: обработано {i}/{n_hulls}")
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue
                        geom = _prep_hull_geom(geom)
                        if geom is None or geom.is_empty:
                            continue
                        if used_geom is not None and not used_geom.is_empty:
                            try:
                                ug = _prep_hull_geom(used_geom)
                                if ug is not None and not ug.is_empty:
                                    geom = geom.difference(ug)
                            except Exception:
                                pass
                        if geom is None or geom.is_empty:
                            continue
                        cleaned_parts.append({
                            "geometry": geom,
                            "weight": row["weight"],
                            "n_buildings": row.get("n_buildings"),
                            "cluster_id": row.get("cluster_id"),
                        })
                        try:
                            used_geom = geom if used_geom is None else shp_union([used_geom, geom])
                            if used_geom is not None and not used_geom.is_empty:
                                used_geom = _prep_hull_geom(used_geom)
                        except Exception:
                            pass
                    hulls_gdf = gpd.GeoDataFrame(cleaned_parts, crs="EPSG:4326")
                    hulls_gdf = hulls_gdf[~hulls_gdf["geometry"].isna() & ~hulls_gdf["geometry"].is_empty].copy()
                except Exception as e:
                    self.logger.warning(f"Не удалось устранить пересечения кластеров: {e}")
                    hulls_gdf = gpd.GeoDataFrame(hull_rows, crs="EPSG:4326")
            else:
                hulls_gdf = gpd.GeoDataFrame()
            # Вес областей спроса не должен превышать 10000
            if not hulls_gdf.empty and "weight" in hulls_gdf.columns:
                hulls_gdf["weight"] = hulls_gdf["weight"].clip(upper=10000).astype(int)
            return (gdf, hulls_gdf)

        return gdf

    @staticmethod
    def _is_apartment(row) -> bool:
        """Многоквартирный дом по OSM-тегу building."""
        tag = row.get('building') if hasattr(row, 'get') else None
        if tag is None or (isinstance(tag, float) and pd.isna(tag)):
            return False
        tag = str(tag).lower().strip()
        return tag in ('apartments', 'apartment', 'apartment_block', 'multistory', 'block', 'flats', 'semidetached_house')

    # Теги OSM building, типичные для частного сектора (в т.ч. в РФ часто yes или без тега)
    _PRIVATE_BUILDING_TAGS = frozenset((
        'house', 'detached', 'hut', 'cabin', 'bungalow', 'terrace',
        'residential',  # в РФ часто помечают частные дома
        'yes', 'true', '1',  # в OSM частные дома часто без уточнения
        'garage', 'shed', 'garages',  # хозпостройки в частном секторе
    ))

    # Публичные типы зданий (amenity/shop): не считать частным сектором
    _PUBLIC_AMENITY_VALUES = frozenset((
        'school', 'university', 'college', 'kindergarten', 'hospital', 'clinic', 'police',
        'townhall', 'community_centre', 'library', 'sports_centre', 'fire_station',
    ))

    # Теги building, при которых квартал считаем «не частным». Без 'residential' — в РФ им часто помечают частные дома.
    _NON_PRIVATE_BLOCK_BUILDING_TAGS = frozenset((
        'apartments', 'apartment', 'apartment_block', 'multistory', 'block', 'flats', 'semidetached_house',
        'commercial', 'office', 'retail', 'industrial', 'school', 'university', 'college', 'hospital', 'kindergarten',
        'civic', 'government', 'public', 'train_station', 'supermarket', 'hotel', 'dormitory', 'service',
    ))

    @classmethod
    def _building_makes_block_non_private(cls, row) -> bool:
        """Здание «делает квартал не частным»: явно многоквартирное (не residential), общественное, коммерческое."""
        for key in ('amenity', 'shop'):
            val = row.get(key) if hasattr(row, 'get') else None
            if val is not None and str(val).strip():
                if str(val).lower().strip() in cls._PUBLIC_AMENITY_VALUES:
                    return True
        tag = row.get('building') if hasattr(row, 'get') else None
        if tag is not None and str(tag).strip() and not (isinstance(tag, float) and pd.isna(tag)):
            if str(tag).lower().strip() in cls._NON_PRIVATE_BLOCK_BUILDING_TAGS:
                return True
        return False

    @classmethod
    def _is_private_house(cls, row, is_apartment: bool = False) -> bool:
        """Частный дом только по явному тегу building (внутри квартала решение по кварталу)."""
        if is_apartment:
            return False
        for key in ('amenity', 'shop'):
            val = row.get(key) if hasattr(row, 'get') else None
            if val is not None and str(val).strip():
                if str(val).lower().strip() in cls._PUBLIC_AMENITY_VALUES:
                    return False
        tag = row.get('building') if hasattr(row, 'get') else None
        if tag is None or (isinstance(tag, float) and pd.isna(tag)) or str(tag).strip() == '':
            return False  # без тега не считаем частным — тип определит квартал
        tag = str(tag).lower().strip()
        return tag in cls._PRIVATE_BUILDING_TAGS
    
    def _sanitize_name(self, name):
        import re
        name = re.sub(r'[^\w\s-]', '', name)
        return re.sub(r'[-\s]+', '_', name).strip('_')[:100]

    # --- Высоты зданий и эшелоны полётов ---
    METERS_PER_FLOOR = 3.0  # типичная высота этажа
    DEFAULT_BUILDING_HEIGHT = 10.0  # м, если нет данных (типичный 3-этажный дом)
    MAX_DRONE_ALTITUDE = 150.0  # м, ограничение максимальной высоты подъёма дрона
    MIN_FLIGHT_LEVEL = 40.0  # м, минимальный эшелон (выше типичной застройки)
    FLIGHT_LEVEL_MARGIN = 10.0  # м, запас над высочайшими зданиями
    NUM_FLIGHT_LEVELS = 5  # количество эшелонов (5-й на MAX_DRONE_ALTITUDE)

    @staticmethod
    def _parse_osm_height(value) -> float | None:
        """Парсит тег height из OSM: '15m', '15', '50 ft', '15.5' и т.п."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        s = str(value).strip().lower()
        if not s:
            return None
        # Убираем пробелы, извлекаем число
        import re
        match = re.match(r'^([\d.]+)\s*(m|м|meters?|метров?|ft|feet)?$', s)
        if match:
            num = float(match.group(1))
            unit = (match.group(2) or 'm').lower()
            if unit in ('ft', 'feet'):
                return num * 0.3048
            return num
        return None

    def _compute_building_heights(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Добавляет колонку height_m к зданиям на основе OSM-тегов.
        Приоритет: height -> building:levels * 3м -> DEFAULT_BUILDING_HEIGHT.
        """
        if buildings is None or len(buildings) == 0:
            return buildings
        heights = []
        for idx, row in buildings.iterrows():
            h = None
            # height: тег OSM или колонка
            for key in ('height', 'height_m'):
                if key in buildings.columns:
                    h = self._parse_osm_height(row.get(key))
                    if h is not None:
                        break
            # building:levels или building_levels
            if h is None:
                for key in ('building:levels', 'building_levels', 'levels'):
                    if key in buildings.columns:
                        try:
                            levels = row.get(key)
                            if levels is not None and not (isinstance(levels, float) and pd.isna(levels)):
                                levels = int(float(str(levels).split('.')[0]))
                                if levels > 0:
                                    h = levels * self.METERS_PER_FLOOR
                                    break
                        except (ValueError, TypeError):
                            pass
            if h is None:
                h = self.DEFAULT_BUILDING_HEIGHT
            heights.append(max(3.0, min(h, 500.0)))  # clamp 3–500 м
        buildings = buildings.copy()
        buildings['height_m'] = heights
        return buildings

    def _compute_flight_levels(
        self,
        buildings: gpd.GeoDataFrame,
        num_levels: int | None = None,
        max_altitude: float | None = None,
        min_altitude: float | None = None,
    ) -> list[dict]:
        """
        Вычисляет эшелоны полётов на основе высот зданий.

        Args:
            buildings: GeoDataFrame с колонкой height_m
            num_levels: количество эшелонов (по умолчанию NUM_FLIGHT_LEVELS, 5)
            max_altitude: максимальная высота (м), по умолчанию MAX_DRONE_ALTITUDE (150)
            min_altitude: минимальная высота первого эшелона (м)

        Returns:
            Список dict: [{"level": 1, "altitude_m": 45, "label": "Эшелон 1"}, ...]
        """
        num_levels = num_levels if num_levels is not None else self.NUM_FLIGHT_LEVELS
        max_alt = max_altitude if max_altitude is not None else self.MAX_DRONE_ALTITUDE
        min_alt = min_altitude

        if buildings is not None and len(buildings) > 0 and 'height_m' in buildings.columns:
            p95 = float(np.percentile(buildings['height_m'], 95))
            base_alt = max(
                p95 + self.FLIGHT_LEVEL_MARGIN,
                self.MIN_FLIGHT_LEVEL,
            )
            if min_alt is not None:
                base_alt = max(base_alt, min_alt)
            self.logger.info(
                f"Высоты зданий: p95={p95:.1f}м, базовый эшелон={base_alt:.1f}м"
            )
        else:
            base_alt = self.MIN_FLIGHT_LEVEL
            if min_alt is not None:
                base_alt = max(base_alt, min_alt)

        # Равномерно распределяем эшелоны от base_alt до max_alt
        step = (max_alt - base_alt) / max(1, num_levels - 1) if num_levels > 1 else 0
        levels = []
        for i in range(num_levels):
            alt = round(base_alt + i * step, 1)
            levels.append({
                "level": i + 1,
                "altitude_m": alt,
                "label": f"Эшелон {i + 1} ({alt:.0f} м)",
            })
        self.logger.info(f"Эшелоны полётов: {[l['altitude_m'] for l in levels]}")
        return levels

    def ensure_flight_levels(self, data: dict, num_levels: int | None = None) -> dict:
        """
        Добавляет в data поля buildings (с height_m), building_height_stats, flight_levels.
        Вызывать после загрузки/кэша для совместимости со старым кэшем.
        По умолчанию 5 эшелонов, верхний на 150 м.
        """
        buildings = data.get('buildings')
        if buildings is None or len(buildings) == 0:
            data['flight_levels'] = []
            data['building_height_stats'] = {}
            return data

        if 'height_m' not in buildings.columns:
            buildings = self._compute_building_heights(buildings)
            data['buildings'] = buildings

        stats = {}
        if 'height_m' in buildings.columns:
            h = buildings['height_m']
            stats = {
                "min_m": float(h.min()),
                "max_m": float(h.max()),
                "mean_m": float(h.mean()),
                "median_m": float(h.median()),
                "p95_m": float(np.percentile(h, 95)),
                "count": len(buildings),
            }
        data['building_height_stats'] = stats
        data['flight_levels'] = self._compute_flight_levels(
            buildings,
            num_levels=num_levels if num_levels is not None else self.NUM_FLIGHT_LEVELS,
            max_altitude=self.MAX_DRONE_ALTITUDE,
        )
        return data

    # --- Кандидаты для размещения станций (зарядка / гараж / ТО) ---
    _ROOFTOP_BUILDING_TAGS = frozenset(
        (
            "apartments",
            "apartment",
            "apartment_block",
            "multistory",
            "block",
            "flats",
            "semidetached_house",
        )
    )
    _GROUND_BUILDING_TAGS = frozenset(
        (
            "retail",
            "supermarket",
            "commercial",
            "mall",
            "shop",
            "kiosk",
            "warehouse",
            "parking",
            "garage",
            "garages",
        )
    )
    _INDUSTRIAL_BUILDING_TAGS = frozenset(
        (
            "industrial",
            "warehouse",
            "factory",
            "manufacture",
            "depot",
            "hangar",
            "commercial",
            "garages",
            "shed",
        )
    )

    def get_station_candidates(
        self,
        buildings: gpd.GeoDataFrame,
        city_boundary,
        no_fly_zones,
        road_graph,
        station_type: str = "rooftop",
    ) -> gpd.GeoDataFrame:
        """
        Точки-кандидаты для размещения станций (центроиды зданий WGS84).
        rooftop — крыши МКД/жилые многоэтажки (source=building);
        ground — наземные площадки / коммерция (source=ground);
        garage / to — промзоны и склады (для гаража и ТО).
        """
        from shapely.geometry import Point

        if buildings is None or len(buildings) == 0:
            return gpd.GeoDataFrame()
        b = buildings.to_crs("EPSG:4326").copy()
        no_fly_union = None
        if no_fly_zones is not None:
            try:
                if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                    try:
                        no_fly_union = no_fly_zones.geometry.union_all()
                    except Exception:
                        no_fly_union = no_fly_zones.geometry.unary_union
                elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                    no_fly_union = unary_union(no_fly_zones)
                if no_fly_union is not None and (getattr(no_fly_union, "is_empty", True) or not getattr(no_fly_union, "is_valid", True)):
                    no_fly_union = None
            except Exception:
                no_fly_union = None
        if no_fly_union is not None:
            try:
                b = b[~b.geometry.intersects(no_fly_union)].copy()
            except Exception:
                pass
        if city_boundary is not None and getattr(city_boundary, "is_valid", True):
            try:
                crs_utm = b.estimate_utm_crs()
                bp = b.to_crs(crs_utm)
                cb = gpd.GeoSeries([city_boundary], crs="EPSG:4326").to_crs(crs_utm).iloc[0]
                mask = bp.geometry.centroid.within(cb)
                b = b[mask].copy()
            except Exception:
                pass

        if "height_m" not in b.columns:
            b = self._compute_building_heights(b)
        try:
            crs_utm = b.estimate_utm_crs()
            b["area_m2"] = b.to_crs(crs_utm).geometry.area.astype(float)
        except Exception:
            b["area_m2"] = 0.0

        rows = []
        for idx, row in b.iterrows():
            g = row.geometry
            if g is None or not getattr(g, "is_valid", True) or g.is_empty:
                continue
            c = g.centroid
            lon, lat = float(c.x), float(c.y)
            tag_raw = row.get("building")
            tag = str(tag_raw).lower().strip() if tag_raw is not None and not (isinstance(tag_raw, float) and pd.isna(tag_raw)) else ""
            area = float(row.get("area_m2", 0) or 0)
            levels = None
            for key in ("building:levels", "levels"):
                v = row.get(key)
                if v is not None and str(v).strip():
                    try:
                        levels = int(float(str(v).split()[0].replace(",", ".")))
                        break
                    except (ValueError, TypeError):
                        pass
            is_mkd = self._is_apartment(row) or tag in self._ROOFTOP_BUILDING_TAGS
            is_residential_high = tag in ("residential", "yes", "house") and levels is not None and levels >= 3
            is_rooftop = is_mkd or is_residential_high

            if station_type == "rooftop":
                if not is_rooftop and not (tag in ("yes", "residential") and area > 800):
                    continue
                rows.append({"geometry": Point(lon, lat), "source": "building", "osm_index": idx})
            elif station_type == "ground":
                if is_rooftop:
                    continue
                if tag in self._GROUND_BUILDING_TAGS or tag in ("office",) or area > 400:
                    rows.append({"geometry": Point(lon, lat), "source": "ground", "osm_index": idx})
            elif station_type in ("garage", "to"):
                if tag not in self._INDUSTRIAL_BUILDING_TAGS and area < 300:
                    continue
                rows.append(
                    {
                        "geometry": Point(lon, lat),
                        "source": "industrial",
                        "osm_index": idx,
                    }
                )
            else:
                continue

        if not rows:
            return gpd.GeoDataFrame()
        return gpd.GeoDataFrame(rows, crs="EPSG:4326")

    def get_power_substation_buffer_union(self, buildings: gpd.GeoDataFrame):
        """Объединённая геометрия подстанций (power=substation) с буфером ~50 м — зоны, где не ставим гараж/ТО."""
        if buildings is None or len(buildings) == 0:
            return None
        try:
            import osmnx as ox

            b = buildings.to_crs("EPSG:4326").total_bounds
            north, south, east, west = b[3], b[1], b[2], b[0]
            subs = ox.features_from_bbox(bbox=(north, south, east, west), tags={"power": "substation"})
            if subs is None or len(subs) == 0:
                return None
            geoms = []
            buf_deg = 0.00045
            for g in subs.geometry:
                if g is None or not getattr(g, "is_valid", True) or g.is_empty:
                    continue
                try:
                    geoms.append(g.buffer(buf_deg))
                except Exception:
                    pass
            if not geoms:
                return None
            return unary_union(geoms)
        except Exception as e:
            self.logger.debug("Подстанции OSM: %s", e)
            return None

    def get_redis_client(self):
        """Возвращает Redis клиент для использования в других сервисах"""
        return self._redis