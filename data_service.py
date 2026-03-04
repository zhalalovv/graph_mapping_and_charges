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
from geopy.extra.rate_limiter import RateLimiter
import logging
import json
from redis import Redis
from shapely.geometry import box, MultiPoint, Point, LineString, Polygon
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree
from shapely.prepared import prep

# Подавляем DeprecationWarning из OSMnx (unary_union и др. — внутренние вызовы библиотеки)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="osmnx")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"osmnx\.features")


class DataService:
    def __init__(self, cache_dir="cache_data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.progress_callbacks = []
        # Используем несколько геокодеров для лучшей поддержки российских адресов
        self.geolocators = {
            'nominatim': Nominatim(user_agent="drone_route_planner", timeout=10),
            'nominatim_ru': Nominatim(user_agent="drone_route_planner", timeout=10, domain='nominatim.openstreetmap.org')
        }
        # Rate-limited reverse geocoder
        self._reverse = RateLimiter(self.geolocators['nominatim'].reverse, min_delay_seconds=1)
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # optional Redis for caching heavy city data
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            self._redis: Redis | None = Redis.from_url(redis_url)
            # do not decode responses here; we store bytes for pickle blobs
            self._redis.ping()
        except Exception:
            self._redis = None
            self.logger.info("Redis not available for DataService cache; using disk cache")
    
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
        cache_file = os.path.join(self.cache_dir, f"{key_suffix}.pkl")
        redis_key = f"drone_planner:city:{key_suffix}"
        
        # Try Redis first
        if self._redis is not None:
            try:
                blob = self._redis.get(redis_key)
                if blob:
                    cached_data = pickle.loads(blob)
                    # Проверяем, есть ли границы города в кэше
                    if 'city_boundary' not in cached_data:
                        self.logger.info("В Redis кэше нет границ города, перезагружаем данные")
                        # Удаляем из Redis и загружаем заново
                        self._redis.delete(redis_key)
                    else:
                        self._update_progress("cache", 100, "Загрузка из Redis")
                        return self.ensure_flight_levels(cached_data)
            except Exception as e:
                self.logger.warning(f"Ошибка чтения из Redis: {e}")

        if os.path.exists(cache_file):
            self._update_progress("cache", 100, "Загрузка из кэша")
            try:
                cached_data = self._load_from_cache(cache_file)
                # Проверяем, есть ли границы города в кэше (для совместимости со старым кэшем)
                if 'city_boundary' not in cached_data:
                    self.logger.info("В кэше нет границ города, перезагружаем данные")
                    os.remove(cache_file)
                    return self._download_city_data(normalized_name, cache_file, redis_key, network_type=network_type, simplify=simplify, load_no_fly_zones=load_no_fly_zones)
                return self.ensure_flight_levels(cached_data)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки кэша: {e}, перезагружаем данные")
                os.remove(cache_file)
        
        return self._download_city_data(normalized_name, cache_file, redis_key, network_type=network_type, simplify=simplify, load_no_fly_zones=load_no_fly_zones)
    
    def _download_city_data(self, city_name, cache_file, redis_key: str | None = None, *, network_type: str = 'drive', simplify: bool = True, load_no_fly_zones: bool = True):
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
                    loc = self.geolocators['nominatim'].geocode(city_name)
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
                }
            }
            
            self._update_progress("download", 85, "Расчёт эшелонов полётов")
            data = self.ensure_flight_levels(data)

            self._update_progress("download", 90, "Сохранение в кэш")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            # Save to Redis as well
            if self._redis is not None and redis_key:
                try:
                    self._redis.set(redis_key, pickle.dumps(data))
                except Exception as e:
                    self.logger.warning(f"Не удалось сохранить данные города в Redis: {e}")
            
            self._update_progress("download", 100, "Данные загружены")
            return data
            
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
    
    # --- Кандидаты размещения станций (Шаг A) и точки спроса (Шаг B) ---
    # Теги OSM, по которым отстраиваются кандидаты станций:
    #
    # Зарядки (крыша): здания с building=* из ALLOWED_ROOF_BUILDING_TAGS (см. ниже),
    #   площадь footprint ≥ ROOF_MIN_AREA_M2; здания грузятся по tags={"building": True}.
    # Зарядки (земля), гаражи, ТО: промзоны и парковки из _load_industrial_and_parking_areas:
    #   landuse=industrial | warehouse; building=industrial | warehouse | factory | manufacture;
    #   amenity=parking; parking=True.
    #   Гаражи и ТО — только здания в промзонах (industrial), без парковок.
    #
    NO_FLY_BUFFER_DEGREES_PER_M = 0.000009  # ~1 м в градусах на широте ~55
    ROOF_MIN_AREA_M2 = 80.0  # минимальная площадь крыши для крышной зарядки
    GROUND_MIN_AREA_M2 = 100.0  # минимальная площадка для наземной зарядки
    # Крышные зарядки: только МКД (многоквартирные дома), без коммерции/офисов
    # Используем тот же набор тегов, что и в _is_apartment.
    ALLOWED_ROOF_BUILDING_TAGS = frozenset((
        'apartments', 'apartment', 'apartment_block', 'multistory', 'block', 'flats',
        'semidetached_house',
    ))
    # Допустимые типы зданий для гаражей/ТО внутри промзон
    ALLOWED_INDUSTRIAL_BUILDING_TAGS = frozenset((
        'industrial', 'warehouse', 'factory', 'manufacture', 'garages', 'garage',
        'service', 'commercial',
    ))
    # Ключевые слова тяжёлой промышленности/опасных объектов, которых нужно избегать
    FORBIDDEN_INDUSTRIAL_KEYWORDS = (
        'пив', 'brew', 'beer',             # пивкомбинаты / breweries
        'завод', 'комбинат',              # крупные заводы/комбинаты
        'судоремонт', 'судостро', 'верф', 'ship', 'dock', 'shipyard',
        'химволокн', 'химическ', 'цемент', 'совхоз',  # заводы, химзаводы, совхозы
    )
    # Спрос доставки: только жилые + часть коммерции; пром/склады — только для кандидатов станций, не для спроса
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
    # Объекты энергетики, которые нельзя использовать для размещения станций (подстанции и т.п.)
    EXCLUDE_POWER_TAGS = frozenset((
        'substation', 'sub_station', 'plant', 'generator', 'transformer',
    ))
    # Буфер вокруг подстанций (м), чтобы гаражи/ТО не ставились рядом
    POWER_SUBSTATION_BUFFER_M = 30.0
    # Вес спроса: многоквартирники дают повышенный спрос (больше людей)
    APARTMENT_DEMAND_WEIGHT = 4

    def _filter_buildings_for_demand(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Оставляет только здания, релевантные для спроса доставки: жилые (house/residential/apartments/dormitory)
        и часть коммерции (retail/office). Пром/склады исключены — они только для кандидатов станций.
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

    def _industrial_polygon_is_forbidden(self, row) -> bool:
        """True, если у объекта OSM (промзона/здание) в тегах есть ключевые слова заводов/опасных объектов."""
        text_fields = []
        for key in ("name", "industrial", "craft", "operator", "brand"):
            val = row.get(key) if hasattr(row, "get") else None
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            text_fields.append(str(val).lower())
        combined = " ".join(text_fields)
        if not combined:
            return False
        return any(k in combined for k in self.FORBIDDEN_INDUSTRIAL_KEYWORDS)

    def _load_power_substation_geometries(self, bbox_osm, crs_4326):
        """
        Загружает из OSM геометрии подстанций и других энергообъектов (power=substation, plant, etc.).
        bbox_osm: (north, south, east, west). Returns: list of Shapely geometry (4326).
        """
        north, south, east, west = bbox_osm
        geoms = []
        try:
            for power_val in self.EXCLUDE_POWER_TAGS:
                gdf = ox.features_from_bbox(bbox=(north, south, east, west), tags={"power": power_val})
                if gdf is None or len(gdf) == 0:
                    continue
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                gdf = gdf.to_crs(crs_4326)
                for _, row in gdf.iterrows():
                    g = row.geometry
                    if g is None or not getattr(g, "is_valid", True):
                        continue
                    if g.geom_type == "Point":
                        geoms.append(g)
                    elif g.geom_type == "Polygon":
                        geoms.append(g)
                    elif g.geom_type == "MultiPolygon":
                        for poly in g.geoms:
                            if poly.is_valid and not poly.is_empty:
                                geoms.append(poly)
        except Exception as e:
            self.logger.warning(f"Загрузка подстанций OSM: {e}")
        return geoms

    def get_power_substation_buffer_union(self, buildings):
        """
        Возвращает объединённую геометрию буферов (POWER_SUBSTATION_BUFFER_M м) вокруг всех
        подстанций/энергообъектов: из зданий с power=* и из OSM (power=substation и др.).
        Используется для жёсткого исключения кандидатов гаража/ТО из этой зоны.
        Returns: Shapely geometry (union) или None.
        """
        if buildings is None or len(buildings) == 0:
            return None
        try:
            buildings = buildings.to_crs("EPSG:4326")
            power_geoms = []
            buffer_deg = self.POWER_SUBSTATION_BUFFER_M * self.NO_FLY_BUFFER_DEGREES_PER_M
            for _, row in buildings.iterrows():
                geom = row.geometry
                if geom is None or not getattr(geom, "is_valid", True):
                    continue
                if str(row.get("power") or "").lower().strip() in self.EXCLUDE_POWER_TAGS:
                    try:
                        power_geoms.append(geom.buffer(buffer_deg))
                    except Exception:
                        continue
            bbox_osm = (buildings.total_bounds[3], buildings.total_bounds[1], buildings.total_bounds[2], buildings.total_bounds[0])
            for g in self._load_power_substation_geometries(bbox_osm, "EPSG:4326"):
                try:
                    power_geoms.append(g.buffer(buffer_deg))
                except Exception:
                    continue
            if not power_geoms:
                return None
            return unary_union(power_geoms)
        except Exception as e:
            self.logger.warning(f"get_power_substation_buffer_union: {e}")
            return None

    def _load_industrial_and_parking_areas(self, bbox_osm, crs_4326):
        """
        Загружает полигоны промзон и парковок/площадок из OSM по bbox.
        bbox_osm: (north, south, east, west).
        Returns: dict with 'industrial': list of Shapely polygons (4326), 'parking': list of (centroid Point or polygon),
                 'industrial_forbidden': list of polygons территории заводов/опасных объектов (не ставить ТО/гараж).
        """
        north, south, east, west = bbox_osm
        industrial = []
        industrial_forbidden = []
        parking = []
        try:
            # Промзоны: landuse=industrial, warehouse; здания industrial/warehouse
            for tags in (
                {"landuse": ["industrial", "warehouse"]},
                {"building": ["industrial", "warehouse", "factory", "manufacture"]},
            ):
                gdf = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
                if gdf is not None and len(gdf) > 0:
                    if gdf.crs is None:
                        gdf = gdf.set_crs("EPSG:4326")
                    gdf = gdf.to_crs(crs_4326)
                    for _, row in gdf.iterrows():
                        g = row.geometry
                        if g is None or not getattr(g, "is_valid", True):
                            continue
                        # Отбрасываем энергообъекты: подстанции и пр. (power=*)
                        power_tag = str(row.get("power") or "").lower().strip()
                        if power_tag in self.EXCLUDE_POWER_TAGS:
                            continue
                        is_forbidden = self._industrial_polygon_is_forbidden(row)
                        if g.geom_type == "Polygon":
                            if is_forbidden:
                                industrial_forbidden.append(g)
                            else:
                                industrial.append(g)
                        elif g.geom_type == "MultiPolygon":
                            for poly in g.geoms:
                                if poly.is_valid and not poly.is_empty:
                                    if is_forbidden:
                                        industrial_forbidden.append(poly)
                                    else:
                                        industrial.append(poly)
        except Exception as e:
            self.logger.warning(f"Загрузка промзон OSM: {e}")
        try:
            # Парковки и площадки: amenity=parking, landuse=commercial (часто с парковками)
            for tags in (
                {"amenity": "parking"},
                {"parking": True},
            ):
                gdf = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
                if gdf is not None and len(gdf) > 0:
                    if gdf.crs is None:
                        gdf = gdf.set_crs("EPSG:4326")
                    gdf = gdf.to_crs(crs_4326)
                    for _, row in gdf.iterrows():
                        g = row.geometry
                        if g is None or not getattr(g, "is_valid", True):
                            continue
                        if g.geom_type == "Point":
                            parking.append(g)
                        else:
                            try:
                                c = g.centroid
                                if not c.is_empty:
                                    parking.append(c)
                            except Exception:
                                pass
        except Exception as e:
            self.logger.warning(f"Загрузка парковок OSM: {e}")
        return {"industrial": industrial, "parking": parking, "industrial_forbidden": industrial_forbidden}

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

    def _point_outside_no_fly(self, pt, no_fly_zones, buffer_deg: float) -> bool:
        """True, если точка вне всех no-fly зон с заданным буфером."""
        if no_fly_zones is None:
            return True
        if isinstance(no_fly_zones, gpd.GeoDataFrame) and no_fly_zones.empty:
            return True
        if isinstance(no_fly_zones, list) and len(no_fly_zones) == 0:
            return True
        try:
            geoms = []
            if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                geoms = list(no_fly_zones.geometry)
            elif isinstance(no_fly_zones, list):
                geoms = [z.geometry if hasattr(z, "geometry") else z for z in no_fly_zones]
            for g in geoms:
                if g is None:
                    continue
                buf = g.buffer(buffer_deg) if buffer_deg != 0 else g
                if buf.contains(pt) or buf.intersects(pt):
                    return False
            return True
        except Exception:
            return True

    def get_station_candidates(
        self,
        buildings: gpd.GeoDataFrame,
        city_boundary,
        no_fly_zones,
        road_graph,
        *,
        station_type: str = "rooftop",
        no_fly_buffer_m: float = 50.0,
        min_roof_area_m2: float = None,
        min_ground_area_m2: float = None,
    ) -> gpd.GeoDataFrame:
        """
        Шаг A: формирует кандидатов размещения станций.
        station_type: 'rooftop' | 'ground' | 'garage' | 'to'
        Кандидаты проходят: вне no-fly + буфер, минимальная площадка/доступность.
        """
        if buildings is None or len(buildings) == 0:
            return gpd.GeoDataFrame()
        buffer_deg = no_fly_buffer_m * self.NO_FLY_BUFFER_DEGREES_PER_M
        min_roof = min_roof_area_m2 if min_roof_area_m2 is not None else self.ROOF_MIN_AREA_M2
        min_ground = min_ground_area_m2 if min_ground_area_m2 is not None else self.GROUND_MIN_AREA_M2

        buildings = buildings.to_crs("EPSG:4326")
        if city_boundary is not None:
            try:
                crs_utm = buildings.estimate_utm_crs()
                buildings_proj = buildings.to_crs(crs_utm)
                boundary_proj = gpd.GeoSeries([city_boundary], crs="EPSG:4326").to_crs(crs_utm).iloc[0]
                mask = buildings_proj.geometry.centroid.within(boundary_proj)
                buildings = buildings[mask]
            except Exception as e:
                self.logger.warning(f"Фильтр зданий по границе: {e}")
        if len(buildings) == 0:
            return gpd.GeoDataFrame()

        # Объединённая геометрия зданий, чтобы не ставить станции прямо "на крыше" действующих объектов
        buildings_union = None
        try:
            geoms = [g for g in buildings.geometry if g is not None and getattr(g, "is_valid", True)]
            if geoms:
                buildings_union = unary_union(geoms)
        except Exception:
            buildings_union = None

        # Буфер вокруг подстанций/энергообъектов (power=*), чтобы не ставить гаражи/ТО ближе заданного расстояния.
        # Учитываем и здания с power=*, и отдельные объекты power=* из OSM (подстанции часто не в слое зданий).
        power_buffer_union = None
        try:
            power_geoms = []
            buffer_deg = self.POWER_SUBSTATION_BUFFER_M * self.NO_FLY_BUFFER_DEGREES_PER_M
            for _idx, row in buildings.iterrows():
                geom = row.geometry
                if geom is None or not getattr(geom, "is_valid", True):
                    continue
                power_tag = str(row.get("power") or "").lower().strip()
                if power_tag in self.EXCLUDE_POWER_TAGS:
                    try:
                        power_geoms.append(geom.buffer(buffer_deg))
                    except Exception:
                        continue
            # Подстанции из OSM (power=substation и др.) — часто отдельные объекты, не в buildings
            bbox_osm = (buildings.total_bounds[3], buildings.total_bounds[1], buildings.total_bounds[2], buildings.total_bounds[0])
            for g in self._load_power_substation_geometries(bbox_osm, "EPSG:4326"):
                try:
                    power_geoms.append(g.buffer(buffer_deg))
                except Exception:
                    continue
            if power_geoms:
                power_buffer_union = unary_union(power_geoms)
        except Exception:
            power_buffer_union = None

        rows = []
        if station_type == "rooftop":
            buildings = self._building_footprint_area_m2(buildings)
            for idx, row in buildings.iterrows():
                geom = row.geometry
                if geom is None or not geom.is_valid:
                    continue
                # Не используем здания с power=* (подстанции, энергообъекты)
                power_tag = str(row.get("power") or "").lower().strip()
                if power_tag in self.EXCLUDE_POWER_TAGS:
                    continue
                area = row.get("area_m2", 0) or 0
                if area < min_roof:
                    continue
                tag = row.get("building")
                tag_str = str(tag).lower().strip() if tag is not None and not (isinstance(tag, float) and pd.isna(tag)) else ""
                # Только многоэтажки: явный тег из ALLOWED_ROOF_BUILDING_TAGS (без yes/true)
                if not tag_str or tag_str not in self.ALLOWED_ROOF_BUILDING_TAGS:
                    continue
                centroid = geom.centroid
                if not self._point_outside_no_fly(centroid, no_fly_zones, buffer_deg):
                    continue
                rows.append({
                    "geometry": centroid,
                    "station_type": "rooftop",
                    "area_m2": area,
                    "source": "building",
                })
        elif station_type == "ground":
            bbox = (buildings.total_bounds[3], buildings.total_bounds[1], buildings.total_bounds[2], buildings.total_bounds[0])
            zones = self._load_industrial_and_parking_areas(bbox, "EPSG:4326")
            # Наземные: центроиды парковок и промзон (площадка ≥ min_ground)
            for g in zones["parking"]:
                if g is None:
                    continue
                pt = g if g.geom_type == "Point" else g.centroid
                if not self._point_outside_no_fly(pt, no_fly_zones, buffer_deg):
                    continue
                try:
                    area = float(g.area) if hasattr(g, "area") and g.geom_type != "Point" else min_ground * 2
                    if g.geom_type == "Point":
                        area = min_ground * 2
                    else:
                        crs_utm = gpd.GeoSeries([g], crs="EPSG:4326").estimate_utm_crs()
                        area = gpd.GeoSeries([g], crs="EPSG:4326").to_crs(crs_utm).iloc[0].area
                    if area < min_ground and g.geom_type != "Point":
                        continue
                except Exception:
                    area = min_ground * 2
                rows.append({
                    "geometry": pt,
                    "station_type": "ground",
                    "area_m2": area,
                    "source": "parking",
                })
            for poly in zones["industrial"]:
                try:
                    c = poly.centroid
                    if not self._point_outside_no_fly(c, no_fly_zones, buffer_deg):
                        continue
                    # Не используем промзоны, которые являются подстанциями/энергообъектами (power=*)
                    try:
                        # У промзон из OSM теги лежат в properties GeoDataFrame; при загрузке выше мы их не сохраняем,
                        # поэтому фильтруем по геометрии позже через пересечение с подстанциями нельзя без доп. запросов.
                        # Здесь оставляем проверку только по buildings_union (см. ниже), а power-теги режем в _load_industrial_and_parking_areas.
                        pass
                    except Exception:
                        pass
                    # Не ставим гаражи/ТО прямо на зданиях (электробудки, корпуса заводов и т.п.)
                    if buildings_union is not None:
                        try:
                            if buildings_union.contains(c) or buildings_union.intersects(c):
                                continue
                        except Exception:
                            pass
                    crs_utm = gpd.GeoSeries([poly], crs="EPSG:4326").estimate_utm_crs()
                    area = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(crs_utm).iloc[0].area
                    if area < min_ground:
                        continue
                    rows.append({
                        "geometry": c,
                        "station_type": "ground",
                        "area_m2": area,
                        "source": "industrial",
                    })
                except Exception:
                    continue
        elif station_type in ("garage", "to"):
            # Гаражи/ТО: здания внутри промзон (industrial), без парковок и тяжёлых/опасных объектов.
            # Территории заводов (industrial_forbidden) исключаем — не ставим ТО/гараж на заводах.
            bbox = (buildings.total_bounds[3], buildings.total_bounds[1], buildings.total_bounds[2], buildings.total_bounds[0])
            zones = self._load_industrial_and_parking_areas(bbox, "EPSG:4326")
            industrial_polys = zones.get("industrial") or []
            forbidden_polys = zones.get("industrial_forbidden") or []
            if industrial_polys:
                try:
                    industrial_union = unary_union([g for g in industrial_polys if g is not None and getattr(g, "is_valid", True)])
                except Exception:
                    industrial_union = None
            else:
                industrial_union = None
            if forbidden_polys:
                try:
                    industrial_forbidden_union = unary_union([g for g in forbidden_polys if g is not None and getattr(g, "is_valid", True)])
                except Exception:
                    industrial_forbidden_union = None
            else:
                industrial_forbidden_union = None

            buildings_industrial = self._building_footprint_area_m2(buildings)
            for idx, row in buildings_industrial.iterrows():
                geom = row.geometry
                if geom is None or not geom.is_valid:
                    continue
                c = geom.centroid
                # Здание должно лежать внутри промзоны
                if industrial_union is not None:
                    try:
                        if not industrial_union.contains(c) and not industrial_union.touches(c):
                            continue
                    except Exception:
                        pass
                # Не ставим ТО/гараж на территории завода (заводские полигоны из OSM)
                if industrial_forbidden_union is not None:
                    try:
                        if industrial_forbidden_union.contains(c) or industrial_forbidden_union.intersects(c):
                            continue
                    except Exception:
                        pass
                # Здание с power=*, в том числе подстанции, и зона вокруг них (10 м) исключаем
                power_tag = str(row.get("power") or "").lower().strip()
                if power_tag in self.EXCLUDE_POWER_TAGS:
                    continue
                if power_buffer_union is not None:
                    try:
                        if power_buffer_union.contains(c) or power_buffer_union.intersects(c):
                            continue
                    except Exception:
                        pass
                # Фильтр по типу здания
                tag = row.get("building")
                tag_str = str(tag).lower().strip() if tag is not None and not (isinstance(tag, float) and pd.isna(tag)) else ""
                if tag_str and tag_str not in self.ALLOWED_INDUSTRIAL_BUILDING_TAGS:
                    continue
                # Избегаем опасных/крупных объектов по ключевым словам в name/industrial/craft/amenity
                text_fields = []
                for key in ("name", "industrial", "craft", "amenity", "man_made", "operator", "brand"):
                    val = row.get(key) if hasattr(row, "get") else None
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    text_fields.append(str(val).lower())
                combined = " ".join(text_fields)
                if combined:
                    if any(k in combined for k in self.FORBIDDEN_INDUSTRIAL_KEYWORDS):
                        continue
                # Площадь площадки/здания
                area = row.get("area_m2", 0) or 0
                if area < min_ground:
                    continue
                if not self._point_outside_no_fly(c, no_fly_zones, buffer_deg):
                    continue
                rows.append({
                    "geometry": c,
                    "station_type": "garage" if station_type == "garage" else "to",
                    "area_m2": area,
                    "source": "industrial_building",
                })
        if not rows:
            return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        self.logger.info(f"Кандидаты размещения ({station_type}): {len(gdf)}")
        return gdf

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
        # Зоны спроса — только по домам (центроиды зданий). Дороги не используются.
        # Вес спроса: многоквартирники — повышенный (APARTMENT_DEMAND_WEIGHT), остальные — 1
        pts_list = []
        weights_list = []
        for idx, b in buildings.iterrows():
            g = b.geometry
            if g is None or not getattr(g, "is_valid", True):
                continue
            c = g.centroid
            pts_list.append([c.x, c.y])
            weights_list.append(self.APARTMENT_DEMAND_WEIGHT if self._is_apartment(b) else 1)
        base_points = np.array(pts_list) if pts_list else np.empty((0, 2))
        base_weights = np.array(weights_list, dtype=float) if weights_list else np.ones(len(base_points))
        if len(base_points) > 0:
            self.logger.info(f"Точки для кластеризации: только по зданиям ({len(base_points)} центроидов), многоквартирники с весом {self.APARTMENT_DEMAND_WEIGHT}")
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
                clustering = DBSCAN(eps=eps_m, min_samples=dbscan_min_samples, metric="euclidean").fit(pts_utm)
                labels = np.array(clustering.labels_, dtype=int)
                # Точки-шум (label -1) не присваиваем кластерам — одиночные здания остаются выбросами, зоной спроса не считаются
                cluster_ids = sorted(set(labels) - {-1})
                n_noise = int((labels == -1).sum())
                if n_noise > 0:
                    self.logger.info(f"Точки-шум (одиночки/выбросы): {n_noise} — не образуют зону спроса")
                inv = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
                from shapely.ops import transform as sh_transform
                for lid in cluster_ids:
                    mask = labels == lid
                    cluster_pts = pts_utm[mask]
                    n_buildings = len(cluster_pts)
                    if n_buildings < min_buildings_for_zone:
                        continue  # одиночки и малые группы — выбросы, зону спроса не создаём
                    weight = float(base_weights[mask].sum())
                    cx, cy = cluster_pts.mean(axis=0)
                    lon, lat = inv.transform(cx, cy)
                    rows.append({"geometry": Point(lon, lat), "weight": int(round(weight))})
                    if return_hulls and hull_rows is not None and len(cluster_pts) >= 1:
                        try:
                            circles_utm = [Point(float(x), float(y)).buffer(eps_m) for x, y in cluster_pts]
                            region_utm = unary_union(circles_utm)
                            if region_utm is None or region_utm.is_empty:
                                continue
                            region_utm = region_utm.simplify(2.0)
                            region_4326 = sh_transform(lambda x, y: inv.transform(x, y), region_utm)
                            # Обрезаем область кластера по границе города
                            if city_boundary is not None and hasattr(city_boundary, "is_valid") and getattr(city_boundary, "is_valid", True) and region_4326.is_valid and not region_4326.is_empty:
                                try:
                                    clipped = region_4326.intersection(city_boundary)
                                    if clipped is not None and not clipped.is_empty and clipped.is_valid:
                                        region_4326 = clipped
                                except Exception:
                                    pass
                            # Исключаем беспилотные зоны из полигона зоны спроса (не выделяем no_fly в зонах спроса)
                            if no_fly_union is not None and not no_fly_union.is_empty and region_4326.is_valid and not region_4326.is_empty:
                                try:
                                    region_4326 = region_4326.difference(no_fly_union)
                                    if region_4326.is_empty:
                                        continue
                                except Exception:
                                    pass
                            if region_4326.is_valid and not region_4326.is_empty:
                                hull_rows.append({"geometry": region_4326, "weight": int(round(weight)), "cluster_id": len(hull_rows)})
                        except Exception as e:
                            self.logger.debug(f"Область кластера {lid}: {e}")
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
            for (gx, gy), weight in cell_centers.items():
                if cell_counts.get((gx, gy), 0) < min_buildings_for_zone:
                    continue  # одиночки в ячейке — выбросы, зону спроса не создаём
                if weight == 0:
                    continue
                if trans_inv:
                    lon, lat = trans_inv.transform(gx, gy)
                else:
                    lon, lat = float(gx), float(gy)
                rows.append({"geometry": Point(lon, lat), "weight": weight})
        if not rows:
            if return_hulls:
                return (gpd.GeoDataFrame(), gpd.GeoDataFrame())
            return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        gdf["weight"] = gdf.get("weight", 1)
        self.logger.info(f"Точки спроса ({method}): {len(gdf)}, сумма весов {gdf['weight'].sum()}")
        if return_hulls:
            hulls_gdf = gpd.GeoDataFrame(hull_rows, crs="EPSG:4326") if hull_rows else gpd.GeoDataFrame()
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
    
    # Смещение точки высадки от линии дороги в сторону здания (~5 м), но не заходя на здание
    OFFSET_FROM_ROAD_DEGREES = 0.00005
    MIN_SHIFT_DEGREES = 0.000005  # ~0.5 м — минимум, ниже не смещаем

    def _point_to_road(self, pt, edge_tree, edge_geoms):
        """Проецирует точку на ближайшее ребро дороги. Возвращает (lon, lat) на дороге."""
        if not edge_tree or not edge_geoms:
            return pt.x, pt.y
        nearest_idx = edge_tree.query_nearest(pt, return_distance=False)
        if not (hasattr(nearest_idx, '__len__') and len(nearest_idx) > 0):
            return pt.x, pt.y
        i = int(nearest_idx.flat[0]) if hasattr(nearest_idx, 'flat') else int(nearest_idx[0])
        line = edge_geoms[i]
        try:
            projected = line.interpolate(line.project(pt))
            return projected.x, projected.y
        except Exception:
            return pt.x, pt.y

    def _point_inside_any_building(self, pt, building_tree, building_geoms):
        """True, если точка внутри хотя бы одного полигона здания."""
        if not building_tree or not building_geoms:
            return False
        try:
            candidates = building_tree.query(pt)
            for c in (candidates if hasattr(candidates, '__iter__') and not isinstance(candidates, (int, float)) else [candidates]):
                idx = int(c)
                if idx < len(building_geoms) and building_geoms[idx].contains(pt):
                    return True
        except Exception:
            pass
        return False

    def _shift_near_road_no_building(self, rx, ry, pt, building_tree, building_geoms):
        """Смещает точку (rx,ry) с дороги в сторону pt, но не заходя на здания. Возвращает (lon, lat)."""
        dx = pt.x - rx
        dy = pt.y - ry
        d = (dx * dx + dy * dy) ** 0.5
        if d < 1e-9:
            return rx, ry
        shift = self.OFFSET_FROM_ROAD_DEGREES
        while shift >= self.MIN_SHIFT_DEGREES:
            lon_d = rx + shift * dx / d
            lat_d = ry + shift * dy / d
            delivery_pt = Point(lon_d, lat_d)
            if not self._point_inside_any_building(delivery_pt, building_tree, building_geoms):
                return lon_d, lat_d
            shift *= 0.5
        return rx, ry

    # Кварталы из дорог: допустимая площадь в м² (в UTM)
    BLOCK_MIN_AREA_M2 = 200.0
    BLOCK_MAX_AREA_M2 = 2.0e6

    def _get_blocks_from_roads(self, edge_geoms, city_boundary, crs_4326):
        """
        Строит полигоны кварталов из линий дорог (polygonize). Возвращает список полигонов в crs_4326,
        отфильтрованных по площади и границе города.
        """
        if not edge_geoms:
            return []
        try:
            polygons = list(polygonize(edge_geoms))
        except Exception as e:
            self.logger.warning(f"Ошибка polygonize дорог: {e}")
            return []
        if not polygons:
            return []
        try:
            crs_utm = gpd.GeoSeries([polygons[0].centroid], crs=crs_4326).estimate_utm_crs()
        except Exception:
            crs_utm = "EPSG:32637"
        result = []
        boundary_geom = city_boundary if city_boundary is not None else None
        for poly in polygons:
            if not poly.is_valid or poly.is_empty:
                continue
            try:
                poly_utm = gpd.GeoSeries([poly], crs=crs_4326).to_crs(crs_utm).iloc[0]
                area_m2 = poly_utm.area
            except Exception:
                continue
            if area_m2 < self.BLOCK_MIN_AREA_M2 or area_m2 > self.BLOCK_MAX_AREA_M2:
                continue
            if boundary_geom is not None:
                try:
                    if not poly.within(boundary_geom) and not poly.intersects(boundary_geom):
                        continue
                    # Оставляем только кварталы, центр которых внутри границы
                    if not boundary_geom.contains(poly.centroid):
                        continue
                except Exception:
                    continue
            result.append(poly)
        return result

    # Если из дорог получилось мало кварталов — делим город сеткой
    MIN_ROAD_BLOCKS_TO_USE = 8
    GRID_CELL_SIZE_M = 350.0
    GRID_MAX_CELLS = 500

    def _get_grid_blocks_fallback(self, buildings, city_boundary, building_centroids, crs_4326):
        """
        Запасной вариант: сетка ячеек по городу. Ограничено GRID_MAX_CELLS для скорости.
        """
        if not building_centroids or not buildings.crs:
            return []
        b = buildings.total_bounds
        try:
            crs_utm = gpd.GeoSeries([Point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)], crs=crs_4326).estimate_utm_crs()
        except Exception:
            return []
        try:
            bounds_geom = city_boundary if city_boundary is not None else box(b[0], b[1], b[2], b[3])
            bounds_gdf = gpd.GeoDataFrame(geometry=[bounds_geom], crs=crs_4326).to_crs(crs_utm)
            minx, miny, maxx, maxy = bounds_gdf.total_bounds
        except Exception:
            return []
        cell = self.GRID_CELL_SIZE_M
        cells = []
        x = minx
        while x < maxx and len(cells) < self.GRID_MAX_CELLS:
            y = miny
            while y < maxy and len(cells) < self.GRID_MAX_CELLS:
                poly_utm = box(x, y, x + cell, y + cell)
                poly_4326 = gpd.GeoSeries([poly_utm], crs=crs_utm).to_crs(crs_4326).iloc[0]
                for pt in building_centroids:
                    if pt is None:
                        continue
                    try:
                        if poly_4326.contains(pt) or poly_4326.intersects(pt):
                            cells.append(poly_4326)
                            break
                    except Exception:
                        pass
                y += cell
            x += cell
        return cells

    def _block_is_private_sector(self, block_poly, building_centroids, makes_non_private_list):
        """Квартал — частный сектор, если в нём есть здания и ни одно не «делает квартал не частным» (нет МКД, школ, офисов и т.д.)."""
        has_building = False
        has_non_private = False
        for pt, makes_np in zip(building_centroids, makes_non_private_list):
            if pt is None:
                continue
            try:
                if block_poly.contains(pt) or block_poly.intersects(pt):
                    has_building = True
                    if makes_np:
                        has_non_private = True
                        break
            except Exception:
                continue
        return has_building and not has_non_private

    def _four_corners_of_block(self, block_poly):
        """Четыре точки на углах ограничивающего прямоугольника. Оставлено для совместимости."""
        minx, miny, maxx, maxy = block_poly.bounds
        return [
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
        ]

    def _block_corner_points(self, block_poly, max_corners=32):
        """
        Углы квартала — вершины полигона (контур), а не bbox.
        Возвращает список (lon, lat). Для сложных полигонов ограничено max_corners.
        """
        try:
            if block_poly is None or not block_poly.is_valid or block_poly.is_empty:
                return []
            if block_poly.geom_type == 'Polygon':
                ring = block_poly.exterior
                if ring is None:
                    return []
                coords = list(ring.coords)
                if len(coords) <= 1:
                    return []
                # кольцо замкнуто: первый и последний совпадают — не дублируем
                points = [(float(x), float(y)) for x, y in coords[:-1]]
            elif block_poly.geom_type == 'MultiPolygon':
                # берём самый большой полигон по площади
                best = max(block_poly.geoms, key=lambda g: g.area if g.is_valid else 0)
                return self._block_corner_points(best, max_corners)
            else:
                return []
            if len(points) > max_corners:
                # упрощаем: равномерно берём max_corners вершин
                step = (len(points) - 1) / max(1, max_corners - 1)
                indices = [min(int(round(i * step)), len(points) - 1) for i in range(max_corners)]
                points = [points[i] for i in sorted(set(indices))]
            return points
        except Exception:
            return []

    def _load_entrances_bbox(self, bbox):
        """Загружает точки подъездов (entrance=*) из OSM по bbox. (north, south, east, west)."""
        try:
            entrances = ox.features_from_bbox(bbox=bbox, tags={"entrance": True})
            if entrances is not None and len(entrances) > 0:
                if entrances.crs is None:
                    entrances = entrances.set_crs("EPSG:4326")
                def to_point(g):
                    if g is None: return None
                    if g.geom_type == 'Point': return g
                    if g.geom_type == 'MultiPoint' and len(g.geoms) > 0: return g.geoms[0]
                    return g.centroid
                entrances = entrances.copy()
                entrances['geometry'] = entrances.geometry.apply(to_point)
                return entrances
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить подъезды из OSM: {e}")
        return gpd.GeoDataFrame()

    def _load_landuse_residential_areas(self, bbox, city_boundary, crs_4326):
        """
        Загружает полигоны для частного сектора: landuse=residential/allotments и place=neighbourhood/suburb (полигоны).
        """
        north, south, east, west = bbox
        polygons = []
        for tags in (
            {"landuse": ["residential", "allotments"]},
            {"place": ["neighbourhood", "suburb"]},
        ):
            try:
                gdf = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
                if gdf is None or len(gdf) == 0:
                    continue
                if gdf.crs is not None and gdf.crs != crs_4326:
                    gdf = gdf.to_crs(crs_4326)
                for _, row in gdf.iterrows():
                    g = row.geometry
                    if g is None or not getattr(g, "is_valid", True):
                        continue
                    if g.geom_type == "Polygon":
                        polygons.append(g)
                    elif g.geom_type == "MultiPolygon":
                        for poly in g.geoms:
                            if poly.is_valid and not poly.is_empty:
                                polygons.append(poly)
            except Exception as e:
                self.logger.debug(f"Загрузка {tags}: {e}")
        if polygons:
            self.logger.info(f"Загружено полигонов landuse/place (частный сектор): {len(polygons)}")
        if not polygons or city_boundary is None:
            return polygons
        filtered = []
        for poly in polygons:
            try:
                if not poly.intersects(city_boundary):
                    continue
                if not city_boundary.contains(poly.centroid) and not poly.centroid.within(city_boundary):
                    continue
                filtered.append(poly)
            except Exception:
                continue
        return filtered if filtered else polygons

    def _landuse_area_is_private_sector(self, poly, building_centroids, makes_non_private_list):
        """
        Классификация: частный сектор, если в полигоне landuse нет высотных/не-частных зданий.
        """
        try:
            prep_poly = prep(poly)
        except Exception:
            prep_poly = None
        for pt, makes_np in zip(building_centroids, makes_non_private_list):
            if pt is None or not makes_np:
                continue
            try:
                if (prep_poly if prep_poly is not None else poly).contains(pt):
                    return False
            except Exception:
                continue
        return True

    def get_delivery_points(
        self,
        buildings: gpd.GeoDataFrame,
        road_graph,
        city_boundary=None,
    ) -> gpd.GeoDataFrame:
        """
        Точки высадки. Частный дом определяется автоматически: по тегу OSM (building=house, yes, detached и т.д.)
        либо по попаданию в зону частного сектора (landuse=residential/allotments, кварталы без МКД/школ/офисов).
        В частном секторе — только 4 точки на углах квартала; на остальные здания — по подъездам (МКД) или по одному.
        """
        if buildings is None or len(buildings) == 0:
            return gpd.GeoDataFrame()
        if buildings.crs is None:
            buildings = buildings.set_crs("EPSG:4326")
        buildings = buildings.to_crs("EPSG:4326")
        if city_boundary is not None:
            try:
                # Проекция в метры для корректного centroid/within (избегаем UserWarning о geographic CRS)
                crs_utm = buildings.estimate_utm_crs()
                buildings_proj = buildings.to_crs(crs_utm)
                boundary_proj = gpd.GeoSeries([city_boundary], crs=buildings.crs).to_crs(crs_utm).iloc[0]
                mask = buildings_proj.geometry.centroid.within(boundary_proj)
                buildings = buildings[mask]
            except Exception as e:
                self.logger.warning(f"Фильтрация зданий по границе: {e}")
        if len(buildings) == 0:
            return gpd.GeoDataFrame()
        
        if road_graph is None or len(road_graph.nodes) == 0:
            edge_geoms = []
        else:
            _, gdf_edges = ox.graph_to_gdfs(road_graph)
            if gdf_edges.crs is not None and gdf_edges.crs != buildings.crs:
                gdf_edges = gdf_edges.to_crs(buildings.crs)
            edge_geoms = list(gdf_edges.geometry) if len(gdf_edges) > 0 else []
        edge_tree = STRtree(edge_geoms) if edge_geoms else None
        
        building_geoms = list(buildings.geometry)
        building_tree = STRtree(building_geoms) if building_geoms else None
        is_apartment = [self._is_apartment(buildings.iloc[i]) for i in range(len(buildings))]
        
        # Подъезды из OSM (ограничиваем объём для больших городов)
        xmin, ymin, xmax, ymax = buildings.total_bounds
        entrances_gdf = self._load_entrances_bbox((ymax, ymin, xmax, xmin))
        if len(entrances_gdf) > 3000:
            entrances_gdf = entrances_gdf.iloc[:3000]
        building_entrances = {}
        if len(entrances_gdf) > 0 and building_tree is not None:
            for _, ent in entrances_gdf.iterrows():
                g = ent.geometry
                if g is None or not getattr(g, 'is_valid', True):
                    continue
                if g.geom_type == 'Point':
                    pt = g
                elif g.geom_type == 'MultiPoint' and len(g.geoms) > 0:
                    pt = g.geoms[0]
                else:
                    pt = g.centroid
                try:
                    candidates = building_tree.query(pt)
                except Exception:
                    continue
                if hasattr(candidates, '__iter__') and not isinstance(candidates, (int, float)):
                    for c in candidates:
                        idx = int(c)
                        if idx >= len(building_geoms):
                            continue
                        poly = building_geoms[idx]
                        try:
                            if poly.buffer(0.00005).contains(pt) and is_apartment[idx]:
                                building_entrances.setdefault(idx, []).append(pt)
                                break
                        except Exception:
                            pass
        
        makes_non_private = [self._building_makes_block_non_private(buildings.iloc[i]) for i in range(len(buildings))]
        is_private = [self._is_private_house(buildings.iloc[i], is_apartment[i]) for i in range(len(buildings))]
        building_centroids = []
        for i in range(len(buildings)):
            geom = buildings.iloc[i].geometry
            if geom is not None and geom.is_valid:
                building_centroids.append(geom.centroid)
            else:
                building_centroids.append(None)

        # Вариант 1 (приоритет): частный сектор по landuse — residential, allotments, neighbourhood, suburb
        bbox_osm = (ymax, ymin, xmax, xmin)  # north, south, east, west
        landuse_polygons = self._load_landuse_residential_areas(bbox_osm, city_boundary, buildings.crs)
        # Частный сектор по landuse только там, где внутри нет многоквартирных/школ/офисов
        max_landuse_to_classify = 150
        if len(landuse_polygons) > max_landuse_to_classify:
            landuse_polygons = sorted(landuse_polygons, key=lambda p: p.area, reverse=True)[:max_landuse_to_classify]
        private_sector_landuse = [
            p for p in landuse_polygons
            if self._landuse_area_is_private_sector(p, building_centroids, makes_non_private)
        ]
        if private_sector_landuse:
            self.logger.info(f"Частный сектор по landuse: {len(private_sector_landuse)} полигонов")

        # Кварталы по дорогам/сетке — используем вместе с landuse, чтобы очистить все частные кварталы
        all_blocks = []
        max_edges_for_polygonize = 6000
        if edge_geoms and len(edge_geoms) <= max_edges_for_polygonize:
            road_blocks = self._get_blocks_from_roads(edge_geoms, city_boundary, buildings.crs)
            if len(road_blocks) >= self.MIN_ROAD_BLOCKS_TO_USE:
                all_blocks = road_blocks
        if not all_blocks and building_centroids:
            all_blocks = self._get_grid_blocks_fallback(buildings, city_boundary, building_centroids, buildings.crs)
        private_sector_blocks = [
            bp for bp in all_blocks
            if self._block_is_private_sector(bp, building_centroids, makes_non_private)
        ]

        # Кварталы внутри частного сектора (landuse): считаем их тоже частным сектором,
        # чтобы делить зону на блоки, а не одну большую зону с 4 точками
        blocks_inside_private_landuse = []
        if private_sector_landuse and all_blocks:
            try:
                prep_landuse = [prep(p) if p.is_valid else None for p in private_sector_landuse]
                for block_poly in all_blocks:
                    if not block_poly.is_valid or block_poly.is_empty:
                        continue
                    try:
                        c = block_poly.centroid
                        for i, poly in enumerate(private_sector_landuse):
                            if (prep_landuse[i] if prep_landuse[i] is not None else poly).contains(c):
                                blocks_inside_private_landuse.append(block_poly)
                                break
                    except Exception:
                        continue
            except Exception as e:
                self.logger.debug(f"Блоки внутри landuse: {e}")

        # Объединяем: кварталы по зданиям + кварталы внутри landuse (без дубликатов по центру)
        seen_centroids = set()
        effective_private_blocks = []
        for bp in private_sector_blocks + blocks_inside_private_landuse:
            try:
                key = (round(bp.centroid.x, 6), round(bp.centroid.y, 6))
                if key in seen_centroids:
                    continue
                seen_centroids.add(key)
                effective_private_blocks.append(bp)
            except Exception:
                effective_private_blocks.append(bp)
        if blocks_inside_private_landuse:
            self.logger.info(f"Кварталов внутри частного сектора (landuse): {len(blocks_inside_private_landuse)}, всего частных кварталов: {len(effective_private_blocks)}")

        # Небольшой буфер (~20 м), чтобы здания на границе частного полигона тоже очищались
        _buffer_deg = 0.0002
        try:
            landuse_for_check = [p.buffer(_buffer_deg) if p.is_valid else p for p in private_sector_landuse]
            blocks_for_check = [p.buffer(_buffer_deg) if p.is_valid else p for p in effective_private_blocks]
        except Exception:
            landuse_for_check = private_sector_landuse
            blocks_for_check = effective_private_blocks

        def _centroid_in_private_sector(centroid):
            if centroid is None:
                return False
            for poly in landuse_for_check:
                try:
                    if poly.contains(centroid) or poly.intersects(centroid):
                        return True
                except Exception:
                    continue
            for block_poly in blocks_for_check:
                try:
                    if block_poly.contains(centroid) or block_poly.intersects(centroid):
                        return True
                except Exception:
                    continue
            return False

        # Частный дом: по тегу OSM ИЛИ по попаданию в зону частного сектора (landuse/кварталы)
        for i in range(len(buildings)):
            if building_centroids[i] is None:
                continue
            if is_private[i]:
                continue  # уже частный по тегу
            if not (is_apartment[i] or makes_non_private[i]) and _centroid_in_private_sector(building_centroids[i]):
                is_private[i] = True  # частный по расположению в частном секторе

        rows = []
        for i in range(len(buildings)):
            try:
                b = buildings.iloc[i]
                geom = b.geometry
                if geom is None or not geom.is_valid:
                    continue
                # Не ставим точку на здание, если оно частное (по тегу OSM или по зоне)
                if is_private[i]:
                    continue
                centroid = geom.centroid
                lon_c, lat_c = centroid.x, centroid.y
                w = self.APARTMENT_DEMAND_WEIGHT if is_apartment[i] else 1
                if is_apartment[i] and i in building_entrances and len(building_entrances[i]) > 0:
                    for entrance_pt in building_entrances[i]:
                        rx, ry = self._point_to_road(entrance_pt, edge_tree, edge_geoms)
                        lon_d, lat_d = self._shift_near_road_no_building(rx, ry, entrance_pt, building_tree, building_geoms)
                        rows.append({
                            'geometry': Point(lon_d, lat_d),
                            'delivery_lon': lon_d,
                            'delivery_lat': lat_d,
                            'weight': w,
                        })
                else:
                    rx, ry = self._point_to_road(centroid, edge_tree, edge_geoms)
                    lon_d, lat_d = self._shift_near_road_no_building(rx, ry, centroid, building_tree, building_geoms)
                    rows.append({
                            'geometry': Point(lon_d, lat_d),
                            'delivery_lon': lon_d,
                            'delivery_lat': lat_d,
                            'weight': w,
                    })
            except Exception as e:
                self.logger.debug(f"Здание {i}: {e}")
                continue

        # На каждом квартале — минимум один узел: центр квартала, привязанный к дороге (для забора заказов)
        if building_centroids and (edge_tree and edge_geoms) and all_blocks:
            seen = set()
            # Landuse-полигоны, внутри которых нет кварталов — одна точка на полигон
            landuse_for_center = []
            for p in private_sector_landuse:
                has_block_inside = False
                try:
                    for b in all_blocks:
                        if not b.is_valid or b.is_empty:
                            continue
                        if p.contains(b.centroid):
                            has_block_inside = True
                            break
                except Exception:
                    pass
                if not has_block_inside:
                    landuse_for_center.append(p)
            # Все кварталы (all_blocks) — по одному узлу на квартал у дороги посередине
            for block_poly in landuse_for_center + all_blocks:
                try:
                    if not block_poly.is_valid or block_poly.is_empty:
                        continue
                    centroid = block_poly.centroid
                    pt = centroid
                    rx, ry = self._point_to_road(pt, edge_tree, edge_geoms)
                    lon_d, lat_d = self._shift_near_road_no_building(rx, ry, pt, building_tree, building_geoms)
                    key = (round(lon_d, 6), round(lat_d, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        'geometry': Point(lon_d, lat_d),
                        'delivery_lon': lon_d,
                        'delivery_lat': lat_d,
                        'weight': 1,
                    })
                except Exception as e:
                    self.logger.debug(f"Квартал: {e}")
            self.logger.info(f"Узлы по кварталам: landuse без кварталов {len(landuse_for_center)}, всего кварталов {len(all_blocks)}, уникальных узлов у дороги: {len(seen)}")

        if not rows:
            return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame(rows, crs=buildings.crs)
        self.logger.info(f"Точки высадки: {len(gdf)} (многоквартирные — по подъездам; на каждом квартале — узел у дороги посередине)")
        return gdf
    
    def address_to_coords(self, address, city_name=None):
        """Улучшенное геокодирование с поддержкой российских адресов"""
        if not address or not address.strip():
            return None
        
        address = address.strip()
        
        # Если это уже координаты
        if ',' in address:
            try:
                coords = tuple(map(float, [x.strip() for x in address.split(',')]))
                if len(coords) == 2 and -90 <= coords[0] <= 90 and -180 <= coords[1] <= 180:
                    return coords
            except ValueError:
                pass
        
        # Если указан город, добавляем его к адресу
        if city_name and city_name.strip():
            city_name = city_name.strip()
            # Убираем "Russia" из названия города если есть
            if city_name.endswith(', Russia'):
                city_name = city_name[:-8].strip()
            
            # Пробуем разные варианты с городом
            search_variants = [
                f"{address}, {city_name}",
                f"{address}, {city_name}, Россия",
                f"{address}, {city_name}, Russia",
                f"{address}, {city_name}, РФ"
            ]
        else:
            # Если город не указан, используем общие варианты
            search_variants = [
                address,
                f"{address}, Россия",
                f"{address}, Russia",
                f"{address}, РФ"
            ]
        
        for geocoder_name, geocoder in self.geolocators.items():
            for search_address in search_variants:
                try:
                    self.logger.info(f"Поиск координат для: {search_address} (геокодер: {geocoder_name})")
                    location = geocoder.geocode(search_address, timeout=10)
                    if location:
                        coords = (location.latitude, location.longitude)
                        self.logger.info(f"Найдены координаты: {coords}")
                        
                        # Если указан город, проверяем что координаты находятся в разумных пределах города
                        if city_name:
                            if not self._validate_coords_in_city(coords, city_name):
                                self.logger.warning(f"Координаты {coords} не находятся в границах города {city_name}")
                                continue
                        
                        return coords
                except Exception as e:
                    self.logger.warning(f"Ошибка геокодирования '{search_address}': {e}")
                    continue
        
        self.logger.warning(f"Не удалось найти координаты для адреса: {address}")
        return None

    def coords_to_address(self, coords: tuple, language: str = 'ru'):
        """Обратное геокодирование координат в строку улицы/адреса."""
        try:
            if not coords or len(coords) != 2:
                return None
            lat, lon = coords
            location = self._reverse((lat, lon), language=language, timeout=10)
            if location and location.address:
                return location.address
            return None
        except Exception as e:
            self.logger.warning(f"Ошибка обратного геокодирования {coords}: {e}")
            return None
    
    def _validate_coords_in_city(self, coords, city_name):
        """Проверка что координаты находятся в разумных пределах города"""
        city_bounds = {
            'волгоград': {'lat': (48.5, 49.0), 'lon': (44.0, 45.0)},
            'volgograd': {'lat': (48.5, 49.0), 'lon': (44.0, 45.0)},
            'москва': {'lat': (55.5, 55.9), 'lon': (37.3, 37.9)},
            'moscow': {'lat': (55.5, 55.9), 'lon': (37.3, 37.9)},
            'санкт-петербург': {'lat': (59.8, 60.1), 'lon': (30.0, 30.7)},
            'st petersburg': {'lat': (59.8, 60.1), 'lon': (30.0, 30.7)},
            'st. petersburg': {'lat': (59.8, 60.1), 'lon': (30.0, 30.7)},
            'petersburg': {'lat': (59.8, 60.1), 'lon': (30.0, 30.7)},
        }
        
        normalized_city = city_name.lower().strip()
        for city_key, bounds in city_bounds.items():
            if city_key in normalized_city:
                lat, lon = coords
                if bounds['lat'][0] <= lat <= bounds['lat'][1] and bounds['lon'][0] <= lon <= bounds['lon'][1]:
                    return True
                else:
                    self.logger.warning(f"Координаты {coords} не в границах {city_name}: lat {bounds['lat']}, lon {bounds['lon']}")
                    return False
        
        # Если город не найден в списке, считаем координаты валидными
        return True
    
    def _load_from_cache(self, cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            if os.path.exists(cache_file):
                os.remove(cache_file)
            raise
    
    def _normalize_city_name(self, city_name):
        """Минимальная нормализация: тримминг без привязки к стране (универсальность)"""
        return city_name.strip()
    
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
    
    def get_redis_client(self):
        """Возвращает Redis клиент для использования в других сервисах"""
        return self._redis