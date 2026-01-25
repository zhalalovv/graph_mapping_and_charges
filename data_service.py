import os
import osmnx as ox
import geopandas as gpd
import time
import pickle
import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import logging
import json
from redis import Redis
from shapely.geometry import box, MultiPoint
from shapely.ops import unary_union

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
    
    def get_city_data(self, city_name: str, network_type: str = 'drive', simplify: bool = True):
        """Получение данных города (универсально для любой страны)"""
        normalized_name = city_name.strip()
        key_suffix = f"{self._sanitize_name(normalized_name)}__{self._sanitize_name(network_type)}__{int(bool(simplify))}"
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
                        return cached_data
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
                    return self._download_city_data(normalized_name, cache_file, redis_key, network_type=network_type, simplify=simplify)
                return cached_data
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки кэша: {e}, перезагружаем данные")
                os.remove(cache_file)
        
        return self._download_city_data(normalized_name, cache_file, redis_key, network_type=network_type, simplify=simplify)
    
    def _download_city_data(self, city_name, cache_file, redis_key: str | None = None, *, network_type: str = 'drive', simplify: bool = True):
        self._update_progress("download", 0, "Начало загрузки данных")
        
        try:
            self._update_progress("download", 20, "Загрузка дорожной сети")
            # Универсальная загрузка для любого города/страны
            road_graph = None
            for query_variant in [city_name]:
                try:
                    road_graph = ox.graph_from_place(query_variant, network_type=network_type, simplify=simplify)
                    if len(road_graph.nodes) > 0:
                        self.logger.info(f"Успешно загружена дорожная сеть для: {query_variant}")
                        break
                except Exception as e:
                    self.logger.warning(f"Не удалось загрузить для '{query_variant}': {e}")
                    continue
            
            if road_graph is None or len(road_graph.nodes) == 0:
                raise Exception(f"Не удалось загрузить дорожную сеть для города: {city_name}")
            
            self._update_progress("download", 40, "Загрузка границ города")
            city_boundary = None
            
            # Сначала получаем узлы дорожного графа для fallback
            gdf_nodes, _ = ox.graph_to_gdfs(road_graph)
            
            try:
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
                        bbox = city_boundary.bounds
                        # north, south, east, west (bbox: minx, miny, maxx, maxy)
                        buildings = ox.features_from_bbox(
                            bbox[3], bbox[1], bbox[2], bbox[0],  # north=maxy, south=miny, east=maxx, west=minx
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
            
            self._update_progress("download", 80, "Загрузка запретных зон")
            no_fly_zones = self._get_no_fly_zones(city_name)
            
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
    
    def _get_no_fly_zones(self, city_name):
        """
        Загружает беспилотные зоны из OpenStreetMap
        
        Признаки для вырезания:
        1. Аэропорты и аэродромы (aeroway=aerodrome, airport)
        2. Военные объекты (military=*, landuse=military)
        3. Зоны безопасности вокруг аэропортов (buffer)
        4. Запретные зоны (restriction:drone=no)
        5. Зоны вокруг важных объектов (government, nuclear)
        
        Returns:
            GeoDataFrame с беспилотными зонами
        """
        no_fly_zones = []
        
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
                    # Добавляем буфер безопасности вокруг аэропортов (5 км)
                    for idx, airport in airports.iterrows():
                        if airport.geometry is not None and airport.geometry.is_valid:
                            # Буфер в градусах: ~5км ≈ 0.045°
                            buffer_zone = airport.geometry.buffer(0.045)
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
                            # Буфер 1 км вокруг военных объектов
                            buffer_zone = obj.geometry.buffer(0.009)
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
                            # Буфер 3 км вокруг атомных объектов
                            buffer_zone = obj.geometry.buffer(0.027)
                            no_fly_zones.append(buffer_zone)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки атомных объектов: {e}")
            
            # 4. Правительственные объекты (опционально, меньший буфер)
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
                            # Буфер 500м вокруг правительственных объектов
                            buffer_zone = obj.geometry.buffer(0.0045)
                            no_fly_zones.append(buffer_zone)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки правительственных объектов: {e}")
            
            # 5. Явные запретные зоны из OSM (если есть теги)
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
                            no_fly_zones.append(zone.geometry)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки запретных зон: {e}")
            
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
    
    def get_redis_client(self):
        """Возвращает Redis клиент для использования в других сервисах"""
        return self._redis
    
    def generate_graph_cache_key(self, city_name: str, network_type: str, simplify: bool, 
                                 graph_type: str, grid_spacing: float, connect_diagonal: bool) -> str:
        """Генерирует ключ кэша для графа на основе всех параметров"""
        normalized_name = city_name.strip()
        key_parts = [
            self._sanitize_name(normalized_name),
            self._sanitize_name(network_type),
            str(int(bool(simplify))),
            self._sanitize_name(graph_type),
            f"{grid_spacing:.6f}".replace('.', '_'),
            str(int(bool(connect_diagonal)))
        ]
        key_suffix = "__".join(key_parts)
        return f"drone_planner:graph:{key_suffix}"