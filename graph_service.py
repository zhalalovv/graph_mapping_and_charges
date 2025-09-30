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
            self.logger.warning("Redis not available for DataService cache; falling back to disk")
    
    def add_progress_callback(self, callback):
        self.progress_callbacks.append(callback)
    
    def _update_progress(self, stage, percentage, message=""):
        for callback in self.progress_callbacks:
            callback(stage, percentage, message)
    
    def get_city_data(self, city_name: str):
        """Получение данных города (универсально для любой страны)"""
        normalized_name = city_name.strip()
        cache_file = os.path.join(self.cache_dir, f"{self._sanitize_name(normalized_name)}.pkl")
        redis_key = f"drone_planner:city:{self._sanitize_name(normalized_name)}"
        
        # Try Redis first
        if self._redis is not None:
            try:
                blob = self._redis.get(redis_key)
                if blob:
                    self._update_progress("cache", 100, "Загрузка из Redis")
                    return pickle.loads(blob)
            except Exception as e:
                self.logger.warning(f"Ошибка чтения из Redis: {e}")

        if os.path.exists(cache_file):
            self._update_progress("cache", 100, "Загрузка из кэша")
            try:
                return self._load_from_cache(cache_file)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки кэша: {e}, перезагружаем данные")
                os.remove(cache_file)
        
        return self._download_city_data(normalized_name, cache_file, redis_key)
    
    def _download_city_data(self, city_name, cache_file, redis_key: str | None = None):
        self._update_progress("download", 0, "Начало загрузки данных")
        
        try:
            self._update_progress("download", 20, "Загрузка дорожной сети")
            # Универсальная загрузка для любого города/страны
            road_graph = None
            for query_variant in [city_name]:
                try:
                    road_graph = ox.graph_from_place(query_variant, network_type='drive', simplify=True)
                    if len(road_graph.nodes) > 0:
                        self.logger.info(f"Успешно загружена дорожная сеть для: {query_variant}")
                        break
                except Exception as e:
                    self.logger.warning(f"Не удалось загрузить для '{query_variant}': {e}")
                    continue
            
            if road_graph is None or len(road_graph.nodes) == 0:
                raise Exception(f"Не удалось загрузить дорожную сеть для города: {city_name}")
            
            self._update_progress("download", 50, "Загрузка зданий")
            buildings = gpd.GeoDataFrame()  # Пустой GeoDataFrame по умолчанию
            try:
                buildings = ox.features_from_place(city_name, tags={"building": True})
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить здания: {e}")
            
            self._update_progress("download", 80, "Загрузка запретных зон")
            no_fly_zones = self._get_no_fly_zones(city_name)
            
            data = {
                'road_graph': road_graph,
                'buildings': buildings,
                'no_fly_zones': no_fly_zones,
                'city_name': city_name,
                'timestamp': time.time(),
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
        return []  # Заглушка - можно добавить реальные данные
    
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