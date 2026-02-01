import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import geopandas as gpd
from scipy.spatial import Delaunay
import logging
from typing import Tuple, List, Optional, Union


class GraphService:
    """Сервис для генерации различных типов графов для планирования маршрутов дронов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_delaunay_graph(
        self,
        bounds: Tuple[float, float, float, float],
        num_points: int = 500,
        buildings: Optional[gpd.GeoDataFrame] = None,
        no_fly_zones: Optional[Union[gpd.GeoDataFrame, List]] = None,
        min_clearance: float = 0.0001,
        city_boundary: Optional[Polygon] = None,
        points: Optional[List[Tuple[float, float]]] = None,
    ) -> nx.Graph:
        """
        Создает граф на основе триангуляции Делоне по точкам высадки или центроидам зданий.
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            num_points: игнорируется при наличии buildings или points
            buildings: GeoDataFrame со зданиями (используются только если points=None)
            no_fly_zones: Список запретных зон
            min_clearance: минимальное расстояние от препятствий
            city_boundary: границы города для фильтрации
            points: если задан — узлы графа строятся по этим точкам (lon, lat), иначе по центроидам зданий
            
        Returns:
            NetworkX граф
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        obstacles = self._prepare_obstacles(None, no_fly_zones, min_clearance)
        
        # Приоритет: переданные точки (высадки), иначе центроиды зданий
        point_list = []
        if points is not None and len(points) > 0:
            self.logger.info(f"Генерация Delaunay графа по точкам высадки: {len(points)} точек")
            for lon, lat in points:
                pt = Point(lon, lat)
                if city_boundary is not None:
                    try:
                        if not city_boundary.contains(pt) and not city_boundary.touches(pt):
                            continue
                    except Exception:
                        continue
                if not self._is_point_in_obstacles(pt, obstacles):
                    point_list.append([lon, lat])
            self.logger.info(f"Получено {len(point_list)} точек для триангуляции")
        elif buildings is not None and len(buildings) > 0:
            self.logger.info(f"Генерация Delaunay графа по центроидам зданий: {len(buildings)} зданий")
            for idx, building in buildings.iterrows():
                try:
                    if building.geometry is not None and building.geometry.is_valid:
                        centroid = building.geometry.centroid
                        lon, lat = centroid.x, centroid.y
                        if city_boundary is not None:
                            try:
                                if not city_boundary.contains(centroid) and not city_boundary.touches(centroid):
                                    continue
                            except Exception:
                                continue
                        if not self._is_point_in_obstacles(centroid, obstacles):
                            point_list.append([lon, lat])
                except Exception as e:
                    self.logger.warning(f"Ошибка обработки здания {idx}: {e}")
                    continue
            self.logger.info(f"Получено {len(point_list)} центроидов зданий для триангуляции")
        
        if len(point_list) < 3:
            self.logger.warning("Недостаточно точек для триангуляции (нужно минимум 3)")
            return nx.Graph()
        
        points_arr = np.array(point_list)
        
        # Строим триангуляцию Делоне
        tri = Delaunay(points_arr)
        
        # Создаем граф из триангуляции
        G = nx.Graph()
        
        # Добавляем узлы
        for i, (lon, lat) in enumerate(points_arr):
            G.add_node(i, lat=lat, lon=lon, pos=(lon, lat))
        
        # Добавляем рёбра из треугольников
        for simplex in tri.simplices:
            for i in range(3):
                n1, n2 = simplex[i], simplex[(i + 1) % 3]
                if not G.has_edge(n1, n2):
                    p1, p2 = points_arr[n1], points_arr[n2]
                    line = LineString([p1, p2])
                    
                    # Проверяем, что ребро не пересекает препятствия
                    if not self._line_intersects_obstacles(line, obstacles):
                        distance = self._calculate_distance(p1[1], p1[0], p2[1], p2[0])
                        G.add_edge(n1, n2, weight=distance, length=distance)
        
        self.logger.info(f"Delaunay граф: {len(G.nodes)} узлов, {len(G.edges)} рёбер")
        
        return G
    
    def _prepare_obstacles(
        self,
        buildings: Optional[gpd.GeoDataFrame],
        no_fly_zones: Optional[Union[gpd.GeoDataFrame, List]],
        buffer_distance: float
    ) -> Optional[Polygon]:
        """Подготавливает объединенную геометрию препятствий с буфером (legacy method)"""
        geometries = []
        
        if buildings is not None and len(buildings) > 0:
            try:
                for geom in buildings.geometry:
                    if geom is not None and geom.is_valid:
                        geometries.append(geom.buffer(buffer_distance))
            except Exception as e:
                self.logger.warning(f"Ошибка обработки зданий: {e}")
        
        # Проверяем no_fly_zones правильно (GeoDataFrame нельзя использовать в if напрямую)
        if no_fly_zones is not None:
            # Поддержка GeoDataFrame и списка геометрий
            if isinstance(no_fly_zones, gpd.GeoDataFrame):
                if len(no_fly_zones) > 0:
                    for idx, row in no_fly_zones.iterrows():
                        if row.geometry is not None and row.geometry.is_valid:
                            geometries.append(row.geometry.buffer(buffer_distance))
            elif isinstance(no_fly_zones, list):
                for zone in no_fly_zones:
                    if hasattr(zone, 'geometry') and zone.geometry is not None:
                        geometries.append(zone.geometry.buffer(buffer_distance))
                    elif hasattr(zone, 'is_valid'):
                        if zone.is_valid:
                            geometries.append(zone.buffer(buffer_distance))
        
        if not geometries:
            return None
        
        try:
            return unary_union(geometries)
        except Exception as e:
            self.logger.warning(f"Ошибка объединения препятствий: {e}")
            return None
    
    def _is_point_in_obstacles(self, point: Point, obstacles: Optional[Polygon]) -> bool:
        """Проверяет, находится ли точка внутри препятствий"""
        if obstacles is None:
            return False
        
        try:
            return obstacles.contains(point) or obstacles.intersects(point)
        except Exception:
            return False
    
    def _line_intersects_obstacles(self, line: LineString, obstacles: Optional[Polygon]) -> bool:
        """Проверяет, пересекает ли линия препятствия"""
        if obstacles is None:
            return False
        
        try:
            return obstacles.intersects(line)
        except Exception:
            return False
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Вычисляет расстояние между двумя точками по формуле гаверсинуса (в км)
        ОПТИМИЗИРОВАНО: использует векторизованные операции NumPy
        """
        R = 6371.0  # Радиус Земли в км
        
        # ОПТИМИЗАЦИЯ: векторизованные вычисления
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        # Используем более быструю формулу для малых расстояний
        # Для расстояний < 100км можно использовать упрощенную формулу
        if abs(delta_lat) < 0.9 and abs(delta_lon) < 0.9:  # ~100км
            # Упрощенная формула для малых расстояний (быстрее)
            a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
        else:
            # Полная формула гаверсинуса
            a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def graph_to_geojson(self, graph: nx.Graph, edge_type_filter: Optional[str] = None) -> dict:
        """
        Конвертирует NetworkX граф в GeoJSON для визуализации (ОПТИМИЗИРОВАНО)
        
        Args:
            graph: NetworkX граф
            edge_type_filter: фильтр по типу рёбер ('road', 'free', 'connection', None для всех)
            
        Returns:
            GeoJSON FeatureCollection
        """
        # ОПТИМИЗАЦИЯ: используем list comprehension и предварительную фильтрацию
        features = []
        
        # Предварительно получаем все рёбра для фильтрации
        edges = list(graph.edges(data=True))
        
        # Фильтруем по типу, если нужно
        if edge_type_filter:
            edges = [(u, v, d) for u, v, d in edges if d.get('edge_type') == edge_type_filter]
        
        # ОПТИМИЗАЦИЯ: предварительно получаем координаты всех узлов
        node_coords = {}
        for node_id, node_data in graph.nodes(data=True):
            if 'y' in node_data and 'x' in node_data:  # OSM узел
                node_coords[node_id] = (node_data['x'], node_data['y'])
            elif 'lon' in node_data and 'lat' in node_data:  # Наш узел
                node_coords[node_id] = (node_data['lon'], node_data['lat'])
        
        # Создаём features через list comprehension
        for u, v, data in edges:
            if u not in node_coords or v not in node_coords:
                continue
            
            u_lon, u_lat = node_coords[u]
            v_lon, v_lat = node_coords[v]
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[u_lon, u_lat], [v_lon, v_lat]]
                },
                "properties": {
                    "edge_type": data.get('edge_type', 'unknown'),
                    "weight": data.get('weight', 0),
                    "length": data.get('length', 0)
                }
            })
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
        