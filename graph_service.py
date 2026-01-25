import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
import geopandas as gpd
from scipy.spatial import Delaunay, cKDTree
import logging
import osmnx as ox
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass


@dataclass
class QuadNode:
    """Узел квадродерева"""
    bounds: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    depth: int
    importance: float  # важность зоны (0-1)
    children: Optional[List['QuadNode']] = None
    points: Optional[List[Tuple[float, float]]] = None  # (lon, lat)
    
    def should_subdivide(self, max_depth: int, min_importance: float) -> bool:
        """Определяет, нужно ли разбивать узел дальше"""
        if self.depth >= max_depth:
            return False
        if self.importance < min_importance:
            return False
        return True
    
    def subdivide(self) -> List['QuadNode']:
        """Разбивает узел на 4 дочерних"""
        min_lon, min_lat, max_lon, max_lat = self.bounds
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0
        
        children = [
            QuadNode((min_lon, min_lat, center_lon, center_lat), self.depth + 1, self.importance),
            QuadNode((center_lon, min_lat, max_lon, center_lat), self.depth + 1, self.importance),
            QuadNode((min_lon, center_lat, center_lon, max_lat), self.depth + 1, self.importance),
            QuadNode((center_lon, center_lat, max_lon, max_lat), self.depth + 1, self.importance),
        ]
        return children


class GraphService:
    """Сервис для генерации различных типов графов для планирования маршрутов дронов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_grid_graph(
        self,
        bounds: Tuple[float, float, float, float],
        grid_spacing: float = 0.001,
        buildings: Optional[gpd.GeoDataFrame] = None,
        no_fly_zones: Optional[Union[gpd.GeoDataFrame, List]] = None,
        connect_diagonal: bool = True,
        min_clearance: float = 0.0001,
        use_adaptive: bool = True,
        city_boundary: Optional[Polygon] = None
    ) -> nx.Graph:
        """
        Создает сеточный граф для свободного пространства
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            grid_spacing: расстояние между узлами сетки в градусах (~0.001° ≈ 100м)
            buildings: GeoDataFrame со зданиями
            no_fly_zones: Список запретных зон
            connect_diagonal: соединять ли диагональные узлы (8-связность vs 4-связность)
            min_clearance: минимальное расстояние от препятствий
            use_adaptive: использовать адаптивную сетку (рекомендуется)
            
        Returns:
            NetworkX граф с узлами в свободном пространстве
        """
        if use_adaptive:
            # Используем адаптивную сетку (оптимизированную)
            return self.create_adaptive_grid_graph(
                bounds=bounds,
                buildings=buildings,
                no_fly_zones=no_fly_zones,
                connect_diagonal=connect_diagonal,
                min_clearance=min_clearance,
                base_spacing=grid_spacing,
                min_spacing=grid_spacing * 0.5,  # 50% от базового для важных зон (увеличено для ускорения)
                max_spacing=grid_spacing * 2.5,  # 250% от базового для неважных зон (увеличено для ускорения)
                max_depth=4,  # уменьшено для ускорения
                city_boundary=city_boundary
            )
        
        # Старый метод с фиксированной сеткой (для совместимости)
        min_lon, min_lat, max_lon, max_lat = bounds
        
        self.logger.info(f"Генерация сеточного графа (фиксированная): bounds={bounds}, spacing={grid_spacing}")
        
        # Генерируем сетку точек
        lons = np.arange(min_lon, max_lon, grid_spacing)
        lats = np.arange(min_lat, max_lat, grid_spacing)
        
        self.logger.info(f"Размер сетки: {len(lats)} x {len(lons)} = {len(lats) * len(lons)} точек")
        
        # Создаем пространственный индекс для препятствий
        spatial_index = self._create_spatial_index(buildings, no_fly_zones, min_clearance)
        
        # Создаем все точки сетки векторизованно
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        all_points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
        
        self.logger.info(f"Фильтрация точек через пространственный индекс...")
        
        # Фильтруем точки используя пространственный индекс и границы города
        valid_indices = []
        for idx, (lon, lat) in enumerate(all_points):
            point = Point(lon, lat)
            
            # СТРОГАЯ проверка: точка должна быть ВНУТРИ границ города
            if city_boundary is not None:
                try:
                    # Проверяем вхождение точки в границы
                    if not city_boundary.contains(point):
                        if not city_boundary.touches(point):
                            attempts += 1
                            continue  # Точка вне границ - пропускаем
                except Exception:
                    attempts += 1
                    continue
            
            # Проверяем препятствия
            if spatial_index is not None:
                if not self._point_intersects_index(point, spatial_index):
                    valid_indices.append(idx)
            else:
                valid_indices.append(idx)
        
        self.logger.info(f"Найдено {len(valid_indices)} валидных точек из {len(all_points)}")
        
        # Создаем граф только с валидными узлами
        G = nx.Graph()
        index_to_node = {}  # индекс в all_points -> node_id
        node_to_coords = {}  # node_id -> (lat, lon)
        
        for node_id, idx in enumerate(valid_indices):
            lon, lat = all_points[idx]
            G.add_node(node_id, lat=lat, lon=lon, pos=(lon, lat))
            index_to_node[idx] = node_id
            node_to_coords[node_id] = (lat, lon)
        
        self.logger.info(f"Создано {len(G.nodes)} узлов")
        self.logger.info(f"Создание рёбер...")
        
        # Создаем lookup таблицу для быстрого поиска соседей
        n_lons = len(lons)
        n_lats = len(lats)
        
        edges_added = 0
        
        # Соединяем соседние узлы более эффективно
        for idx in valid_indices:
            if idx not in index_to_node:
                continue
            
            node_id = index_to_node[idx]
            i = idx // n_lons  # индекс по lat
            j = idx % n_lons   # индекс по lon
            
            # Определяем соседей по индексам
            neighbor_offsets = [
                (i + 1, j),      # север
                (i, j + 1),      # восток
            ]
            
            if connect_diagonal:
                neighbor_offsets.extend([
                    (i + 1, j + 1),  # северо-восток
                    (i + 1, j - 1),  # северо-запад
                ])
            
            for ni, nj in neighbor_offsets:
                if 0 <= ni < n_lats and 0 <= nj < n_lons:
                    neighbor_idx = ni * n_lons + nj
                    
                    if neighbor_idx in index_to_node:
                        neighbor_node = index_to_node[neighbor_idx]
                        
                        if not G.has_edge(node_id, neighbor_node):
                            lat1, lon1 = node_to_coords[node_id]
                            lat2, lon2 = node_to_coords[neighbor_node]
                            
                            # Упрощенная проверка пересечений только для диагоналей
                            if connect_diagonal and abs(ni - i) == 1 and abs(nj - j) == 1:
                                line = LineString([(lon1, lat1), (lon2, lat2)])
                                if spatial_index and self._line_intersects_index(line, spatial_index):
                                    continue
                            
                            distance = self._calculate_distance(lat1, lon1, lat2, lon2)
                            G.add_edge(node_id, neighbor_node, weight=distance, length=distance)
                            edges_added += 1
        
        self.logger.info(f"Добавлено {edges_added} рёбер в граф")
        
        return G
    
    def create_delaunay_graph(
        self,
        bounds: Tuple[float, float, float, float],
        num_points: int = 500,
        buildings: Optional[gpd.GeoDataFrame] = None,
        no_fly_zones: Optional[Union[gpd.GeoDataFrame, List]] = None,
        min_clearance: float = 0.0001,
        city_boundary: Optional[Polygon] = None
    ) -> nx.Graph:
        """
        Создает граф на основе триангуляции Делоне
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            num_points: количество случайных точек
            buildings: GeoDataFrame со зданиями
            no_fly_zones: Список запретных зон
            min_clearance: минимальное расстояние от препятствий
            
        Returns:
            NetworkX граф
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        self.logger.info(f"Генерация Delaunay графа: {num_points} точек")
        
        # Подготовка препятствий
        obstacles = self._prepare_obstacles(buildings, no_fly_zones, min_clearance)
        
        # Генерируем случайные точки в свободном пространстве
        points = []
        attempts = 0
        max_attempts = num_points * 10
        
        while len(points) < num_points and attempts < max_attempts:
            lon = np.random.uniform(min_lon, max_lon)
            lat = np.random.uniform(min_lat, max_lat)
            point = Point(lon, lat)
            
            # СТРОГАЯ проверка: точка должна быть ВНУТРИ границ города
            if city_boundary is not None:
                if not city_boundary.contains(point) and not city_boundary.touches(point):
                    attempts += 1
                    continue
            
            if not self._is_point_in_obstacles(point, obstacles):
                points.append([lon, lat])
            
            attempts += 1
        
        if len(points) < 3:
            self.logger.warning("Недостаточно точек для триангуляции")
            return nx.Graph()
        
        points = np.array(points)
        
        # Строим триангуляцию Делоне
        tri = Delaunay(points)
        
        # Создаем граф из триангуляции
        G = nx.Graph()
        
        # Добавляем узлы
        for i, (lon, lat) in enumerate(points):
            G.add_node(i, lat=lat, lon=lon, pos=(lon, lat))
        
        # Добавляем рёбра из треугольников
        for simplex in tri.simplices:
            for i in range(3):
                n1, n2 = simplex[i], simplex[(i + 1) % 3]
                if not G.has_edge(n1, n2):
                    p1, p2 = points[n1], points[n2]
                    line = LineString([p1, p2])
                    
                    # Проверяем, что ребро не пересекает препятствия
                    if not self._line_intersects_obstacles(line, obstacles):
                        distance = self._calculate_distance(p1[1], p1[0], p2[1], p2[0])
                        G.add_edge(n1, n2, weight=distance, length=distance)
        
        self.logger.info(f"Delaunay граф: {len(G.nodes)} узлов, {len(G.edges)} рёбер")
        
        return G
    
    def merge_graphs(self, road_graph: nx.Graph, free_space_graph: nx.Graph, 
                     max_connection_distance: float = 0.001) -> nx.Graph:
        """
        Объединяет дорожный граф и граф свободного пространства
        
        Args:
            road_graph: граф дорог из OSM
            free_space_graph: граф свободного пространства
            max_connection_distance: максимальное расстояние для соединения графов
            
        Returns:
            Объединенный граф
        """
        self.logger.info("Объединение дорожного и свободного графов")
        
        # Создаем новый граф
        merged = nx.Graph()
        
        # Добавляем узлы из дорожного графа с префиксом 'road_'
        for node, data in road_graph.nodes(data=True):
            merged.add_node(f"road_{node}", **data, graph_type='road')
        
        # Добавляем рёбра из дорожного графа
        for u, v, data in road_graph.edges(data=True):
            merged.add_edge(f"road_{u}", f"road_{v}", **data, edge_type='road')
        
        # Добавляем узлы из графа свободного пространства с префиксом 'free_'
        for node, data in free_space_graph.nodes(data=True):
            merged.add_node(f"free_{node}", **data, graph_type='free')
        
        # Добавляем рёбра из графа свободного пространства
        for u, v, data in free_space_graph.edges(data=True):
            merged.add_edge(f"free_{u}", f"free_{v}", **data, edge_type='free')
        
        # Соединяем близкие узлы между графами
        connections = 0
        for road_node, road_data in road_graph.nodes(data=True):
            if 'y' not in road_data or 'x' not in road_data:
                continue
            
            road_lat, road_lon = road_data['y'], road_data['x']
            
            for free_node, free_data in free_space_graph.nodes(data=True):
                if 'lat' not in free_data or 'lon' not in free_data:
                    continue
                
                free_lat, free_lon = free_data['lat'], free_data['lon']
                
                distance = self._calculate_distance(road_lat, road_lon, free_lat, free_lon)
                
                if distance <= max_connection_distance:
                    merged.add_edge(
                        f"road_{road_node}",
                        f"free_{free_node}",
                        weight=distance,
                        length=distance,
                        edge_type='connection'
                    )
                    connections += 1
        
        self.logger.info(f"Создано {connections} соединений между графами")
        self.logger.info(f"Объединенный граф: {len(merged.nodes)} узлов, {len(merged.edges)} рёбер")
        
        return merged
    
    def _create_spatial_index(
        self,
        buildings: Optional[gpd.GeoDataFrame],
        no_fly_zones: Optional[Union[gpd.GeoDataFrame, List]],
        buffer_distance: float
    ) -> Optional[STRtree]:
        """Создает пространственный индекс для быстрой проверки препятствий"""
        geometries = []
        
        if buildings is not None and len(buildings) > 0:
            try:
                # Ограничиваем количество зданий для ускорения
                max_buildings = 5000
                buildings_sample = buildings.head(max_buildings) if len(buildings) > max_buildings else buildings
                
                for geom in buildings_sample.geometry:
                    if geom is not None and geom.is_valid:
                        buffered = geom.buffer(buffer_distance)
                        geometries.append(buffered)
                
                self.logger.info(f"Добавлено {len(geometries)} зданий в пространственный индекс")
            except Exception as e:
                self.logger.warning(f"Ошибка обработки зданий: {e}")
        
        # Проверяем no_fly_zones правильно (GeoDataFrame нельзя использовать в if напрямую)
        if no_fly_zones is not None:
            # Поддержка GeoDataFrame и списка геометрий
            if isinstance(no_fly_zones, gpd.GeoDataFrame):
                # Проверяем, не пустой ли GeoDataFrame
                if len(no_fly_zones) > 0:
                    # Если это GeoDataFrame, обрабатываем каждую строку
                    for idx, row in no_fly_zones.iterrows():
                        if row.geometry is not None and row.geometry.is_valid:
                            # Зоны уже имеют буфер, добавляем дополнительный для безопасности
                            geometries.append(row.geometry.buffer(buffer_distance))
                    self.logger.info(f"Добавлено {len(no_fly_zones)} беспилотных зон из GeoDataFrame")
            elif isinstance(no_fly_zones, list):
                # Если это список геометрий или объектов с geometry
                for zone in no_fly_zones:
                    if hasattr(zone, 'geometry') and zone.geometry is not None:
                        geometries.append(zone.geometry.buffer(buffer_distance))
                    elif hasattr(zone, '__geo_interface__'):
                        # Если это shapely геометрия
                        from shapely.geometry import shape
                        geom = shape(zone)
                        if geom.is_valid:
                            geometries.append(geom.buffer(buffer_distance))
                    elif hasattr(zone, 'is_valid'):
                        # Если это прямая shapely геометрия
                        if zone.is_valid:
                            geometries.append(zone.buffer(buffer_distance))
                if len(no_fly_zones) > 0:
                    self.logger.info(f"Добавлено беспилотных зон из списка")
        
        if not geometries:
            return None
        
        try:
            return STRtree(geometries)
        except Exception as e:
            self.logger.warning(f"Ошибка создания пространственного индекса: {e}")
            return None
    
    def _point_intersects_index(self, point: Point, spatial_index: STRtree) -> bool:
        """Проверяет пересечение точки с препятствиями через пространственный индекс"""
        if spatial_index is None:
            return False
        
        try:
            # Быстрый поиск потенциальных пересечений
            potential_matches = spatial_index.query(point)
            
            # Точная проверка пересечений
            for geom in potential_matches:
                if geom.contains(point) or geom.intersects(point):
                    return True
            
            return False
        except Exception:
            return False
    
    def _line_intersects_index(self, line: LineString, spatial_index: STRtree) -> bool:
        """Проверяет пересечение линии с препятствиями через пространственный индекс"""
        if spatial_index is None:
            return False
        
        try:
            potential_matches = spatial_index.query(line)
            
            for geom in potential_matches:
                if geom.intersects(line):
                    return True
            
            return False
        except Exception:
            return False
    
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
    
    def _find_node_at_position(
        self,
        node_positions: dict,
        target_pos: Tuple[float, float],
        tolerance: float = 1e-9
    ) -> Optional[int]:
        """Находит узел в указанной позиции"""
        for node_id, (lat, lon) in node_positions.items():
            if abs(lat - target_pos[0]) < tolerance and abs(lon - target_pos[1]) < tolerance:
                return node_id
        return None
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Вычисляет расстояние между двумя точками по формуле гаверсинуса (в км)
        """
        R = 6371.0  # Радиус Земли в км
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def _calculate_zone_importance(
        self,
        bounds: Tuple[float, float, float, float],
        center: Tuple[float, float],
        buildings_index: Optional[STRtree],
        min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> float:
        """
        УПРОЩЕННОЕ вычисление важности зоны (0-1) для ускорения
        
        Returns:
            Важность от 0 (неважно) до 1 (очень важно)
        """
        min_lon_zone, min_lat_zone, max_lon_zone, max_lat_zone = bounds
        center_lon, center_lat = center
        
        # 1. Близость к центру города (быстрое вычисление)
        zone_center_lon = (min_lon_zone + max_lon_zone) / 2.0
        zone_center_lat = (min_lat_zone + max_lat_zone) / 2.0
        
        city_width = max_lon - min_lon
        city_height = max_lat - min_lat
        max_dim = max(city_width, city_height)
        
        # Нормализованное расстояние до центра
        dist_lon = abs(zone_center_lon - center_lon) / max_dim if max_dim > 0 else 1.0
        dist_lat = abs(zone_center_lat - center_lat) / max_dim if max_dim > 0 else 1.0
        dist_to_center = np.sqrt(dist_lon**2 + dist_lat**2)
        center_importance = max(0, 1.0 - dist_to_center * 2.0) * 0.6  # 60% веса
        
        # 2. Плотность зданий (быстрая проверка)
        building_density = 0.0
        if buildings_index is not None:
            try:
                zone_box = box(min_lon_zone, min_lat_zone, max_lon_zone, max_lat_zone)
                intersecting_buildings = buildings_index.query(zone_box)
                # Простая нормализация: есть здания = важно
                if len(intersecting_buildings) > 0:
                    building_density = min(len(intersecting_buildings) / 5.0, 1.0) * 0.4  # 40% веса
            except Exception:
                pass
        
        total_importance = center_importance + building_density
        return min(total_importance, 1.0)
    
    def _build_quadtree(
        self,
        bounds: Tuple[float, float, float, float],
        center: Tuple[float, float],
        buildings_index: Optional[STRtree],
        min_lon: float, min_lat: float, max_lon: float, max_lat: float,
        max_depth: int = 4,  # Уменьшено с 6 до 4 для ускорения
        min_importance: float = 0.2,  # Увеличено для меньшего разбиения
        min_size: float = 0.001,  # Увеличено для меньшего разбиения
        city_boundary: Optional[Polygon] = None
    ) -> QuadNode:
        """
        Строит квадродерево для адаптивного разбиения пространства (ОПТИМИЗИРОВАНО)
        """
        root = QuadNode(bounds, 0, 1.0)
        
        def subdivide_recursive(node: QuadNode):
            if not node.should_subdivide(max_depth, min_importance):
                return
            
            min_lon_zone, min_lat_zone, max_lon_zone, max_lat_zone = node.bounds
            
            # Проверяем минимальный размер
            width = max_lon_zone - min_lon_zone
            height = max_lat_zone - min_lat_zone
            if width < min_size or height < min_size:
                return
            
            # Проверяем, пересекается ли зона с границами города
            if city_boundary is not None:
                try:
                    zone_box = box(min_lon_zone, min_lat_zone, max_lon_zone, max_lat_zone)
                    if not city_boundary.intersects(zone_box):
                        # Зона полностью вне границ города - не разбиваем
                        return
                except Exception:
                    # Если ошибка, продолжаем разбиение
                    pass
            
            # Вычисляем важность зоны (упрощенная версия)
            node.importance = self._calculate_zone_importance(
                node.bounds, center, buildings_index,
                min_lon, min_lat, max_lon, max_lat
            )
            
            # Если важность достаточна, разбиваем дальше
            if node.importance >= min_importance:
                children = node.subdivide()
                node.children = children
                
                for child in children:
                    subdivide_recursive(child)
        
        subdivide_recursive(root)
        return root
    
    def _generate_adaptive_points(
        self,
        node: QuadNode,
        spatial_index: Optional[STRtree],
        base_spacing: float,
        min_spacing: float,
        max_spacing: float,
        city_boundary: Optional[Polygon] = None
    ) -> List[Tuple[float, float]]:
        """
        Генерирует точки в узле квадродерева с адаптивным шагом (ОПТИМИЗИРОВАНО)
        """
        points = []
        min_lon, min_lat, max_lon, max_lat = node.bounds
        
        # Адаптивный шаг: чем важнее зона, тем меньше шаг
        spacing = max_spacing - (max_spacing - min_spacing) * node.importance
        
        # Если это листовой узел, генерируем точки
        if node.children is None:
            lons = np.arange(min_lon, max_lon, spacing)
            lats = np.arange(min_lat, max_lat, spacing)
            
            # Батчинг проверок препятствий для ускорения
            if spatial_index is not None:
                # Создаем все точки сразу
                lon_mesh, lat_mesh = np.meshgrid(lons, lats)
                all_candidates = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
                
                # Фильтруем через пространственный индекс батчами
                for lon, lat in all_candidates:
                    point = Point(lon, lat)
                    
                    # СТРОГАЯ проверка: точка должна быть ВНУТРИ границ города
                    if city_boundary is not None:
                        # Проверяем вхождение точки в границы (работает для Polygon и MultiPolygon)
                        try:
                            if not city_boundary.contains(point):
                                # Если точка не строго внутри, проверяем, не на границе ли она
                                if not city_boundary.touches(point):
                                    continue  # Точка вне границ - пропускаем
                        except Exception as e:
                            # Если ошибка проверки, пропускаем точку для безопасности
                            self.logger.warning(f"Ошибка проверки границ для точки {point}: {e}")
                            continue
                    
                    if not self._point_intersects_index(point, spatial_index):
                        points.append((lon, lat))
            else:
                # Нет препятствий - добавляем только точки внутри границ города
                for lat in lats:
                    for lon in lons:
                        point = Point(lon, lat)
                        # СТРОГАЯ проверка: точка должна быть ВНУТРИ границ города
                        if city_boundary is None:
                            points.append((lon, lat))
                        else:
                            try:
                                if city_boundary.contains(point) or city_boundary.touches(point):
                                    points.append((lon, lat))
                            except Exception:
                                # Если ошибка проверки, пропускаем точку
                                pass
        else:
            # Рекурсивно генерируем точки в дочерних узлах
            for child in node.children:
                child_points = self._generate_adaptive_points(
                    child, spatial_index, base_spacing, min_spacing, max_spacing, city_boundary
                )
                points.extend(child_points)
        
        return points
    
    def create_adaptive_grid_graph(
        self,
        bounds: Tuple[float, float, float, float],
        buildings: Optional[gpd.GeoDataFrame] = None,
        no_fly_zones: Optional[Union[gpd.GeoDataFrame, List]] = None,
        connect_diagonal: bool = True,
        min_clearance: float = 0.0001,
        base_spacing: float = 0.001,
        min_spacing: float = 0.0005,  # увеличен для ускорения (~50м)
        max_spacing: float = 0.0025,  # увеличен для ускорения (~250м)
        max_depth: int = 4,  # уменьшено для ускорения
        city_boundary: Optional[Polygon] = None
    ) -> nx.Graph:
        """
        Создает адаптивный сеточный граф с использованием квадродерева
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            buildings: GeoDataFrame со зданиями
            no_fly_zones: Список запретных зон
            connect_diagonal: соединять ли диагональные узлы
            min_clearance: минимальное расстояние от препятствий
            base_spacing: базовый шаг сетки
            min_spacing: минимальный шаг для важных зон
            max_spacing: максимальный шаг для неважных зон
            max_depth: максимальная глубина квадродерева
            
        Returns:
            NetworkX граф с адаптивными узлами
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0
        
        self.logger.info(f"Генерация адаптивного сеточного графа: bounds={bounds}")
        if city_boundary is not None:
            self.logger.info(f"Границы города: {city_boundary.geom_type}, площадь: {city_boundary.area:.6f}")
        else:
            self.logger.warning("Границы города не заданы! Граф будет строиться в bounding box")
        
        # Создаем пространственный индекс
        spatial_index = self._create_spatial_index(buildings, no_fly_zones, min_clearance)
        
        # Создаем индекс зданий для вычисления важности (упрощенный)
        buildings_index = None
        if buildings is not None and len(buildings) > 0:
            try:
                geometries = []
                # Уменьшаем количество зданий для ускорения
                max_buildings = 2000
                buildings_sample = buildings.head(max_buildings) if len(buildings) > max_buildings else buildings
                for geom in buildings_sample.geometry:
                    if geom is not None and geom.is_valid:
                        geometries.append(geom)
                if geometries:
                    buildings_index = STRtree(geometries)
            except Exception as e:
                self.logger.warning(f"Ошибка создания индекса зданий: {e}")
        
        # Строим квадродерево (упрощенное)
        self.logger.info("Построение квадродерева...")
        quadtree = self._build_quadtree(
            bounds, (center_lon, center_lat), buildings_index,
            min_lon, min_lat, max_lon, max_lat, max_depth,
            city_boundary=city_boundary
        )
        
        # Генерируем адаптивные точки
        self.logger.info("Генерация адаптивных точек...")
        all_points = self._generate_adaptive_points(
            quadtree, spatial_index, base_spacing, min_spacing, max_spacing, city_boundary
        )
        
        self.logger.info(f"Сгенерировано {len(all_points)} адаптивных точек")
        
        # Дополнительная проверка: фильтруем точки по границам города еще раз (на всякий случай)
        if city_boundary is not None:
            filtered_points = []
            for lon, lat in all_points:
                point = Point(lon, lat)
                try:
                    if city_boundary.contains(point) or city_boundary.touches(point):
                        filtered_points.append((lon, lat))
                except Exception:
                    pass  # Пропускаем точку при ошибке
            if len(filtered_points) < len(all_points):
                self.logger.info(f"Дополнительная фильтрация: {len(all_points)} -> {len(filtered_points)} точек")
                all_points = filtered_points
        
        # Создаем граф
        G = nx.Graph()
        point_to_node = {}  # (lon, lat) -> node_id
        
        for node_id, (lon, lat) in enumerate(all_points):
            G.add_node(node_id, lat=lat, lon=lon, pos=(lon, lat))
            point_to_node[(lon, lat)] = node_id
        
        self.logger.info(f"Создано {len(G.nodes)} узлов")
        self.logger.info("Создание рёбер...")
        
        # Соединяем соседние узлы (ОПТИМИЗИРОВАНО - только ближайшие соседи)
        edges_added = 0
        max_connection_dist = max_spacing * 1.5  # уменьшено для ускорения
        
        # Создаем пространственный индекс узлов для быстрого поиска соседей
        if len(all_points) > 0:
            coords_array = np.array([(lon, lat) for lon, lat in all_points])
            kdtree = cKDTree(coords_array)
            
            # Ищем только k ближайших соседей для каждого узла (быстрее чем query_ball_point)
            k_neighbors = 8 if connect_diagonal else 4
            
            for node_id, (lon1, lat1) in enumerate(all_points):
                # Ищем k ближайших соседей
                distances, indices = kdtree.query((lon1, lat1), k=min(k_neighbors + 1, len(all_points)))
                
                for dist, neighbor_idx in zip(distances, indices):
                    if neighbor_idx == node_id or neighbor_idx >= len(all_points):
                        continue
                    
                    # Проверяем максимальное расстояние
                    if dist > max_connection_dist:
                        continue
                    
                    lon2, lat2 = all_points[neighbor_idx]
                    
                    # Для диагоналей проверяем пересечения (только если нужно)
                    if connect_diagonal and dist > min_spacing * 0.7:
                        dist_lon = abs(lon2 - lon1)
                        dist_lat = abs(lat2 - lat1)
                        if dist_lon > min_spacing * 0.5 and dist_lat > min_spacing * 0.5:
                            line = LineString([(lon1, lat1), (lon2, lat2)])
                            if spatial_index and self._line_intersects_index(line, spatial_index):
                                continue
                    
                    # Вычисляем реальное расстояние
                    distance = self._calculate_distance(lat1, lon1, lat2, lon2)
                    if not G.has_edge(node_id, neighbor_idx):
                        G.add_edge(node_id, neighbor_idx, weight=distance, length=distance)
                        edges_added += 1
        
        self.logger.info(f"Добавлено {edges_added} рёбер в граф")
        
        return G
    
    def graph_to_geojson(self, graph: nx.Graph, edge_type_filter: Optional[str] = None) -> dict:
        """
        Конвертирует NetworkX граф в GeoJSON для визуализации
        
        Args:
            graph: NetworkX граф
            edge_type_filter: фильтр по типу рёбер ('road', 'free', 'connection', None для всех)
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        for u, v, data in graph.edges(data=True):
            # Фильтрация по типу
            if edge_type_filter and data.get('edge_type') != edge_type_filter:
                continue
            
            # Получаем координаты узлов
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            
            # Определяем координаты в зависимости от типа узла
            if 'y' in u_data and 'x' in u_data:  # OSM узел
                u_lon, u_lat = u_data['x'], u_data['y']
            elif 'lon' in u_data and 'lat' in u_data:  # Наш узел
                u_lon, u_lat = u_data['lon'], u_data['lat']
            else:
                continue
            
            if 'y' in v_data and 'x' in v_data:
                v_lon, v_lat = v_data['x'], v_data['y']
            elif 'lon' in v_data and 'lat' in v_data:
                v_lon, v_lat = v_data['lon'], v_data['lat']
            else:
                continue
            
            feature = {
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
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
