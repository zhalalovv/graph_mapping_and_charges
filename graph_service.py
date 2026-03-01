import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
import geopandas as gpd
from scipy.spatial import Delaunay, cKDTree
from shapely.strtree import STRtree
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
                    # Рёбра над беспилотными зонами не добавляем — они не нужны
                    if not self._line_intersects_obstacles(line, obstacles):
                        distance = self._calculate_distance(p1[1], p1[0], p2[1], p2[0])
                        G.add_edge(n1, n2, weight=distance, length=distance)
        
        G = self._remove_nodes_in_obstacles(G, obstacles)
        G = self._remove_edges_over_obstacles(G, obstacles)
        
        self.logger.info(f"Delaunay граф: {len(G.nodes)} узлов, {len(G.edges)} рёбер")
        
        return G
    
    def _get_node_coords(self, G: nx.Graph, node_id) -> Optional[Tuple[float, float]]:
        """Возвращает (lon, lat) для узла или None."""
        d = G.nodes.get(node_id, {})
        if "x" in d and "y" in d:
            return (d["x"], d["y"])
        if "lon" in d and "lat" in d:
            return (d["lon"], d["lat"])
        pos = d.get("pos")
        if pos is not None and len(pos) >= 2:
            return (float(pos[0]), float(pos[1]))
        return None
    
    def _remove_nodes_in_obstacles(
        self, G: nx.Graph, obstacles: Optional[Polygon]
    ) -> nx.Graph:
        """Удаляет узлы, находящиеся внутри препятствий (и все их рёбра)."""
        if obstacles is None:
            return G
        to_remove = []
        for n in list(G.nodes()):
            coords = self._get_node_coords(G, n)
            if coords is None:
                continue
            pt = Point(coords[0], coords[1])
            if self._is_point_in_obstacles(pt, obstacles):
                to_remove.append(n)
        for n in to_remove:
            G.remove_node(n)
        if to_remove:
            self.logger.info(f"Удалено {len(to_remove)} узлов внутри no_fly_zones")
        return G
    
    def _remove_edges_over_obstacles(
        self,
        G: nx.Graph,
        obstacles: Optional[Polygon],
        edge_types: Optional[List[str]] = None,
    ) -> nx.Graph:
        """
        Удаляет все рёбра, проходящие над беспилотной зоной. Такие рёбра удаляются полностью —
        они не нужны и не используются.
        
        Args:
            edge_types: если задан — проверять только рёбра с edge_type из списка ('free', 'connection').
                       Если None — проверять все рёбра.
        """
        if obstacles is None:
            return G
        to_remove = []
        for u, v, data in list(G.edges(data=True)):
            if edge_types is not None:
                if data.get("edge_type") not in edge_types:
                    continue
            coords_u = self._get_node_coords(G, u)
            coords_v = self._get_node_coords(G, v)
            if coords_u is None or coords_v is None:
                continue
            line = LineString([coords_u, coords_v])
            if self._line_intersects_obstacles(line, obstacles):
                to_remove.append((u, v))
        for u, v in to_remove:
            G.remove_edge(u, v)
        if to_remove:
            self.logger.info(f"Удалено {len(to_remove)} рёбер над no_fly_zones")
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

    def poisson_disk_sample_in_polygon(
        self,
        polygon,
        min_distance_deg: float,
        max_points: int = 3000,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Tuple[float, float]]:
        """
        Poisson-disk sampling внутри полигона (алгоритм Бридсона).
        Возвращает список (lon, lat) в градусах.

        Args:
            polygon: Shapely Polygon или MultiPolygon
            min_distance_deg: минимальное расстояние между точками в градусах (~0.0003 ≈ 30–35 м)
            max_points: максимальное количество точек
            rng: опциональный генератор случайных чисел
        """
        if polygon is None or polygon.is_empty or not polygon.is_valid:
            return []
        rng = rng or np.random.default_rng()
        minx, miny, maxx, maxy = polygon.bounds
        try:
            prep_poly = prep(polygon)
        except Exception:
            prep_poly = polygon

        def contains(pt):
            try:
                return prep_poly.contains(pt)
            except Exception:
                return polygon.contains(pt) if hasattr(polygon, "contains") else False

        r = min_distance_deg
        cell_size = r / np.sqrt(2)
        cols = int(np.ceil((maxx - minx) / cell_size)) or 1
        rows = int(np.ceil((maxy - miny) / cell_size)) or 1
        grid = np.full((rows, cols), -1, dtype=np.int32)
        points = []
        active: List[int] = []

        def grid_cell(lon, lat):
            c = int((lon - minx) / cell_size)
            rw = int((lat - miny) / cell_size)
            c = max(0, min(c, cols - 1))
            rw = max(0, min(rw, rows - 1))
            return rw, c

        def valid_point(lon, lat):
            pt = Point(lon, lat)
            if not contains(pt):
                return False
            rr, cc = grid_cell(lon, lat)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < rows and 0 <= c2 < cols:
                        idx = grid[r2, c2]
                        if idx >= 0:
                            px, py = points[idx][0], points[idx][1]
                            if np.hypot(px - lon, py - lat) < r:
                                return False
            return True

        candidates = []
        for _ in range(30):
            lon = minx + rng.random() * (maxx - minx)
            lat = miny + rng.random() * (maxy - miny)
            if contains(Point(lon, lat)):
                candidates.append((lon, lat))
                break
        if not candidates:
            return []
        lon, lat = candidates[0]
        points.append((lon, lat))
        idx0 = len(points) - 1
        rr, cc = grid_cell(lon, lat)
        grid[rr, cc] = idx0
        active.append(idx0)

        k = 30
        while active and len(points) < max_points:
            i = rng.integers(0, len(active))
            idx = active[i]
            px, py = points[idx][0], points[idx][1]
            found = False
            for _ in range(k):
                angle = rng.random() * 2 * np.pi
                rad = r + rng.random() * r
                nlon = px + rad * np.cos(angle)
                nlat = py + rad * np.sin(angle)
                if minx <= nlon <= maxx and miny <= nlat <= maxy and valid_point(nlon, nlat):
                    points.append((nlon, nlat))
                    new_idx = len(points) - 1
                    rr, cc = grid_cell(nlon, nlat)
                    grid[rr, cc] = new_idx
                    active.append(new_idx)
                    found = True
                    break
            if not found:
                active[i] = active[-1]
                active.pop()

        self.logger.info(f"Poisson-disk: {len(points)} точек в свободном пространстве (min_dist={min_distance_deg:.6f}°)")
        return points

    def get_flyable_points(
        self,
        city_boundary,
        buildings: Optional[gpd.GeoDataFrame] = None,
        no_fly_zones: Optional[Union[gpd.GeoDataFrame, List]] = None,
        building_buffer_deg: float = 0.00025,
        min_distance_deg: float = 0.0003,
        max_points: int = 3000,
    ) -> List[Tuple[float, float]]:
        """
        Точки для полёта в свободном пространстве (леса, поля, парки) — Poisson-disk sampling.
        Свободная зона = city_boundary - здания (с буфером) - no_fly_zones.

        Returns:
            Список (lon, lat) в градусах.
        """
        if city_boundary is None or (not hasattr(city_boundary, "is_valid") or not city_boundary.is_valid):
            return []
        try:
            if city_boundary.geom_type == "MultiPolygon":
                free_area = city_boundary
            else:
                free_area = city_boundary
            geometries = []
            if buildings is not None and len(buildings) > 0:
                for geom in buildings.geometry:
                    if geom is not None and geom.is_valid:
                        geometries.append(geom.buffer(building_buffer_deg))
            if no_fly_zones is not None:
                if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                    for _, row in no_fly_zones.iterrows():
                        if row.geometry is not None and row.geometry.is_valid:
                            geometries.append(row.geometry.buffer(building_buffer_deg))
                elif isinstance(no_fly_zones, list):
                    for zone in no_fly_zones:
                        g = getattr(zone, "geometry", zone)
                        if g is not None and getattr(g, "is_valid", True):
                            geometries.append(g.buffer(building_buffer_deg) if hasattr(g, "buffer") else g)
            if geometries:
                obstacles_union = unary_union(geometries)
                if obstacles_union is not None and not obstacles_union.is_empty:
                    free_area = free_area.difference(obstacles_union)
            if free_area is None or free_area.is_empty:
                return []
            if free_area.geom_type == "MultiPolygon":
                all_points = []
                for poly in free_area.geoms:
                    if poly.is_valid and not poly.is_empty and poly.area > 1e-12:
                        pts = self.poisson_disk_sample_in_polygon(
                            poly, min_distance_deg, max_points=max_points // max(1, len(free_area.geoms))
                        )
                        all_points.extend(pts)
                return all_points[:max_points]
            return self.poisson_disk_sample_in_polygon(free_area, min_distance_deg, max_points)
        except Exception as e:
            self.logger.warning(f"Ошибка get_flyable_points: {e}")
            return []

    def merge_points_min_distance(
        self,
        points_a: List[Tuple[float, float]],
        points_b: List[Tuple[float, float]],
        min_distance_deg: float,
    ) -> List[Tuple[float, float]]:
        """Объединяет точки, отбрасывая из points_b те, что ближе min_distance_deg к любой из points_a."""
        if not points_b:
            return list(points_a)
        if not points_a:
            return list(points_b)
        arr_a = np.array(points_a)
        tree = cKDTree(arr_a)
        result = list(points_a)
        for lon, lat in points_b:
            d, _ = tree.query([lon, lat], k=1)
            if d >= min_distance_deg:
                result.append((lon, lat))
                arr_a = np.vstack([arr_a, [[lon, lat]]])
                tree = cKDTree(arr_a)
        return result

    def merge_road_and_delaunay(
        self,
        road_graph: nx.Graph,
        delaunay_graph: nx.Graph,
        max_connection_distance_deg: float = 0.0003,
        obstacles: Optional[Polygon] = None,
    ) -> nx.Graph:
        """
        Объединяет дорожный граф (OSM) и граф Delaunay в один граф, добавляя рёбра-соединения
        между узлами Delaunay и ближайшими узлами дорог в пределах max_connection_distance_deg.
        
        У узлов дорог сохраняются атрибуты x, y (lon, lat); у узлов Delaunay — lon, lat.
        Рёбра помечаются edge_type: 'road', 'free', 'connection'.
        
        Args:
            road_graph: граф дорог (OSM), узлы с x, y
            delaunay_graph: граф Delaunay, узлы с lon, lat
            max_connection_distance_deg: макс. расстояние в градусах для соединения узла Delaunay с дорогой
            obstacles: препятствия (no_fly_zones) — рёбра над ними удаляются полностью
            
        Returns:
            Объединённый неориентированный NetworkX граф
        """
        G = nx.Graph()
        
        # 1. Добавляем все узлы и рёбра дорожного графа
        for n, data in road_graph.nodes(data=True):
            G.add_node(n, **data)
        seen_road_edges = set()
        for u, v, data in road_graph.edges(data=True):
            key = (min(u, v), max(u, v))
            if key in seen_road_edges:
                continue
            seen_road_edges.add(key)
            length = data.get("length") or data.get("weight") or 0
            length = float(length) if length else 0.0
            G.add_edge(u, v, weight=length, length=length, edge_type="road")
        
        # 2. Собираем координаты дорожных узлов для поиска ближайших
        road_coords = []
        road_node_ids = []
        for n, data in road_graph.nodes(data=True):
            if "x" in data and "y" in data:
                road_coords.append((data["x"], data["y"]))
                road_node_ids.append(n)
            elif "lon" in data and "lat" in data:
                road_coords.append((data["lon"], data["lat"]))
                road_node_ids.append(n)
        if not road_coords:
            self.logger.warning("В дорожном графе нет узлов с координатами")
            return G
        
        road_coords_arr = np.array(road_coords)
        road_tree = cKDTree(road_coords_arr)
        
        # 3. Добавляем узлы Delaunay с префиксом id (избегаем пересечения с OSM id)
        d_prefix = "d_"
        delaunay_id_to_merged = {}
        for i, data in delaunay_graph.nodes(data=True):
            merged_id = f"{d_prefix}{i}"
            delaunay_id_to_merged[i] = merged_id
            lon = data.get("lon", data.get("pos", (0, 0))[0])
            lat = data.get("lat", data.get("pos", (0, 0))[1])
            G.add_node(merged_id, lon=lon, lat=lat)
        
        # 4. Добавляем рёбра Delaunay
        for u, v, data in delaunay_graph.edges(data=True):
            mu, mv = delaunay_id_to_merged.get(u, f"{d_prefix}{u}"), delaunay_id_to_merged.get(v, f"{d_prefix}{v}")
            w = data.get("weight") or data.get("length") or 0
            w = float(w) if w else 0.0
            G.add_edge(mu, mv, weight=w, length=w, edge_type="free")
        
        # 4b. Дерево Delaunay-узлов для поиска ближайших к тупикам дорог
        delaunay_coords = []
        delaunay_merged_ids = []
        for n, data in G.nodes(data=True):
            if str(n).startswith(d_prefix):
                lon, lat = data.get("lon"), data.get("lat")
                if lon is not None and lat is not None:
                    delaunay_coords.append((lon, lat))
                    delaunay_merged_ids.append(n)
        delaunay_tree = cKDTree(delaunay_coords) if delaunay_coords else None
        
        # 5. Соединяем каждый узел Delaunay с несколькими ближайшими узлами дороги (больше соединений)
        k_road = min(5, len(road_node_ids))
        for merged_id, data in list(G.nodes(data=True)):
            if not str(merged_id).startswith(d_prefix):
                continue
            lon = data.get("lon")
            lat = data.get("lat")
            if lon is None or lat is None:
                continue
            if k_road == 0:
                break
            dists_deg, indices = road_tree.query([lon, lat], k=k_road)
            dists_deg = np.atleast_1d(dists_deg)
            indices = np.atleast_1d(indices)
            for dist_deg, idx_min in zip(dists_deg, indices):
                if float(dist_deg) > max_connection_distance_deg:
                    continue
                idx_min = int(idx_min)
                road_node = road_node_ids[idx_min]
                road_lon, road_lat = road_coords_arr[idx_min, 0], road_coords_arr[idx_min, 1]
                conn_line = LineString([(lon, lat), (road_lon, road_lat)])
                if obstacles is not None and self._line_intersects_obstacles(conn_line, obstacles):
                    continue
                if obstacles is not None and (
                    self._is_point_in_obstacles(Point(lon, lat), obstacles)
                    or self._is_point_in_obstacles(Point(road_lon, road_lat), obstacles)
                ):
                    continue
                dist_km = self._calculate_distance(lat, lon, road_lat, road_lon)
                G.add_edge(merged_id, road_node, weight=dist_km, length=dist_km, edge_type="connection")
        
        # 5b. Тупики дорог (degree == 1): соединяем с ближайшим узлом Delaunay
        dead_end_distance_deg = max_connection_distance_deg * 2.5  # больший радиус, чтобы тупик точно связался
        road_degree = dict(road_graph.degree())
        for i, road_node in enumerate(road_node_ids):
            if road_degree.get(road_node, 0) != 1:
                continue
            if delaunay_tree is None:
                break
            lon_r, lat_r = road_coords_arr[i, 0], road_coords_arr[i, 1]
            dist_deg, idx_d = delaunay_tree.query([lon_r, lat_r], k=1)
            idx_d = int(np.atleast_1d(idx_d).flat[0])
            if float(dist_deg) > dead_end_distance_deg:
                continue
            merged_d_id = delaunay_merged_ids[idx_d]
            d_lon, d_lat = delaunay_coords[idx_d][0], delaunay_coords[idx_d][1]
            if G.has_edge(merged_d_id, road_node):
                continue
            conn_line = LineString([(lon_r, lat_r), (d_lon, d_lat)])
            if obstacles is not None and self._line_intersects_obstacles(conn_line, obstacles):
                continue
            if obstacles is not None and (
                self._is_point_in_obstacles(Point(lon_r, lat_r), obstacles)
                or self._is_point_in_obstacles(Point(d_lon, d_lat), obstacles)
            ):
                continue
            dist_km = self._calculate_distance(lat_r, lon_r, d_lat, d_lon)
            G.add_edge(merged_d_id, road_node, weight=dist_km, length=dist_km, edge_type="connection")
        
        # 6. Удаляем все рёбра (free, connection), проходящие над беспилотными зонами
        if obstacles is not None:
            G = self._remove_edges_over_obstacles(
                G, obstacles, edge_types=["free", "connection"]
            )
        
        n_road = sum(1 for n in G.nodes() if not str(n).startswith(d_prefix))
        n_free = G.number_of_nodes() - n_road
        n_conn = sum(1 for _u, _v, d in G.edges(data=True) if d.get("edge_type") == "connection")
        self.logger.info(f"Объединённый граф: дорог узлов {n_road}, Delaunay узлов {n_free}, рёбер-соединений {n_conn}")
        return G
    
    def graph_to_geojson(
        self,
        graph: nx.Graph,
        edge_type_filter: Optional[str] = None,
        obstacles: Optional[Polygon] = None,
        include_node_ids: bool = False,
    ) -> dict:
        """
        Конвертирует NetworkX граф в GeoJSON для визуализации.

        Args:
            graph: NetworkX граф
            edge_type_filter: фильтр по типу рёбер ('road', 'free', 'connection', None для всех)
            obstacles: препятствия — рёбра над ними исключаются из вывода
            include_node_ids: если True, в свойства каждого ребра добавляются source_id, target_id,
                             и в ответ добавляется ключ "nodes" для многоуровневого планирования

        Returns:
            GeoJSON FeatureCollection (при include_node_ids=True — с ключом "nodes")
        """
        features = []
        edges = list(graph.edges(data=True))

        if edge_type_filter:
            edges = [(u, v, d) for u, v, d in edges if d.get('edge_type') == edge_type_filter]

        node_coords = {}
        for node_id, node_data in graph.nodes(data=True):
            if 'y' in node_data and 'x' in node_data:
                node_coords[node_id] = (node_data['x'], node_data['y'])
            elif 'lon' in node_data and 'lat' in node_data:
                node_coords[node_id] = (node_data['lon'], node_data['lat'])
            elif 'pos' in node_data:
                pos = node_data['pos']
                node_coords[node_id] = (pos[0], pos[1])

        for u, v, data in edges:
            if u not in node_coords or v not in node_coords:
                continue
            u_lon, u_lat = node_coords[u]
            v_lon, v_lat = node_coords[v]
            if obstacles is not None:
                line = LineString([(u_lon, u_lat), (v_lon, v_lat)])
                if self._line_intersects_obstacles(line, obstacles):
                    continue
                if self._is_point_in_obstacles(Point(u_lon, u_lat), obstacles) or self._is_point_in_obstacles(Point(v_lon, v_lat), obstacles):
                    continue

            props = {
                "edge_type": data.get('edge_type', 'unknown'),
                "weight": data.get('weight', 0),
                "length": data.get('length', 0)
            }
            if include_node_ids:
                props["source_id"] = str(u)
                props["target_id"] = str(v)

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[u_lon, u_lat], [v_lon, v_lat]]
                },
                "properties": props
            })

        out = {
            "type": "FeatureCollection",
            "features": features
        }
        if include_node_ids:
            out["nodes"] = [
                {"id": str(n), "lon": float(lon), "lat": float(lat)}
                for n, (lon, lat) in node_coords.items()
            ]
        return out

    # Запас над зданием для разрешения полёта. Строгий: эшелон 1 (40м) не летает над оранжевыми (30+м)
    FLIGHT_LEVEL_SAFETY_MARGIN = 15.0

    def assign_flight_levels_to_edges(
        self,
        edges_geojson: dict,
        buildings: gpd.GeoDataFrame,
        flight_levels: list,
        margin_m: float | None = None,
        corridor_buffer_deg: float = 0.00003,
    ) -> dict:
        """
        Добавляет к каждому ребру allowed_flight_levels — список эшелонов,
        на которых можно пролететь по этому ребру (не пролегает над слишком высокими зданиями).

        Ребро блокируется на эшелоне L, если линия ребра пересекает здание,
        высота которого >= (altitude_L - margin_m). Используется строгий запас 15 м:
        эшелон 1 (40 м) разрешает только здания до 25 м (зелёные, жёлтые),
        оранжевые (30–50 м) и красные запрещены.

        Args:
            edges_geojson: GeoJSON FeatureCollection рёбер
            buildings: GeoDataFrame с колонками geometry, height_m
            flight_levels: [{"level": 1, "altitude_m": 40}, ...]
            margin_m: запас над зданием (м), по умолчанию FLIGHT_LEVEL_SAFETY_MARGIN=15
            corridor_buffer_deg: буфер линии ребра в градусах (~3 м)

        Returns:
            Тот же edges_geojson с properties.allowed_flight_levels в каждом feature
        """
        margin_m = margin_m if margin_m is not None else self.FLIGHT_LEVEL_SAFETY_MARGIN
        if buildings is None or len(buildings) == 0 or 'height_m' not in buildings.columns:
            for f in edges_geojson.get("features", []):
                f.setdefault("properties", {})["allowed_flight_levels"] = [
                    fl["level"] for fl in (flight_levels or [])
                ]
            return edges_geojson

        if not flight_levels:
            return edges_geojson

        building_geoms = list(buildings.geometry)
        building_heights = list(buildings["height_m"])
        tree = STRtree(building_geoms)

        levels_alt = [(fl["level"], fl["altitude_m"]) for fl in flight_levels]

        for feature in edges_geojson.get("features", []):
            geom = feature.get("geometry")
            if not geom or geom.get("type") != "LineString":
                feature.setdefault("properties", {})["allowed_flight_levels"] = [
                    fl[0] for fl in levels_alt
                ]
                continue
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                feature.setdefault("properties", {})["allowed_flight_levels"] = [
                    fl[0] for fl in levels_alt
                ]
                continue

            line = LineString(coords)
            corridor = line.buffer(corridor_buffer_deg)
            candidates = tree.query(corridor)

            max_height_under = 0.0
            for idx in candidates:
                idx = int(idx)
                if idx < len(building_geoms):
                    try:
                        if building_geoms[idx].intersects(corridor):
                            h = float(building_heights[idx])
                            max_height_under = max(max_height_under, h)
                    except Exception:
                        pass

            allowed = []
            for level, altitude in levels_alt:
                # Строгое условие: здание должно быть заметно ниже эшелона (margin_m)
                if max_height_under < altitude - margin_m:
                    allowed.append(level)

            feature.setdefault("properties", {})["allowed_flight_levels"] = allowed

        return edges_geojson

    # --- Многоуровневое планирование: связь эшелонов и траектория по кривой ---
    # Вес перехода по высоте: спуск дешевле подъёма (экономия батареи)
    VERTICAL_DESCENT_COST_PER_M = 0.002   # условных единиц за метр спуска
    VERTICAL_CLIMB_COST_PER_M = 0.01      # условных единиц за метр подъёма

    def build_multilevel_graph(
        self,
        graph: nx.Graph,
        edges_geojson: dict,
        flight_levels: list,
    ) -> nx.Graph:
        """
        Строит граф, в котором эшелоны связаны: дрон может переходить с одного уровня на другой в узле.
        Узлы нового графа — пары (node_id, level). Рёбра: горизонтальные (то же ребро на допустимом уровне)
        и вертикальные (тот же node_id, соседние уровни) для планирования с переходом эшелонов.

        Returns:
            NetworkX граф с узлами (node_id, level) и весами рёбер (горизонталь — длина, вертикаль — стоимость по высоте).
        """
        if not flight_levels:
            return nx.Graph()
        level_list = [fl["level"] for fl in flight_levels]
        alt_by_level = {fl["level"]: float(fl["altitude_m"]) for fl in flight_levels}

        # (u, v) -> [allowed levels]; u,v как строки из GeoJSON
        edge_allowed = {}
        for f in edges_geojson.get("features", []):
            p = f.get("properties") or {}
            sid = p.get("source_id")
            tid = p.get("target_id")
            if sid is None or tid is None:
                continue
            allowed = p.get("allowed_flight_levels") or []
            key = (min(sid, tid), max(sid, tid))
            edge_allowed[key] = list(allowed)

        G_multi = nx.Graph()
        node_coords = {}
        for n, data in graph.nodes(data=True):
            c = self._get_node_coords(graph, n)
            if c is None:
                continue
            node_coords[str(n)] = c
            for level in level_list:
                G_multi.add_node((str(n), level), lon=c[0], lat=c[1], level=level, altitude_m=alt_by_level.get(level))

        for u, v, data in graph.edges(data=True):
            u_s, v_s = str(u), str(v)
            key = (min(u_s, v_s), max(u_s, v_s))
            allowed = edge_allowed.get(key)
            if not allowed:
                continue
            w = float(data.get("weight") or data.get("length") or 0)
            for level in allowed:
                if (u_s, level) not in G_multi.nodes or (v_s, level) not in G_multi.nodes:
                    continue
                G_multi.add_edge((u_s, level), (v_s, level), weight=w, length=w, edge_kind="horizontal")

        for n_s in node_coords:
            for i in range(len(level_list) - 1):
                L1, L2 = level_list[i], level_list[i + 1]
                alt1 = alt_by_level.get(L1, 0)
                alt2 = alt_by_level.get(L2, 0)
                if (n_s, L1) not in G_multi.nodes or (n_s, L2) not in G_multi.nodes:
                    continue
                delta = abs(alt2 - alt1)
                cost = delta * (self.VERTICAL_DESCENT_COST_PER_M if alt2 < alt1 else self.VERTICAL_CLIMB_COST_PER_M)
                G_multi.add_edge((n_s, L1), (n_s, L2), weight=cost, length=0, edge_kind="vertical")

        self.logger.info(
            f"Многоуровневый граф: {G_multi.number_of_nodes()} узлов, {G_multi.number_of_edges()} рёбер "
            f"(эшелоны связаны, переход по кривой возможен)"
        )
        return G_multi

    def path_to_3d_curve(
        self,
        path_with_levels: List[Tuple],
        node_coords: dict,
        flight_levels: list,
        points_per_segment: int = 10,
    ) -> List[dict]:
        """
        Преобразует путь [(node_id, level), ...] в 3D-траекторию по кривой:
        высота плавно меняется вдоль сегмента (не только вертикальный скачок в узле).
        Возвращает список {"lon", "lat", "alt_m"} для построения маршрута дрона.

        path_with_levels: список кортежей (node_id, level). node_id — строка.
        node_coords: {node_id: (lon, lat)}
        flight_levels: [{"level": 1, "altitude_m": 40}, ...]
        points_per_segment: число точек на сегменте для плавной кривой (по умолчанию 10).
        """
        if not path_with_levels or not flight_levels:
            return []
        alt_by_level = {fl["level"]: float(fl["altitude_m"]) for fl in flight_levels}
        out = []
        for k in range(len(path_with_levels) - 1):
            n1, l1 = path_with_levels[k]
            n2, l2 = path_with_levels[k + 1]
            n1_s, n2_s = str(n1), str(n2)
            c1 = node_coords.get(n1_s)
            c2 = node_coords.get(n2_s)
            if c1 is None or c2 is None:
                continue
            lon1, lat1 = c1
            lon2, lat2 = c2
            alt1 = alt_by_level.get(l1, 0)
            alt2 = alt_by_level.get(l2, 0)

            if n1_s == n2_s:
                out.append({"lon": lon1, "lat": lat1, "alt_m": alt1})
                out.append({"lon": lon2, "lat": lat2, "alt_m": alt2})
                continue

            for i in range(points_per_segment + 1):
                t = i / points_per_segment
                lon = lon1 + t * (lon2 - lon1)
                lat = lat1 + t * (lat2 - lat1)
                alt = alt1 + t * (alt2 - alt1)
                out.append({"lon": lon, "lat": lat, "alt_m": round(alt, 2)})
        if path_with_levels and not out:
            n1, l1 = path_with_levels[0]
            c1 = node_coords.get(str(n1))
            if c1:
                out.append({"lon": c1[0], "lat": c1[1], "alt_m": round(alt_by_level.get(l1, 0), 2)})
        return out

    def shortest_path_multilevel(
        self,
        graph: nx.Graph,
        edges_geojson: dict,
        flight_levels: list,
        source_node,
        target_node,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
    ) -> Tuple[List[Tuple], List[dict]]:
        """
        Строит кратчайший путь в многоуровневом графе (с переходами между эшелонами).
        Возвращает путь как список (node_id, level) и 3D-кривую для траектории (плавное изменение высоты).

        source_node, target_node: id узла в исходном графе (строка или как в графе).
        source_level, target_level: желаемый эшелон в начале/конце (None — любой).
        """
        G_multi = self.build_multilevel_graph(graph, edges_geojson, flight_levels)
        if G_multi.number_of_nodes() == 0:
            return [], []

        node_coords = {}
        for n, data in graph.nodes(data=True):
            c = self._get_node_coords(graph, n)
            if c is not None:
                node_coords[str(n)] = c

        levels = [fl["level"] for fl in flight_levels]
        src_s, tgt_s = str(source_node), str(target_node)

        starts = [(src_s, source_level)] if source_level is not None else [(src_s, L) for L in levels if (src_s, L) in G_multi.nodes]
        ends = [(tgt_s, target_level)] if target_level is not None else [(tgt_s, L) for L in levels if (tgt_s, L) in G_multi.nodes]
        if not starts or not ends:
            return [], []

        best_path = None
        best_len = float("inf")

        for start in starts:
            for end in ends:
                try:
                    path = nx.shortest_path(G_multi, start, end, weight="weight")
                    length = sum(
                        G_multi.edges[path[i], path[i + 1]].get("weight", 0)
                        for i in range(len(path) - 1)
                    )
                    if length < best_len:
                        best_len = length
                        best_path = path
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        if best_path is None:
            return [], []
        curve = self.path_to_3d_curve(best_path, node_coords, flight_levels)
        return best_path, curve

    def graph_from_geojson_with_nodes(self, edges_geojson: dict) -> nx.Graph:
        """
        Собирает NetworkX граф из GeoJSON с узлами и рёбрами (source_id, target_id, weight).
        Нужен для расчёта маршрута по API, когда есть только edges + nodes.
        """
        G = nx.Graph()
        nodes = edges_geojson.get("nodes") or []
        for no in nodes:
            nid = no.get("id")
            lon = no.get("lon")
            lat = no.get("lat")
            if nid is not None and lon is not None and lat is not None:
                G.add_node(str(nid), lon=float(lon), lat=float(lat))
        for f in edges_geojson.get("features", []):
            p = f.get("properties") or {}
            u, v = p.get("source_id"), p.get("target_id")
            if u is None or v is None:
                continue
            u, v = str(u), str(v)
            if not G.has_node(u) or not G.has_node(v):
                continue
            w = float(p.get("weight") or p.get("length") or 0)
            G.add_edge(u, v, weight=w, length=w)
        return G
