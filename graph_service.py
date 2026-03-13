import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
import logging
from typing import Tuple, List, Optional, Union
class GraphService:
    """Сервис для генерации различных типов графов для планирования маршрутов дронов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
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

    def build_air_graph(
        self,
        city_boundary,
        buildings: Optional[gpd.GeoDataFrame] = None,
        no_fly_zones: Optional[Union[gpd.GeoDataFrame, List]] = None,
        *,
        building_buffer_deg: float = 0.00025,
        min_distance_deg: float = 0.0003,
        max_points: int = 3000,
        k_neighbors: int = 6,
    ) -> nx.Graph:
        """
        Строит навигационный граф в свободном воздушном пространстве, не опираясь на дорожный граф.
        Узлы — точки в свободной зоне (Poisson-disk по городу), рёбра — прямые отрезки между
        ближайшими соседями, не пересекающие бесполётные зоны и буфер зданий.

        Такой граф используется для обхода no_fly_zones только в воздухе.
        """
        G = nx.Graph()
        if city_boundary is None:
            return G

        try:
            points = self.get_flyable_points(
                city_boundary=city_boundary,
                buildings=buildings,
                no_fly_zones=no_fly_zones,
                building_buffer_deg=building_buffer_deg,
                min_distance_deg=min_distance_deg,
                max_points=max_points,
            )
        except Exception as e:
            self.logger.warning(f"Ошибка build_air_graph/get_flyable_points: {e}")
            return G

        if not points:
            return G

        obstacles = None
        try:
            obstacles = self._prepare_obstacles(buildings, no_fly_zones, buffer_distance=0.0)
        except Exception:
            obstacles = None

        coords = np.array(points)
        for idx, (lon, lat) in enumerate(points):
            G.add_node(idx, lon=float(lon), lat=float(lat))

        if len(points) < 2:
            return G

        tree = cKDTree(coords)
        k = max(1, min(int(k_neighbors), len(points) - 1))

        for i, (lon_i, lat_i) in enumerate(points):
            dists, idxs = tree.query([lon_i, lat_i], k=k + 1)
            try:
                it = zip(dists, idxs)
            except TypeError:
                dists = [dists]
                idxs = [idxs]
                it = zip(dists, idxs)
            for dist, j in it:
                if j == i or j < 0:
                    continue
                lon_j, lat_j = points[int(j)]
                if obstacles is not None:
                    line = LineString([(lon_i, lat_i), (lon_j, lat_j)])
                    if self._line_intersects_obstacles(line, obstacles):
                        continue
                w = self._calculate_distance(lat_i, lon_i, lat_j, lon_j)
                if w <= 0:
                    continue
                if G.has_edge(i, int(j)):
                    continue
                G.add_edge(i, int(j), weight=float(w), length=float(w))

        self.logger.info(
            f"Воздушный граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер "
            f"(k_nearest={k_neighbors}, max_points={max_points})"
        )
        return G

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
