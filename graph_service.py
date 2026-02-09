import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, nearest_points
import geopandas as gpd
from scipy.spatial import Delaunay, cKDTree
from shapely.strtree import STRtree
import logging
from typing import Tuple, List, Optional, Union
from collections import defaultdict


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
        add_perpendicular_edges: bool = True,
        max_perpendicular_distance_deg: float = 0.009,
        add_vertex_to_midpoint_edges: bool = True,
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
            add_perpendicular_edges: добавлять перпендикуляры между противоположными рёбрами смежных треугольников
            max_perpendicular_distance_deg: макс. длина перпендикуляра в градусах (~0.009 ≈ 1000 м)
            add_vertex_to_midpoint_edges: добавлять рёбра от вершин в центр (середину) противолежащего ребра
            
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
        
        if add_perpendicular_edges:
            G = self._add_perpendicular_edges(
                G, points_arr, tri, obstacles, max_perpendicular_distance_deg
            )
        
        if add_vertex_to_midpoint_edges:
            G = self._add_vertex_to_midpoint_edges(G, points_arr, tri, obstacles)
        
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
    
    def _add_perpendicular_edges(
        self,
        G: nx.Graph,
        points_arr: np.ndarray,
        tri,
        obstacles: Optional[Polygon],
        max_distance_deg: float,
    ) -> nx.Graph:
        """
        Добавляет перпендикулярные рёбра между противоположными рёбрами смежных треугольников Delaunay.
        Для каждой внутренней стороны (общей для 2 треугольников) берутся два противоположных ребра
        четырёхугольника и соединяются кратчайшим перпендикуляром.
        """
        # Собираем пары противоположных рёбер: для каждого internal edge (n1,n2) получаем
        # два треугольника, откуда пары (n1,n3)-(n2,n4) и (n2,n3)-(n1,n4)
        edge_to_triangles = defaultdict(list)  # (min(n1,n2), max(n1,n2)) -> [simplex_idx, ...]
        for idx, simplex in enumerate(tri.simplices):
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                key = (min(a, b), max(a, b))
                edge_to_triangles[key].append(idx)
        
        next_node_id = len(points_arr)
        perpendiculars_added = 0
        processed_pairs = set()  # frozenset({(a,b),(c,d)}) чтобы не дублировать
        
        for (n1, n2), tri_idxs in edge_to_triangles.items():
            if len(tri_idxs) != 2:
                continue  # граничное ребро
            s1, s2 = tri.simplices[tri_idxs[0]], tri.simplices[tri_idxs[1]]
            # Находим общие вершины n1,n2 и оставшиеся n3, n4
            common = set(s1) & set(s2)
            others = (set(s1) | set(s2)) - common
            if len(common) != 2 or len(others) != 2:
                continue
            n3, n4 = list(others)
            # Пары противоположных рёбер (без общей вершины): (n1,n3)-(n2,n4) и (n1,n4)-(n2,n3)
            pairs = [
                ((n1, n3), (n2, n4)),
                ((n1, n4), (n2, n3)),
            ]
            for (e1_a, e1_b), (e2_a, e2_b) in pairs:
                pair_key = frozenset([(min(e1_a, e1_b), max(e1_a, e1_b)), (min(e2_a, e2_b), max(e2_a, e2_b))])
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                if not G.has_edge(e1_a, e1_b) or not G.has_edge(e2_a, e2_b):
                    continue
                line1 = LineString([points_arr[e1_a], points_arr[e1_b]])
                line2 = LineString([points_arr[e2_a], points_arr[e2_b]])
                try:
                    pt1, pt2 = nearest_points(line1, line2)
                except Exception:
                    continue
                lon1, lat1 = pt1.x, pt1.y
                lon2, lat2 = pt2.x, pt2.y
                if self._is_point_in_obstacles(Point(lon1, lat1), obstacles) or self._is_point_in_obstacles(Point(lon2, lat2), obstacles):
                    continue
                dist_deg = np.hypot(lon2 - lon1, lat2 - lat1)
                if dist_deg <= 1e-9 or dist_deg > max_distance_deg:
                    continue
                perp_line = LineString([pt1, pt2])
                if self._line_intersects_obstacles(perp_line, obstacles):
                    continue
                # Добавляем узлы и ребро; разбиваем исходные рёбра
                id1, id2 = next_node_id, next_node_id + 1
                next_node_id += 2
                G.add_node(id1, lat=lat1, lon=lon1, pos=(lon1, lat1))
                G.add_node(id2, lat=lat2, lon=lon2, pos=(lon2, lat2))
                dist_km = self._calculate_distance(lat1, lon1, lat2, lon2)
                G.add_edge(id1, id2, weight=dist_km, length=dist_km)
                perpendiculars_added += 1
                # Подключаем к графу: id1 на линии (e1_a,e1_b), id2 на (e2_a,e2_b)
                for (na, nb), (nid, flon, flat) in [
                    ((e1_a, e1_b), (id1, lon1, lat1)),
                    ((e2_a, e2_b), (id2, lon2, lat2)),
                ]:
                    G.remove_edge(na, nb)
                    pa, pb = points_arr[na], points_arr[nb]
                    da = self._calculate_distance(pa[1], pa[0], flat, flon)
                    db = self._calculate_distance(pb[1], pb[0], flat, flon)
                    G.add_edge(na, nid, weight=da, length=da)
                    G.add_edge(nid, nb, weight=db, length=db)
        
        if perpendiculars_added > 0:
            self.logger.info(f"Добавлено {perpendiculars_added} перпендикулярных рёбер")
        return G
    
    def _add_vertex_to_midpoint_edges(
        self,
        G: nx.Graph,
        points_arr: np.ndarray,
        tri,
        obstacles: Optional[Polygon],
    ) -> nx.Graph:
        """
        Добавляет рёбра от каждой вершины треугольника в центр (середину) противолежащего ребра.
        Для треугольника ABC: A->mid(BC), B->mid(AC), C->mid(AB).
        """
        def get_node_coords(n):
            if n < len(points_arr):
                return points_arr[n][0], points_arr[n][1]
            d = G.nodes.get(n, {})
            return d.get("lon", d.get("pos", (0, 0))[0]), d.get("lat", d.get("pos", (0, 0))[1])
        
        def find_edge_containing_point(ua, wb, mid_pt):
            """Находит ребро на пути ua->wb, содержащее точку mid_pt."""
            try:
                path = nx.shortest_path(G, ua, wb)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
            pt = Point(mid_pt.x, mid_pt.y)
            tol = 1e-8
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                ax, ay = get_node_coords(a)
                bx, by = get_node_coords(b)
                line = LineString([(ax, ay), (bx, by)])
                if line.distance(pt) > tol:
                    continue
                proj = line.project(pt)
                if -tol <= proj <= line.length + tol:
                    return (a, b)
            return None
        
        edge_to_midpoint_id = {}
        next_node_id = max(G.nodes(), default=-1) + 1
        added = 0
        mid_pt = Point(0, 0)  # placeholder
        
        for simplex in tri.simplices:
            for i in range(3):
                v = simplex[i]
                u, w = simplex[(i + 1) % 3], simplex[(i + 2) % 3]
                edge_key = (min(u, w), max(u, w))
                pu, pw, pv = points_arr[u], points_arr[w], points_arr[v]
                mid_lon = (pu[0] + pw[0]) / 2
                mid_lat = (pu[1] + pw[1]) / 2
                mid_pt = Point(mid_lon, mid_lat)
                if self._is_point_in_obstacles(mid_pt, obstacles):
                    continue
                if edge_key not in edge_to_midpoint_id:
                    mid_id = next_node_id
                    next_node_id += 1
                    G.add_node(mid_id, lat=mid_lat, lon=mid_lon, pos=(mid_lon, mid_lat))
                    edge_to_midpoint_id[edge_key] = mid_id
                    if G.has_edge(u, w):
                        G.remove_edge(u, w)
                        d_um = self._calculate_distance(pu[1], pu[0], mid_lat, mid_lon)
                        d_mw = self._calculate_distance(mid_lat, mid_lon, pw[1], pw[0])
                        G.add_edge(u, mid_id, weight=d_um, length=d_um)
                        G.add_edge(mid_id, w, weight=d_mw, length=d_mw)
                    else:
                        seg = find_edge_containing_point(u, w, mid_pt)
                        if seg:
                            a, b = seg
                            G.remove_edge(a, b)
                            ax, ay = get_node_coords(a)
                            bx, by = get_node_coords(b)
                            d_am = self._calculate_distance(ay, ax, mid_lat, mid_lon)
                            d_mb = self._calculate_distance(mid_lat, mid_lon, by, bx)
                            G.add_edge(a, mid_id, weight=d_am, length=d_am)
                            G.add_edge(mid_id, b, weight=d_mb, length=d_mb)
                        else:
                            d_um = self._calculate_distance(pu[1], pu[0], mid_lat, mid_lon)
                            d_mw = self._calculate_distance(mid_lat, mid_lon, pw[1], pw[0])
                            G.add_edge(u, mid_id, weight=d_um, length=d_um)
                            G.add_edge(mid_id, w, weight=d_mw, length=d_mw)
                mid_id = edge_to_midpoint_id[edge_key]
                line_vm = LineString([pv, (mid_lon, mid_lat)])
                if self._line_intersects_obstacles(line_vm, obstacles):
                    continue
                if not G.has_edge(v, mid_id):
                    dist = self._calculate_distance(pv[1], pv[0], mid_lat, mid_lon)
                    G.add_edge(v, mid_id, weight=dist, length=dist)
                    added += 1
        
        if added > 0:
            self.logger.info(f"Добавлено {added} рёбер вершина->центр противолежащего ребра")
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
        
        # 5. Соединяем каждый узел Delaunay с ближайшим узлом дороги, если в пределах порога
        for merged_id, data in list(G.nodes(data=True)):
            if not str(merged_id).startswith(d_prefix):
                continue
            lon = data.get("lon")
            lat = data.get("lat")
            if lon is None or lat is None:
                continue
            dist_deg, idx_min = road_tree.query([lon, lat], k=1)
            idx_min = int(np.atleast_1d(idx_min).flat[0])
            if float(dist_deg) <= max_connection_distance_deg:
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
    ) -> dict:
        """
        Конвертирует NetworkX граф в GeoJSON для визуализации (ОПТИМИЗИРОВАНО)
        
        Args:
            graph: NetworkX граф
            edge_type_filter: фильтр по типу рёбер ('road', 'free', 'connection', None для всех)
            obstacles: препятствия — рёбра над ними исключаются из вывода
            
        Returns:
            GeoJSON FeatureCollection
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
