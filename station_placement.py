# -*- coding: utf-8 -*-
"""
Шаги:
  1) Зоны спроса
  2) Зарядки
  3) Гаражи и ТО: по ceil(N_A/3) каждого, промзона; разнос между точками и от зарядок (см. MIN_FACILITY_SEPARATION_KM); ветви только к ближайшей зарядке A.
  4) Магистраль: только между станциями Charge_A (MST по A); станции Б — по одной ветке к A (глобальный min-cost при квоте MAX_TYPE_B_BRANCHES_PER_TYPE_A Б на A), плюс до двух локальных связей Б↔Б; гараж/ТО — только к A.
  5) Локальная сеть Б↔Б, метрики; в ответе placement — полный слой Вороного (сайты зданий + станции с cluster_id).

Тут реализовано:
Этап 5: размещение инфраструктурных объектов -> `StationPlacement` и жадные функции покрытия.
Этап 6-7: транспортный граф и магистраль -> `run_full_pipeline()` + trunk/branch сборка.
Этап 8: локальная внутрикластерная сеть -> слой Voronoi подключается в финале пайплайна.
Этап 9-10: маршруты A* и обход препятствий/no-fly -> `astar_path_safe()` и detour-функции.
Этап 11: эшелонирование -> эшелоны 4-5 для магистралей/ветвей, 1-3 для Voronoi (через соседний модуль).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union, transform as shapely_transform

from pyproj import CRS, Transformer

# Радиус Земли в км
EARTH_R_KM = 6371.0

# FlyCart 30 с грузом: D_max = 16 км (радиусы уменьшены в 4 раза для отображения/расчёта)
D_MAX_KM = 16.0
# Зарядка + ожидание (туда-обратно): R = 0.4 * D_max / 4 = 1.6 км
R_CHARGE_KM = 0.4 * D_MAX_KM / 4.0
# Гаражи и ТО (в одну сторону): R = 0.8 * D_max / 4 = 3.2 км
R_GARAGE_TO_KM = 0.8 * D_MAX_KM / 4.0
# Разные типы станций (гараж / ТО / зарядка) и несколько гаражей — не в одной точке (разные корпуса).
MIN_FACILITY_SEPARATION_KM = 0.4

# Резерв имён (раньше использовался доп. буфер вокруг NFZ для маршрутизации; сейчас геометрия зон берётся как в данных).
NO_FLY_DETOUR_OFFSET_M = 10.0
NO_FLY_CLEARANCE_M = 0.0
# Максимум станций типа Б, присоединённых веткой к одной станции типа А.
MAX_TYPE_B_BRANCHES_PER_TYPE_A = 4

# Выбор между boundary и A*: сначала минимизируем длину/хорду (насколько обход близок к прямой), затем абсолютную длину.
# Доп. штраф за километры «сверх хорды» (слабее ratio, чтобы не ломать выбор при почти равных ratio).
DETOUR_SCORE_EXCESS_PER_KM = 0.08

# Шаг дискретизации контура при обходе по границе (метры) — меньше = плавнее повороты.
NO_FLY_BOUNDARY_STEP_M = 10.0
# Скругление обходов после A*/контура: (edge_frac, итерации Chaikin), по возрастанию силы.
# Для каждого варианта полилиния сглаживается заново от исходной; берётся самый сильный прошедший no-fly.
DETOUR_SMOOTH_SCHEDULE: Tuple[Tuple[float, int], ...] = (
    (0.14, 1),
    (0.18, 1),
    (0.22, 2),
    (0.26, 1),
    (0.30, 1),
    (0.32, 2),
    (0.36, 2),
    (0.40, 2),
    # Классический Chaikin (0.25) с несколькими итерациями — мягкие дуги без «ломаных» после simplify.
    (0.25, 3),
    (0.25, 4),
    (0.22, 3),
    (0.20, 2),
)
# Обратная совместимость имён (если где-то импортировали константу).
DETOUR_SMOOTH_EDGE_FRACS: Tuple[float, ...] = tuple(f for f, _ in DETOUR_SMOOTH_SCHEDULE)
# Параметры сетки «воздушного» графа для A* (плотнее — короче допустимый обход при тех же ограничениях).
# Грубее сетка и меньше узлов — быстрее A* (раньше до 24k узлов × полный copy на каждый сегмент).
UAV_AIR_GRID_SPACING_KM = 0.10
UAV_AIR_GRID_MAX_NODES = 10000

# Назначение Б→А: только рёбра «каждая А — до 4 ближайших Б» (+ гарантия хотя бы одно ребро на Б).
BRANCH_NEAREST_B_PER_STATION_A = 4
# Штраф в целевой функции min-cost за превышение max_branch_km: scale * (d + K * max(0, d-max)^2).
BRANCH_ASSIGN_OVER_MAX_KM_PENALTY = 4.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками в км (формула гаверсинуса)."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(min(1.0, a)))
    return EARTH_R_KM * c


def _coords_from_geom(g) -> Tuple[float, float]:
    """Извлекает координаты (lon, lat) из Point/геометрии (через centroid как fallback)."""
    if g is None:
        return (0.0, 0.0)
    if hasattr(g, "x") and hasattr(g, "y"):
        return (float(g.x), float(g.y))
    c = getattr(g, "centroid", None)
    if c is not None:
        return (float(c.x), float(c.y))
    return (0.0, 0.0)


def _points_array(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Преобразует GeoDataFrame геометрий в массив координат Nx2."""
    if gdf is None or len(gdf) == 0:
        return np.empty((0, 2))
    coords = []
    for _, row in gdf.iterrows():
        g = row.geometry
        coords.append(_coords_from_geom(g))
    return np.array(coords)


def _lonlat_exclusions_from_gdf(gdf: Optional[gpd.GeoDataFrame]) -> List[Tuple[float, float]]:
    """Список (lon, lat) для запрета колокации."""
    if gdf is None or len(gdf) == 0:
        return []
    return [_coords_from_geom(row.geometry) for _, row in gdf.iterrows()]


def _respect_min_separation(
    lon: float,
    lat: float,
    chosen_idx: List[int],
    cand_pts: np.ndarray,
    exclude_lonlat: List[Tuple[float, float]],
    min_sep_km: float,
) -> bool:
    """Проверяет, что кандидат не ближе `min_sep_km` к выбранным и исключённым точкам."""
    if min_sep_km <= 0:
        return True
    for elon, elat in exclude_lonlat:
        if haversine_km(lat, lon, elat, elon) < min_sep_km:
            return False
    for ci in chosen_idx:
        elon, elat = float(cand_pts[ci, 0]), float(cand_pts[ci, 1])
        if haversine_km(lat, lon, elat, elon) < min_sep_km:
            return False
    return True


def _place_n_facilities_by_demand_coverage(
    demand: Optional[gpd.GeoDataFrame],
    candidates: Optional[gpd.GeoDataFrame],
    *,
    n: int,
    radius_km: float,
    station_type_label: str,
    log: Optional[logging.Logger] = None,
    min_separation_km: float = 0.0,
    exclude_lonlat: Optional[List[Tuple[float, float]]] = None,
) -> gpd.GeoDataFrame:
    """
    Ровно до `n` точек из кандидатов: жадно по максимальному приросту покрытого веса спроса в радиусе.
    Опционально — минимальное расстояние между выбранными точками и до ``exclude_lonlat`` (другие типы станций).
    Если с разнесением кандидатов не хватает, один раз ослабляем проверку для текущего шага (с предупреждением в лог).
    """
    if candidates is None or len(candidates) == 0 or n <= 0:
        return gpd.GeoDataFrame()
    n = min(int(n), len(candidates))
    exclude = list(exclude_lonlat or [])
    cand_pts = _points_array(candidates)

    if demand is None or len(demand) == 0:
        chosen_ns: List[int] = []
        for _ in range(n):
            pick = -1
            for c in range(len(cand_pts)):
                if c in chosen_ns:
                    continue
                lon_c, lat_c = float(cand_pts[c, 0]), float(cand_pts[c, 1])
                if _respect_min_separation(lon_c, lat_c, chosen_ns, cand_pts, exclude, min_separation_km):
                    pick = c
                    break
            if pick < 0:
                for c in range(len(cand_pts)):
                    if c not in chosen_ns:
                        pick = c
                        if log and min_separation_km > 0:
                            log.warning(
                                "%s: не найден кандидат с разносом ≥ %.2f км, берём ближайший доступный",
                                station_type_label,
                                min_separation_km,
                            )
                        break
            if pick < 0:
                break
            chosen_ns.append(pick)
        rows_ns = [candidates.iloc[i].copy() for i in chosen_ns]
        for r in rows_ns:
            r["station_type"] = station_type_label
        out_ns = gpd.GeoDataFrame(rows_ns, crs=candidates.crs) if rows_ns else gpd.GeoDataFrame()
        if log:
            log.info("%s: размещено %s из целевых %s", station_type_label, len(out_ns), n)
        return out_ns

    dem_pts = _points_array(demand)
    weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
    chosen: List[int] = []
    covered = np.zeros(len(dem_pts), dtype=bool)
    warned_relax = False
    for _ in range(n):
        best_cand = -1
        best_new = -1.0
        for c in range(len(cand_pts)):
            if c in chosen:
                continue
            lat_c, lon_c = cand_pts[c, 1], cand_pts[c, 0]
            if not _respect_min_separation(lon_c, lat_c, chosen, cand_pts, exclude, min_separation_km):
                continue
            new_w = 0.0
            for j in range(len(dem_pts)):
                if covered[j]:
                    continue
                if haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0]) <= radius_km:
                    new_w += float(weights[j])
            if new_w > best_new:
                best_new = new_w
                best_cand = c
        if best_cand < 0 and best_new < 0:
            for c in range(len(cand_pts)):
                if c in chosen:
                    continue
                lat_c, lon_c = cand_pts[c, 1], cand_pts[c, 0]
                new_w = 0.0
                for j in range(len(dem_pts)):
                    if covered[j]:
                        continue
                    if haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0]) <= radius_km:
                        new_w += float(weights[j])
                if new_w > best_new:
                    best_new = new_w
                    best_cand = c
            if best_cand >= 0 and log and min_separation_km > 0 and not warned_relax:
                log.warning(
                    "%s: для шага размещения нет кандидата с разносом ≥ %.2f км — ослабляем критерий",
                    station_type_label,
                    min_separation_km,
                )
                warned_relax = True
        if best_cand < 0:
            break
        chosen.append(best_cand)
        lat_c, lon_c = cand_pts[best_cand, 1], cand_pts[best_cand, 0]
        for j in range(len(dem_pts)):
            if haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0]) <= radius_km:
                covered[j] = True

    rows = [candidates.iloc[i].copy() for i in chosen]
    for r in rows:
        r["station_type"] = station_type_label
    out = gpd.GeoDataFrame(rows, crs=candidates.crs)
    if log:
        log.info("%s: размещено %s из целевых %s", station_type_label, len(out), n)
    return out


def _lat_center_from_boundary(city_boundary) -> float:
    """Средняя широта для приближения км -> градусы."""
    if city_boundary is None:
        return 55.0
    try:
        b = city_boundary.bounds
        return (b[1] + b[3]) / 2.0
    except Exception:
        return 55.0


def km_to_deg_approx(km: float, lat_center: float) -> float:
    """Приблизительное преобразование км в градусы (для радиуса). 1° широты ≈ 111 км; долгота ≈ 111*cos(lat)."""
    deg_lat = km / 111.0
    deg_lon = km / (111.0 * max(0.2, np.cos(np.radians(lat_center))))
    return max(deg_lat, deg_lon)


def reconstruct_path(came_from: Dict[Any, Any], current: Any) -> List[Any]:
    """Восстановление пути A* из came_from."""
    path: List[Any] = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _iter_no_fly_geoms(no_fly_zones: Any) -> List[Any]:
    """Нормализует no-fly zones к списку shapely-геометрий."""
    if no_fly_zones is None:
        return []
    try:
        if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
            geoms = [g for g in no_fly_zones.geometry.tolist() if g is not None]
        elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
            geoms = [g for g in no_fly_zones if g is not None]
        else:
            if getattr(no_fly_zones, "is_empty", True):
                return []
            geoms = list(getattr(no_fly_zones, "geoms", [])) or [no_fly_zones]
        return [g for g in geoms if g is not None and not getattr(g, "is_empty", True)]
    except Exception:
        return []


def is_point_in_no_fly_zone(point: Tuple[float, float], no_fly_zones: Any, safety_buffer: float = 0.0) -> bool:
    """Проверяет, находится ли точка внутри no-fly зоны (граница считается допустимой)."""
    if no_fly_zones is None:
        return False
    lon, lat = float(point[0]), float(point[1])
    pt = Point(lon, lat)
    for zone in _iter_no_fly_geoms(no_fly_zones):
        try:
            z = zone.buffer(safety_buffer) if safety_buffer else zone
            # Считаем запретным только "внутри", касание границы допускаем,
            # чтобы маршрут мог идти по контуру ближе (detour будет короче).
            if z.contains(pt):
                return True
        except Exception:
            continue
    return False


def is_edge_blocked(edge_line: LineString, no_fly_zones: Any, safety_buffer: float = 0.0) -> bool:
    """Запрещает ребро, если оно пересекает/касается/проходит внутри no-fly зоны."""
    if no_fly_zones is None:
        return False
    if edge_line is None or getattr(edge_line, "is_empty", True):
        return False
    eps = 1e-9
    for zone in _iter_no_fly_geoms(no_fly_zones):
        try:
            z = zone.buffer(safety_buffer) if safety_buffer else zone
            inter = z.intersection(edge_line)
            if inter.is_empty:
                continue
            # Если пересечение только точки (касание границы), считаем ребро допустимым.
            if inter.geom_type in ("Point", "MultiPoint"):
                continue
            if inter.geom_type == "GeometryCollection":
                try:
                    parts = list(getattr(inter, "geoms", []))
                    if parts and all((p.geom_type in ("Point", "MultiPoint") or getattr(p, "length", 0.0) <= eps) for p in parts):
                        continue
                except Exception:
                    pass
            # Если пересечение имеет длину (линия в/через полигон) — ребро блокируем.
            if getattr(inter, "length", 0.0) > eps:
                return True
            # Если вдруг это многоугольная/площадная часть — блокируем.
            if getattr(inter, "area", 0.0) > eps:
                return True
            # Остальные типы (LineString/GeometryCollection с линиями и т.п.) — консервативно блокируем.
            return True
        except Exception:
            continue
    return False


def mark_blocked_edges(
    graph: nx.Graph,
    no_fly_zones: Any,
    safety_buffer: float = 0.0,
) -> Tuple[Set[Tuple[Any, Any]], Set[Any]]:
    """
    Предобработка: находит заблокированные рёбра и узлы.
    Блокировка: узел внутри зоны или ребро пересекает/касается/проходит внутри зоны.
    """
    blocked_edges: Set[Tuple[Any, Any]] = set()
    blocked_nodes: Set[Any] = set()
    if graph is None or graph.number_of_nodes() == 0 or no_fly_zones is None:
        return blocked_edges, blocked_nodes

    zones = _iter_no_fly_geoms(no_fly_zones)
    if safety_buffer and safety_buffer != 0.0:
        # safety_buffer задаётся в метрах (не в градусах WGS84).
        sum_lon, sum_lat, nn = 0.0, 0.0, 0
        for nid in graph.nodes():
            ll = _node_lon_lat(nid)
            if ll is not None:
                sum_lon += float(ll[0])
                sum_lat += float(ll[1])
                nn += 1
        anchor_lon = (sum_lon / nn) if nn else 0.0
        anchor_lat = (sum_lat / nn) if nn else 0.0
        buffered: List[Any] = []
        for z in zones:
            try:
                buffered.append(buffer_geometry_wgs84_m(z, anchor_lon, anchor_lat, float(safety_buffer)))
            except Exception:
                buffered.append(z)
        zones = buffered

    def _node_lon_lat(nid: Any) -> Optional[Tuple[float, float]]:
        d = graph.nodes[nid]
        lon = d.get("lon") if isinstance(d, dict) else None
        lat = d.get("lat") if isinstance(d, dict) else None
        if lon is None:
            lon = d.get("x") if isinstance(d, dict) else None
        if lat is None:
            lat = d.get("y") if isinstance(d, dict) else None
        if lon is None or lat is None:
            return None
        return float(lon), float(lat)

    # Узлы: блокируем только если точка реально внутри (касание границы допускаем).
    for nid in graph.nodes():
        ll = _node_lon_lat(nid)
        if ll is None:
            continue
        lon, lat = ll
        pt = Point(lon, lat)
        try:
            if any(z.contains(pt) for z in zones):
                blocked_nodes.add(nid)
        except Exception:
            continue

    # Рёбра: блокируем если пересечение не сводится к касанию точкой/точками.
    for u, v in graph.edges():
        if u in blocked_nodes or v in blocked_nodes:
            blocked_edges.add((u, v))
            continue
        ll_u = _node_lon_lat(u)
        ll_v = _node_lon_lat(v)
        if ll_u is None or ll_v is None:
            continue
        lon_u, lat_u = ll_u
        lon_v, lat_v = ll_v
        line = LineString([(lon_u, lat_u), (lon_v, lat_v)])
        try:
            if is_edge_blocked(line, zones, safety_buffer=0.0):
                blocked_edges.add((u, v))
        except Exception:
            continue

    return blocked_edges, blocked_nodes


def build_safe_graph(graph: nx.Graph, no_fly_zones: Any, safety_buffer: float = 0.0) -> nx.Graph:
    """Возвращает копию графа с удалёнными заблокированными узлами и рёбрами."""
    if graph is None:
        return graph
    if no_fly_zones is None:
        safe_graph = graph.copy()
        safe_graph.graph["__no_fly_safe"] = False
        return safe_graph

    safe_graph = graph.copy()
    blocked_edges, blocked_nodes = mark_blocked_edges(safe_graph, no_fly_zones, safety_buffer=safety_buffer)
    if blocked_nodes:
        safe_graph.remove_nodes_from(blocked_nodes)
    if blocked_edges:
        safe_graph.remove_edges_from(list(blocked_edges))

    safe_graph.graph["__no_fly_safe"] = True
    safe_graph.graph["__no_fly_safe_safety_buffer"] = float(safety_buffer or 0.0)
    return safe_graph


def _node_lon_lat_from_graph(graph: nx.Graph, nid: Any) -> Optional[Tuple[float, float]]:
    """Читает координаты узла графа из полей lon/lat или x/y."""
    d = graph.nodes[nid]
    lon = d.get("lon") if isinstance(d, dict) else None
    lat = d.get("lat") if isinstance(d, dict) else None
    if lon is None:
        lon = d.get("x") if isinstance(d, dict) else None
    if lat is None:
        lat = d.get("y") if isinstance(d, dict) else None
    if lon is None or lat is None:
        return None
    return float(lon), float(lat)


# Этап 9: построение маршрута A* с учетом no-fly ограничений.
def astar_path_safe(
    graph: nx.Graph,
    start_node: Any,
    goal_node: Any,
    no_fly_zones: Any,
    safety_buffer: float = 0.0,
    weight_attr: str = "weight",
) -> Dict[str, Any]:
    """
    A* по безопасному графу: запрещает узлы/рёбра, пересекающие no-fly зоны.
    Возвращает статус: success|no_path|start_in_no_fly_zone|goal_in_no_fly_zone.
    """
    if graph is None:
        return {"status": "no_path", "path_nodes": [], "path_coords": [], "path_length_km": 0.0}

    start_ll = _node_lon_lat_from_graph(graph, start_node)
    goal_ll = _node_lon_lat_from_graph(graph, goal_node)

    if start_ll is not None and is_point_in_no_fly_zone(start_ll, no_fly_zones, safety_buffer=safety_buffer):
        return {"status": "start_in_no_fly_zone", "path_nodes": [], "path_coords": [], "path_length_km": 0.0}
    if goal_ll is not None and is_point_in_no_fly_zone(goal_ll, no_fly_zones, safety_buffer=safety_buffer):
        return {"status": "goal_in_no_fly_zone", "path_nodes": [], "path_coords": [], "path_length_km": 0.0}

    is_safe_flag = bool(graph.graph.get("__no_fly_safe", False))
    safe_buffer_flag = graph.graph.get("__no_fly_safe_safety_buffer", None)
    if is_safe_flag and safe_buffer_flag is not None and float(safe_buffer_flag) == float(safety_buffer or 0.0):
        safe_graph = graph
    else:
        safe_graph = build_safe_graph(graph, no_fly_zones, safety_buffer=safety_buffer)

    if start_node not in safe_graph or goal_node not in safe_graph:
        return {"status": "no_path", "path_nodes": [], "path_coords": [], "path_length_km": 0.0}

    def _heuristic(a: Any, b: Any) -> float:
        la = _node_lon_lat_from_graph(safe_graph, a)
        lb = _node_lon_lat_from_graph(safe_graph, b)
        if la is None or lb is None:
            return 0.0
        lon_a, lat_a = la
        lon_b, lat_b = lb
        return haversine_km(lat_a, lon_a, lat_b, lon_b)

    import heapq

    open_heap: List[Tuple[float, float, Any]] = []
    came_from: Dict[Any, Any] = {}
    g_score: Dict[Any, float] = {start_node: 0.0}
    heapq.heappush(open_heap, (_heuristic(start_node, goal_node), 0.0, start_node))

    while open_heap:
        _, cur_g, current = heapq.heappop(open_heap)
        if cur_g > g_score.get(current, float("inf")) + 1e-12:
            continue
        if current == goal_node:
            path_nodes = reconstruct_path(came_from, current)
            path_coords: List[Tuple[float, float]] = []
            for nid in path_nodes:
                ll = _node_lon_lat_from_graph(safe_graph, nid)
                if ll is not None:
                    path_coords.append(ll)
            return {
                "status": "success",
                "path_nodes": path_nodes,
                "path_coords": path_coords,
                "path_length_km": float(cur_g),
            }

        for neighbor in safe_graph.neighbors(current):
            edge_data = {}
            try:
                edge_data = safe_graph[current][neighbor]
            except Exception:
                pass
            w = edge_data.get(weight_attr)
            if w is None:
                w = edge_data.get("length")
            if w is None:
                w = 1.0
            try:
                w = float(w)
            except Exception:
                w = 1.0

            tentative_g = cur_g + w
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_heap, (tentative_g + _heuristic(neighbor, goal_node), tentative_g, neighbor))

    return {"status": "no_path", "path_nodes": [], "path_coords": [], "path_length_km": 0.0}


def buildings_footprint_union_wgs84(buildings: Any) -> Any:
    """Объединение контуров зданий (EPSG:4326), упрощение для скорости. None если нечего."""
    if buildings is None:
        return None
    try:
        if hasattr(buildings, "__len__") and len(buildings) == 0:
            return None
        b = buildings
        if hasattr(b, "to_crs"):
            b = b.to_crs("EPSG:4326")
        geoms: List[Any] = []
        geom_series = getattr(b, "geometry", None)
        if geom_series is None:
            return None
        for g in geom_series:
            if g is None or getattr(g, "is_empty", True):
                continue
            try:
                geoms.append(g.simplify(0.00006, preserve_topology=True))
            except Exception:
                geoms.append(g)
        if not geoms:
            return None
        # Полный union тысяч зданий может занимать минуты; упрощаем сильнее и при перегрузе пропускаем.
        if len(geoms) > 3000:
            log = logging.getLogger(__name__)
            log.warning(
                "Зданий %s — объединение контуров для воздушного графа пропущено (слишком много полигонов). "
                "Учитываются только NFZ; при необходимости уменьшите область города.",
                len(geoms),
            )
            return None
        u = unary_union(geoms)
        if u is None or getattr(u, "is_empty", True):
            return None
        return u
    except Exception:
        return None


def buildings_footprint_union_min_height_wgs84(buildings: Any, min_height_m: float) -> Any:
    """
    Объединение контуров зданий (EPSG:4326) с height_m >= min_height_m.
    Для магистрали / Вороного: препятствие на эшелоне, если здание не ниже эшелона более чем на 5 м.
    """
    if buildings is None:
        return None
    try:
        if hasattr(buildings, "__len__") and len(buildings) == 0:
            return None
        mh = float(min_height_m)
    except (TypeError, ValueError):
        return None
    try:
        b = buildings
        if hasattr(b, "to_crs"):
            b = b.to_crs("EPSG:4326")
        if getattr(b, "geometry", None) is None:
            return None
        geoms: List[Any] = []
        for _idx, row in b.iterrows():
            try:
                h = float(row.get("height_m", 0.0))
            except (TypeError, ValueError):
                h = 0.0
            if h < mh:
                continue
            g = row.geometry
            if g is None or getattr(g, "is_empty", True):
                continue
            try:
                geoms.append(g.simplify(0.00006, preserve_topology=True))
            except Exception:
                geoms.append(g)
        if not geoms:
            return None
        if len(geoms) > 3000:
            log = logging.getLogger(__name__)
            log.warning(
                "Зданий выше порога %.1f м — %s; объединение контуров для маршрутизации пропущено.",
                mh,
                len(geoms),
            )
            return None
        u = unary_union(geoms)
        if u is None or getattr(u, "is_empty", True):
            return None
        return u
    except Exception:
        return None


def build_uav_air_graph_grid(
    city_boundary: Any,
    no_fly_zones: Any,
    *,
    building_union: Any = None,
    grid_spacing_km: float = 0.35,
    connect_diagonal: bool = True,
    max_nodes: int = 8000,
    safety_buffer: float = 0.0,
) -> nx.Graph:
    """
    Сеточный "воздушный" граф для A*: узлы — точки в свободном пространстве,
    рёбра — прямые сегменты между соседями (8-связность), веса — haversine_km.
    Узлы/рёбра фильтруются по no-fly и (опционально) по контурам зданий.
    """
    G = nx.Graph()
    if city_boundary is None or getattr(city_boundary, "is_empty", True):
        return G
    if grid_spacing_km <= 0:
        return G

    # city_boundary и no_fly_zones ожидаются в тех же координатах, что и остальная логика (EPSG:4326).
    minx, miny, maxx, maxy = city_boundary.bounds
    if minx == maxx or miny == maxy:
        return G

    lat_center = float((miny + maxy) / 2.0)
    # Конвертация км -> градусы (приближённо), чтобы задать шаг сетки в lat/lon.
    deg_lat = grid_spacing_km / 111.0
    cos_lat = float(np.cos(np.radians(lat_center)))
    cos_lat = max(0.2, cos_lat)
    deg_lon = grid_spacing_km / (111.0 * cos_lat)

    if deg_lat <= 0 or deg_lon <= 0:
        return G

    nx_est = int(math.ceil((maxx - minx) / deg_lon)) + 1
    ny_est = int(math.ceil((maxy - miny) / deg_lat)) + 1
    est_nodes = nx_est * ny_est
    grid_scale = 1
    if est_nodes > max_nodes:
        grid_scale = int(math.ceil(math.sqrt(est_nodes / float(max_nodes))))
        grid_scale = max(1, grid_scale)
        deg_lon *= grid_scale
        deg_lat *= grid_scale

    lon_vals = np.arange(minx, maxx + deg_lon * 0.5, deg_lon, dtype=float)
    lat_vals = np.arange(miny, maxy + deg_lat * 0.5, deg_lat, dtype=float)

    grid_to_node: Dict[Tuple[int, int], Any] = {}
    node_lons_lats: Dict[Any, Tuple[float, float]] = {}

    # Создаём узлы. Фактический шаг ячейки (после укрупнения под max_nodes) — для привязки A* к сетке.
    G.graph["__grid_spacing_km"] = float(grid_spacing_km * grid_scale)
    for ix, lon in enumerate(lon_vals):
        for iy, lat in enumerate(lat_vals):
            pt = Point(float(lon), float(lat))
            # Узел должен быть внутри границы города.
            if not city_boundary.covers(pt):
                continue
            # Узел должен быть вне no-fly.
            if no_fly_zones is not None and is_point_in_no_fly_zone((lon, lat), no_fly_zones, safety_buffer=safety_buffer):
                continue
            if building_union is not None and not getattr(building_union, "is_empty", True):
                try:
                    if building_union.contains(pt) or building_union.covers(pt):
                        continue
                except Exception:
                    pass
            nid = f"air_{ix}_{iy}"
            G.add_node(nid, lon=float(lon), lat=float(lat), node_type="air")
            grid_to_node[(ix, iy)] = nid
            node_lons_lats[nid] = (float(lon), float(lat))

    if G.number_of_nodes() == 0:
        return G

    # Создаём рёбра между соседними узлами.
    # Чтобы не дублировать рёбра в undirected графе — добавляем только "вперёд" (в i/j направлениях).
    forward_offsets: List[Tuple[int, int]] = [(1, 0), (0, 1)]
    if connect_diagonal:
        forward_offsets += [(1, 1), (1, -1)]

    offsets_in_bounds = set(forward_offsets)

    for (ix, iy), nid in list(grid_to_node.items()):
        lon1, lat1 = node_lons_lats[nid]
        for dx, dy in offsets_in_bounds:
            jx = ix + dx
            jy = iy + dy
            nid2 = grid_to_node.get((jx, jy))
            if nid2 is None:
                continue
            lon2, lat2 = node_lons_lats[nid2]

            line = LineString([(lon1, lat1), (lon2, lat2)])
            if no_fly_zones is not None and is_edge_blocked(line, no_fly_zones, safety_buffer=safety_buffer):
                continue
            if building_union is not None and not getattr(building_union, "is_empty", True):
                try:
                    if is_edge_blocked(line, building_union, safety_buffer=0.0):
                        continue
                except Exception:
                    pass

            w_km = haversine_km(lat1, lon1, lat2, lon2)
            if w_km <= 0:
                continue
            G.add_edge(nid, nid2, weight=float(w_km), length=float(w_km), edge_type="air")

    # Удалим изолированные узлы (они не помогут A*).
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    if isolated:
        G.remove_nodes_from(isolated)

    return G


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """Локальный UTM EPSG по долготе/широте."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def buffer_geometry_wgs84_m(geom: Any, anchor_lon: float, anchor_lat: float, buffer_m: float) -> Any:
    """
    Буфер в метрах для геометрии в EPSG:4326 (через локальный UTM).
    Не использовать shapely buffer() в градусах — это не метры.
    """
    if geom is None or getattr(geom, "is_empty", True) or buffer_m <= 0:
        return geom
    try:
        utm_epsg = _utm_epsg_from_lonlat(float(anchor_lon), float(anchor_lat))
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_epsg(utm_epsg)
        fwd = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        inv = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
        g_m = shapely_transform(fwd.transform, geom)
        g_buf = g_m.buffer(float(buffer_m))
        return shapely_transform(inv.transform, g_buf)
    except Exception:
        return geom


def _smooth_polyline_chaikin_open_metric(
    xy: List[Tuple[float, float]],
    iterations: int,
    *,
    edge_frac: float = 0.25,
) -> List[Tuple[float, float]]:
    """
    Chaikin для открытой ломаной в метрических координатах; концы фиксированы.
    edge_frac=0.25 — классика; меньше (напр. 0.12) — слабее сглаживание, траектория ближе к исходной.
    """
    f = float(edge_frac)
    f = min(0.45, max(0.06, f))
    g = 1.0 - f
    pts = [tuple(p) for p in xy]
    for _ in range(max(0, int(iterations))):
        if len(pts) < 2:
            break
        new_pts: List[Tuple[float, float]] = [pts[0]]
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            new_pts.append((g * x0 + f * x1, g * y0 + f * y1))
            new_pts.append((f * x0 + g * x1, f * y0 + g * y1))
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def smooth_lonlat_polyline_chaikin(
    coords_lonlat: List[Tuple[float, float]],
    anchor_lon: float,
    anchor_lat: float,
    iterations: int,
    *,
    edge_frac: float = 0.25,
) -> List[Tuple[float, float]]:
    """Сглаживание маршрута в UTM и обратно в lon/lat."""
    if len(coords_lonlat) < 3 or iterations <= 0:
        return list(coords_lonlat)
    utm_epsg = _utm_epsg_from_lonlat(float(anchor_lon), float(anchor_lat))
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(utm_epsg)
    fwd = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    inv = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    xy = [fwd.transform(float(lo), float(la)) for lo, la in coords_lonlat]
    sm = _smooth_polyline_chaikin_open_metric(xy, iterations, edge_frac=edge_frac)
    out: List[Tuple[float, float]] = []
    for x, y in sm:
        lo, la = inv.transform(float(x), float(y))
        out.append((float(lo), float(la)))
    return out


def densify_lonlat_polyline_metric(
    coords_lonlat: List[Tuple[float, float]],
    anchor_lon: float,
    anchor_lat: float,
    *,
    max_step_m: float = 180.0,
    max_output_points: int = 48,
) -> List[Tuple[float, float]]:
    """
    Уплотняет ломаную в UTM: на длинных сегментах добавляет промежуточные точки
    с шагом не больше max_step_m. Концы сохраняются.
    """
    if len(coords_lonlat) < 2:
        return list(coords_lonlat)
    step_m = max(20.0, float(max_step_m))
    utm_epsg = _utm_epsg_from_lonlat(float(anchor_lon), float(anchor_lat))
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(utm_epsg)
    fwd = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    inv = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    xy = [fwd.transform(float(lo), float(la)) for lo, la in coords_lonlat]
    out_xy: List[Tuple[float, float]] = [tuple(xy[0])]
    for i in range(len(xy) - 1):
        x0, y0 = xy[i]
        x1, y1 = xy[i + 1]
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        seg_len = float(np.hypot(dx, dy))
        n_parts = max(1, int(np.ceil(seg_len / step_m)))
        for k in range(1, n_parts + 1):
            t = float(k) / float(n_parts)
            out_xy.append((float(x0 + t * dx), float(y0 + t * dy)))
            if len(out_xy) >= max(8, int(max_output_points)):
                break
        if len(out_xy) >= max(8, int(max_output_points)):
            break
    if out_xy[-1] != tuple(xy[-1]):
        out_xy.append(tuple(xy[-1]))
    out: List[Tuple[float, float]] = []
    for x, y in out_xy:
        lo, la = inv.transform(float(x), float(y))
        out.append((float(lo), float(la)))
    return out


def route_coords_clear_of_nfz(
    route_coords: Optional[List[Tuple[float, float]]],
    nfz_union: Any,
    *,
    nfz_metric: Optional[Tuple[Any, Optional[Transformer]]] = None,
    no_fly_safety_buffer: float = 0.0,
) -> bool:
    """Проверка, что вершины не внутри зоны и сегменты не пересекают препятствие."""
    if route_coords is None or len(route_coords) < 2:
        return False
    if nfz_union is None or getattr(nfz_union, "is_empty", True):
        return True
    nfz_union_metric: Optional[Any] = None
    to_metric: Optional[Transformer] = None
    if nfz_metric is not None:
        nfz_union_metric, to_metric = nfz_metric
    try:
        for lon, lat in route_coords:
            if nfz_union_metric is not None and to_metric is not None:
                x, y = to_metric.transform(float(lon), float(lat))
                if is_point_in_no_fly_zone((float(x), float(y)), nfz_union_metric, safety_buffer=0.0):
                    return False
            else:
                if is_point_in_no_fly_zone((float(lon), float(lat)), nfz_union, safety_buffer=no_fly_safety_buffer):
                    return False
        for i in range(len(route_coords) - 1):
            lon_a, lat_a = route_coords[i]
            lon_b, lat_b = route_coords[i + 1]
            if nfz_union_metric is not None and to_metric is not None:
                x1, y1 = to_metric.transform(float(lon_a), float(lat_a))
                x2, y2 = to_metric.transform(float(lon_b), float(lat_b))
                line_m = LineString([(float(x1), float(y1)), (float(x2), float(y2))])
                if is_edge_blocked(line_m, nfz_union_metric, safety_buffer=0.0):
                    return False
            else:
                line = LineString([(float(lon_a), float(lat_a)), (float(lon_b), float(lat_b))])
                if is_edge_blocked(line, nfz_union, safety_buffer=no_fly_safety_buffer):
                    return False
        return True
    except Exception:
        return False


def polish_detour_polyline(
    coords_lonlat: List[Tuple[float, float]],
    nfz_union: Any,
    *,
    nfz_metric: Optional[Tuple[Any, Optional[Transformer]]] = None,
    utm_anchor_lonlat: Optional[Tuple[float, float]] = None,
    no_fly_safety_buffer: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Скругление углов Chaikin в UTM (концы фиксированы). Для каждого шага DETOUR_SMOOTH_SCHEDULE
    сглаживается исходная ломаная; из прошедших проверку no-fly выбирается самый сильный (последний в расписании).
    """
    if len(coords_lonlat) < 3:
        return list(coords_lonlat)
    alon = float(utm_anchor_lonlat[0]) if utm_anchor_lonlat is not None else float(coords_lonlat[0][0])
    alat = float(utm_anchor_lonlat[1]) if utm_anchor_lonlat is not None else float(coords_lonlat[0][1])
    best = list(coords_lonlat)
    best_changed = False
    for frac, iters in DETOUR_SMOOTH_SCHEDULE:
        cand = smooth_lonlat_polyline_chaikin(
            coords_lonlat, alon, alat, int(iters), edge_frac=float(frac)
        )
        if route_coords_clear_of_nfz(
            cand,
            nfz_union,
            nfz_metric=nfz_metric,
            no_fly_safety_buffer=no_fly_safety_buffer,
        ):
            best = cand
            best_changed = True
    # Fallback для коротких/плотных обходов у границы NFZ:
    # очень мягкое Chaikin сглаживание, чтобы не оставлять исходные 3–4 точки.
    if not best_changed:
        for frac, iters in ((0.08, 1), (0.06, 1)):
            cand = smooth_lonlat_polyline_chaikin(
                coords_lonlat, alon, alat, int(iters), edge_frac=float(frac)
            )
            if route_coords_clear_of_nfz(
                cand,
                nfz_union,
                nfz_metric=nfz_metric,
                no_fly_safety_buffer=no_fly_safety_buffer,
            ):
                best = cand
                break
    return best


def light_extra_smoothing_lonlat(
    coords_lonlat: List[Tuple[float, float]],
    nfz_union: Any,
    *,
    nfz_metric: Optional[Tuple[Any, Optional[Transformer]]] = None,
    utm_anchor_lonlat: Optional[Tuple[float, float]] = None,
    no_fly_safety_buffer: float = 0.0,
) -> List[Tuple[float, float]]:
    """Дополнительное сглаживание после polish_detour — только если ломаная остаётся допустимой для NFZ."""
    if len(coords_lonlat) < 3:
        return list(coords_lonlat)
    alon = float(utm_anchor_lonlat[0]) if utm_anchor_lonlat is not None else float(coords_lonlat[0][0])
    alat = float(utm_anchor_lonlat[1]) if utm_anchor_lonlat is not None else float(coords_lonlat[0][1])
    for it, ef in ((2, 0.18), (1, 0.14)):
        cand = smooth_lonlat_polyline_chaikin(coords_lonlat, alon, alat, int(it), edge_frac=float(ef))
        if len(cand) < 3:
            continue
        if nfz_union is None or getattr(nfz_union, "is_empty", True):
            return cand
        if route_coords_clear_of_nfz(
            cand,
            nfz_union,
            nfz_metric=nfz_metric,
            no_fly_safety_buffer=no_fly_safety_buffer,
        ):
            return cand
    return list(coords_lonlat)


def _ring_signed_area_xy(coords_xy: List[Tuple[float, float]]) -> float:
    """Знак площади по внешнему кольцу: >0 обычно CCW."""
    area2 = 0.0
    n = len(coords_xy)
    for i in range(n):
        x1, y1 = coords_xy[i]
        x2, y2 = coords_xy[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return area2 / 2.0


def _nearest_visible_boundary_point_metric(
    start_xy: Tuple[float, float],
    safe_poly_metric: Polygon,
    *,
    step_samples: int = 72,
    eps: float = 1e-9,
) -> Point:
    """
    Ближайшая точка на exterior safe_poly_metric, такая что отрезок start->point
    не проходит через внутренность safe_poly_metric (считаем по midpoint.contains).
    """
    ring = safe_poly_metric.exterior
    if ring is None or ring.is_empty:
        raise ValueError("safe_poly_metric has no exterior ring")

    sx, sy = start_xy
    start_pt = Point(sx, sy)

    def _segment_safe(seg: LineString) -> bool:
        """Допускаем касание границы, запрещаем вход в внутренность."""
        try:
            inter = safe_poly_metric.intersection(seg)
        except Exception:
            return False
        if inter.is_empty:
            return True
        if getattr(inter, "length", 0.0) > eps:
            return False
        if getattr(inter, "area", 0.0) > eps:
            return False
        if inter.geom_type in ("Point", "MultiPoint"):
            return True
        if inter.geom_type == "GeometryCollection":
            try:
                parts = list(getattr(inter, "geoms", []))
                if not parts:
                    return True
                # Безопасно только если все части — точки/нуль-длины.
                for p in parts:
                    if p.geom_type in ("Point", "MultiPoint"):
                        continue
                    if getattr(p, "length", 0.0) > eps or getattr(p, "area", 0.0) > eps:
                        return False
                return True
            except Exception:
                return False
        return False

    # Быстрый кандидат: проектирование на кольцо
    d0 = ring.project(start_pt)
    p0 = ring.interpolate(d0)
    if _segment_safe(LineString([start_pt, p0])):
        return p0

    ring_len = ring.length
    if ring_len <= eps:
        return p0

    best_p = p0
    best_dist2 = float("inf")
    for i in range(step_samples + 1):
        d = (ring_len * i) / step_samples
        p = ring.interpolate(d)
        if not _segment_safe(LineString([start_pt, p])):
            continue
        dist2 = (p.x - sx) ** 2 + (p.y - sy) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_p = p
    return best_p


def _build_boundary_arcs_metric(
    safe_poly_metric: Polygon,
    entry_xy: Tuple[float, float],
    exit_xy: Tuple[float, float],
    *,
    step_m: float = 25.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Возвращает две дуги по exterior кольцу safe_poly_metric между entry и exit:
    - (clockwise_arc, counterclockwise_arc)
    каждая включает entry и exit (в metric-координатах).
    """
    ring = safe_poly_metric.exterior
    if ring is None or ring.is_empty:
        raise ValueError("safe_poly_metric has no exterior")

    entry_pt = Point(entry_xy[0], entry_xy[1])
    exit_pt = Point(exit_xy[0], exit_xy[1])

    ring_coords = list(ring.coords)
    if ring_coords and ring_coords[0] == ring_coords[-1]:
        ring_coords = ring_coords[:-1]
    ring_is_ccw = _ring_signed_area_xy(ring_coords) > 0.0

    L = ring.length
    if L <= 1e-9:
        arc = [(entry_xy[0], entry_xy[1]), (exit_xy[0], exit_xy[1])]
        return arc, arc

    d_entry = ring.project(entry_pt)
    d_exit = ring.project(exit_pt)

    # Длина дуги при движении "вперёд" по параметру кольца.
    forward_len = (d_exit - d_entry) % L
    backward_len = (d_entry - d_exit) % L

    n_fwd = max(2, int(math.ceil(forward_len / max(1e-6, step_m))))
    n_bwd = max(2, int(math.ceil(backward_len / max(1e-6, step_m))))

    # forward: d = d_entry + t, t in [0, forward_len]
    fwd_arc: List[Tuple[float, float]] = []
    for i in range(n_fwd):
        t = forward_len * (i / (n_fwd - 1))
        d = (d_entry + t) % L
        p = ring.interpolate(d)
        fwd_arc.append((float(p.x), float(p.y)))

    # backward: d = d_entry - t, t in [0, backward_len]
    bwd_arc: List[Tuple[float, float]] = []
    for i in range(n_bwd):
        t = backward_len * (i / (n_bwd - 1))
        d = (d_entry - t) % L
        p = ring.interpolate(d)
        bwd_arc.append((float(p.x), float(p.y)))

    # Соотнесём "forward" с CCW или CW в зависимости от ориентации внешнего кольца.
    # Если ring_is_ccw=True, то forward соответствует CCW.
    if ring_is_ccw:
        counterclockwise_arc = fwd_arc
        clockwise_arc = bwd_arc
    else:
        clockwise_arc = fwd_arc
        counterclockwise_arc = bwd_arc
    return clockwise_arc, counterclockwise_arc


def _calculate_metric_path_length_m(path_xy: List[Tuple[float, float]]) -> float:
    """Считает длину метрической полилинии как сумму длин сегментов."""
    if not path_xy or len(path_xy) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path_xy) - 1):
        x1, y1 = path_xy[i]
        x2, y2 = path_xy[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    return float(total)


def boundary_detour_route_for_edge(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    no_fly_union: Any,
    *,
    safety_margin_m: float = 0.0,
    boundary_step_m: float = 25.0,
) -> Tuple[Optional[List[Tuple[float, float]]], Optional[float]]:
    """
    Строит обход вокруг (buffered) no-fly polygon по внешнему контуру.
    Возвращает (detour_coords_lonlat, detour_length_km) или (None, None) если не смогли.
    """
    if no_fly_union is None or getattr(no_fly_union, "is_empty", True):
        return None, None

    src_lon = float(lon1 + lon2) / 2.0
    src_lat = float(lat1 + lat2) / 2.0
    utm_epsg = _utm_epsg_from_lonlat(src_lon, src_lat)
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(utm_epsg)
    fwd = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    inv = Transformer.from_crs(dst_crs, src_crs, always_xy=True)

    start_xy = fwd.transform(lon1, lat1)
    goal_xy = fwd.transform(lon2, lat2)
    direct_line_xy = LineString([start_xy, goal_xy])

    zones_metric = unary_union([z for z in no_fly_union.geoms]) if hasattr(no_fly_union, "geoms") else no_fly_union
    zones_metric = zones_metric
    zones_metric = shapely_transform(fwd.transform, zones_metric)

    # Выбираем компонент, который пересекает прямую в metric.
    selected_safe_poly: Optional[Polygon] = None
    best_score = -1.0

    comps: List[Any]
    if isinstance(zones_metric, MultiPolygon):
        comps = list(zones_metric.geoms)
    elif isinstance(zones_metric, Polygon):
        comps = [zones_metric]
    else:
        return None, None

    for comp in comps:
        try:
            safe_comp = comp.buffer(safety_margin_m) if safety_margin_m and safety_margin_m != 0.0 else comp
            if safe_comp is None or getattr(safe_comp, "is_empty", True):
                continue
            if not direct_line_xy.intersects(safe_comp):
                continue
            inter = direct_line_xy.intersection(safe_comp)
            score = float(getattr(inter, "length", 0.0) or 0.0)
            # если пересечение точечное, используем дистанцию как score
            if score <= 1e-12:
                score = -float(direct_line_xy.distance(safe_comp))
            if score > best_score and isinstance(safe_comp, Polygon):
                best_score = score
                selected_safe_poly = safe_comp
        except Exception:
            continue

    if selected_safe_poly is None:
        # fallback: возьмём ближайшую компоненту
        for comp in comps:
            try:
                safe_comp = comp.buffer(safety_margin_m) if safety_margin_m and safety_margin_m != 0.0 else comp
                if safe_comp is None or getattr(safe_comp, "is_empty", True):
                    continue
                score = -float(direct_line_xy.distance(safe_comp))
                if score > best_score and isinstance(safe_comp, Polygon):
                    best_score = score
                    selected_safe_poly = safe_comp
            except Exception:
                continue

    if selected_safe_poly is None or not isinstance(selected_safe_poly, Polygon):
        return None, None

    # entry/exit на внешнем контуре
    # Чтобы маршрут был максимально коротким (как "правильный" естественный обход),
    # точки entry/exit берём из пересечения прямого сегмента с границей safe-полигона,
    # а не "по ближайшему" к start/goal.
    def _extract_points(g) -> List[Point]:
        if g is None or g.is_empty:
            return []
        gt = getattr(g, "geom_type", None)
        if gt == "Point":
            return [g]
        if gt == "MultiPoint":
            return list(g.geoms)
        if gt == "GeometryCollection":
            out: List[Point] = []
            for part in getattr(g, "geoms", []):
                out.extend(_extract_points(part))
            return out
        if gt in ("LineString", "LinearRing"):
            # Линия "вдоль границы": возьмём концы перекрытия как точки входа/выхода
            coords = list(g.coords)
            if len(coords) >= 2:
                return [Point(coords[0]), Point(coords[-1])]
        return []

    boundary_inter = direct_line_xy.intersection(selected_safe_poly.boundary)
    inter_points = _extract_points(boundary_inter)

    inter_points_sorted = sorted(inter_points, key=lambda p: direct_line_xy.project(p))

    entry_pt_metric: Optional[Point] = None
    exit_pt_metric: Optional[Point] = None

    # Выбираем пару соседних пересечений, между которыми линия реально проходит через interior safe-полигона.
    for i in range(len(inter_points_sorted) - 1):
        p1 = inter_points_sorted[i]
        p2 = inter_points_sorted[i + 1]
        # Считаем mid параметр-относительно, между p1 и p2 по координате вдоль прямого сегмента.
        proj1 = direct_line_xy.project(p1)
        proj2 = direct_line_xy.project(p2)
        mid_proj = (proj1 + proj2) / 2.0
        mid_pt = direct_line_xy.interpolate(mid_proj)
        try:
            if selected_safe_poly.contains(mid_pt):
                entry_pt_metric = p1
                exit_pt_metric = p2
                break
        except Exception:
            continue

    # Fallback: если пересечений оказалось недостаточно/не нашли interior-сегмент,
    # используем прежнюю nearest_visible_boundary_point логику.
    if entry_pt_metric is None or exit_pt_metric is None:
        entry_pt_metric = _nearest_visible_boundary_point_metric(start_xy, selected_safe_poly)
        exit_pt_metric = _nearest_visible_boundary_point_metric(goal_xy, selected_safe_poly)

    entry_xy = (float(entry_pt_metric.x), float(entry_pt_metric.y))
    exit_xy = (float(exit_pt_metric.x), float(exit_pt_metric.y))

    clockwise_arc, counterclockwise_arc = _build_boundary_arcs_metric(
        selected_safe_poly,
        entry_xy,
        exit_xy,
        step_m=boundary_step_m,
    )

    route_cw_metric = [start_xy, clockwise_arc[0]] + clockwise_arc[1:-1] + [clockwise_arc[-1], goal_xy]
    route_ccw_metric = [start_xy, counterclockwise_arc[0]] + counterclockwise_arc[1:-1] + [counterclockwise_arc[-1], goal_xy]

    len_cw_m = _calculate_metric_path_length_m(route_cw_metric)
    len_ccw_m = _calculate_metric_path_length_m(route_ccw_metric)

    chosen = route_cw_metric if len_cw_m <= len_ccw_m else route_ccw_metric
    chosen_len_km = (len_cw_m if len_cw_m <= len_ccw_m else len_ccw_m) / 1000.0

    # обратная проекция в lon/lat
    out_lonlat: List[Tuple[float, float]] = []
    for x, y in chosen:
        lon, lat = inv.transform(x, y)
        out_lonlat.append((float(lon), float(lat)))

    return out_lonlat, float(chosen_len_km)


def route_lonlat_segment_with_nfz_detours(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    nfz_union: Any,
    *,
    air_graph: Optional[nx.Graph] = None,
    air_coords: Optional[np.ndarray] = None,
    air_ids: Optional[List[Any]] = None,
    air_tree: Optional[Any] = None,
    no_fly_safety_buffer: float = 0.0,
    nfz_metric: Optional[Tuple[Any, Optional[Transformer]]] = None,
    utm_anchor_lonlat: Optional[Tuple[float, float]] = None,
    logger: Optional[logging.Logger] = None,
    detour_counter: Optional[List[int]] = None,
    log_edge_label: Optional[str] = None,
) -> Optional[Tuple[List[Tuple[float, float]], float, str]]:
    """
    Построение сегмента между двумя точками: прямая, либо кратчайший безопасный обход
    (контур no-fly и/или A* по воздушному графу), как на магистрали.

    Возвращает (полилиния lon/lat, длина км, режим: straight|boundary|air_astar) или None,
    если концы в no-fly или безопасный маршрут не найден.
    """
    d_km = haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))
    straight: List[Tuple[float, float]] = [(float(lon1), float(lat1)), (float(lon2), float(lat2))]

    if nfz_union is None or getattr(nfz_union, "is_empty", True):
        return (straight, float(d_km), "straight")

    if is_point_in_no_fly_zone((lon1, lat1), nfz_union, safety_buffer=no_fly_safety_buffer) or is_point_in_no_fly_zone(
        (lon2, lat2), nfz_union, safety_buffer=no_fly_safety_buffer
    ):
        return None

    nfz_union_metric: Optional[Any] = None
    to_metric: Optional[Transformer] = None
    if nfz_metric is not None:
        nfz_union_metric, to_metric = nfz_metric
    else:
        try:
            alon = float(utm_anchor_lonlat[0]) if utm_anchor_lonlat is not None else float(lon1 + lon2) / 2.0
            alat = float(utm_anchor_lonlat[1]) if utm_anchor_lonlat is not None else float(lat1 + lat2) / 2.0
            utm_epsg = _utm_epsg_from_lonlat(alon, alat)
            src_crs = CRS.from_epsg(4326)
            dst_crs = CRS.from_epsg(utm_epsg)
            to_metric = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            nfz_union_metric = shapely_transform(to_metric.transform, nfz_union)
        except Exception:
            nfz_union_metric = None
            to_metric = None

    ctr = detour_counter if detour_counter is not None else [0]
    log = logger or logging.getLogger(__name__)
    lbl = log_edge_label or "%s–%s" % (lon1, lon2)

    def _edge_blocked(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> bool:
        if nfz_union is None:
            return False
        try:
            if nfz_union_metric is not None and to_metric is not None:
                x1, y1 = to_metric.transform(float(lon_a), float(lat_a))
                x2, y2 = to_metric.transform(float(lon_b), float(lat_b))
                line = LineString([(float(x1), float(y1)), (float(x2), float(y2))])
                return is_edge_blocked(line, nfz_union_metric, safety_buffer=0.0)
            line = LineString([(float(lon_a), float(lat_a)), (float(lon_b), float(lat_b))])
            return is_edge_blocked(line, nfz_union, safety_buffer=no_fly_safety_buffer)
        except Exception:
            return False

    def _route_is_safe(route_coords: Optional[List[Tuple[float, float]]]) -> bool:
        if route_coords is None or len(route_coords) < 2:
            return False
        if nfz_union is None:
            return True
        try:
            for lon, lat in route_coords:
                if nfz_union_metric is not None and to_metric is not None:
                    x, y = to_metric.transform(float(lon), float(lat))
                    if is_point_in_no_fly_zone((float(x), float(y)), nfz_union_metric, safety_buffer=0.0):
                        return False
                else:
                    if is_point_in_no_fly_zone((float(lon), float(lat)), nfz_union, safety_buffer=no_fly_safety_buffer):
                        return False
            for i in range(len(route_coords) - 1):
                la, lo_a = route_coords[i]
                lb, lo_b = route_coords[i + 1]
                if _edge_blocked(float(la), float(lo_a), float(lb), float(lo_b)):
                    return False
            return True
        except Exception:
            return False

    def _route_length_km(route_coords: Optional[List[Tuple[float, float]]]) -> Optional[float]:
        if route_coords is None or len(route_coords) < 2:
            return None
        total = 0.0
        for i in range(len(route_coords) - 1):
            la, lo_a = route_coords[i]
            lb, lo_b = route_coords[i + 1]
            total += haversine_km(float(lo_a), float(la), float(lo_b), float(lb))
        return float(total)

    def _compress_route_greedy_visibility(
        route_coords: Optional[List[Tuple[float, float]]],
    ) -> Optional[List[Tuple[float, float]]]:
        if route_coords is None:
            return None
        if len(route_coords) <= 2:
            return route_coords
        if nfz_union is None:
            return [route_coords[0], route_coords[-1]]
        try:
            pts = [(float(lon), float(lat)) for lon, lat in route_coords]
            out: List[Tuple[float, float]] = [pts[0]]
            i = 0
            n = len(pts)
            while i < n - 1:
                best_j = i + 1
                for j in range(n - 1, i, -1):
                    if not _edge_blocked(pts[i][0], pts[i][1], pts[j][0], pts[j][1]):
                        best_j = j
                        break
                out.append(pts[best_j])
                i = best_j
            return out
        except Exception:
            return route_coords

    def _compress_route_min_length_dp(
        route_coords: Optional[List[Tuple[float, float]]],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Кратчайшая полилиния по подмножеству вершин исходной ломаной: ребро i→j допустимо,
        если прямой сегмент не пересекает no-fly. Жадный «максимальный прыжок» здесь не подходит —
        он не минимизирует суммарную длину.
        """
        if route_coords is None:
            return None
        if len(route_coords) <= 2:
            return route_coords
        if nfz_union is None:
            return [route_coords[0], route_coords[-1]]
        try:
            pts = [(float(lon), float(lat)) for lon, lat in route_coords]
            n = len(pts)
            if n <= 2:
                return route_coords
            INF = float("inf")
            dp = [INF] * n
            nxt = [-1] * n
            dp[n - 1] = 0.0
            for i in range(n - 2, -1, -1):
                best = INF
                best_j = -1
                for j in range(i + 1, n):
                    if _edge_blocked(pts[i][0], pts[i][1], pts[j][0], pts[j][1]):
                        continue
                    w = haversine_km(pts[i][1], pts[i][0], pts[j][1], pts[j][0])
                    if dp[j] >= INF:
                        continue
                    cand = w + dp[j]
                    if cand < best:
                        best = cand
                        best_j = j
                dp[i] = best
                nxt[i] = best_j
            if dp[0] >= INF:
                return _compress_route_greedy_visibility(route_coords)
            out: List[Tuple[float, float]] = [pts[0]]
            cur = 0
            while cur < n - 1:
                nj = nxt[cur]
                if nj < 0:
                    return _compress_route_greedy_visibility(route_coords)
                out.append(pts[nj])
                cur = nj
            return out
        except Exception:
            return _compress_route_greedy_visibility(route_coords)

    def _shortcut_visible_endpoints(
        route_coords: Optional[List[Tuple[float, float]]],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Подтягивает ломаную к короткому обходу препятствия (как «плотнее» к зоне), без широких дуг:
        с старта — хорда к самой дальней видимой вершине; к финишу — от ближайшей к началу точки,
        откуда виден финиш. Итерации до стабилизации (string-pulling с концов).
        """
        if route_coords is None or len(route_coords) < 3:
            return route_coords
        try:
            pts: List[List[float]] = [[float(lo), float(la)] for lo, la in route_coords]
            max_pass = max(24, len(pts) * 2)
            for _ in range(max_pass):
                if len(pts) < 3:
                    break
                changed = False
                # От старта: максимальный j, что сегмент p0–pj свободен → выкидываем p1..p{j-1}.
                best_j = -1
                for j in range(len(pts) - 1, 1, -1):
                    if not _edge_blocked(pts[0][0], pts[0][1], pts[j][0], pts[j][1]):
                        best_j = j
                        break
                if best_j > 1:
                    pts = [pts[0]] + pts[best_j:]
                    changed = True
                # К финишу: минимальный i, что pi–p_last свободен → выкидываем pi+1..p{n-2}.
                if len(pts) >= 3:
                    best_i = -1
                    for i in range(0, len(pts) - 2):
                        if not _edge_blocked(pts[i][0], pts[i][1], pts[-1][0], pts[-1][1]):
                            best_i = i
                            break
                    if best_i >= 0 and best_i < len(pts) - 2:
                        pts = pts[: best_i + 1] + [pts[-1]]
                        changed = True
                if not changed:
                    break
            return [(float(p[0]), float(p[1])) for p in pts]
        except Exception:
            return route_coords

    def _tighten_detour_route(
        route_coords: Optional[List[Tuple[float, float]]],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Сжимает обход к более короткому эквиваленту (подмножество вершин + string-pull с концов),
        чтобы траектория была ближе к длине прямой хорды, пока удаётся уменьшать число вершин.
        """
        cur = route_coords
        if cur is None or len(cur) < 2:
            return cur
        for _ in range(8):
            prev_n = len(cur)
            cur2 = _compress_route_min_length_dp(cur)
            if cur2 is None:
                break
            cur3 = _shortcut_visible_endpoints(cur2)
            cur = cur3 if cur3 is not None else cur2
            if len(cur) >= prev_n:
                break
        return cur

    def _iterative_boundary_detour(
        alon1: float,
        alat1: float,
        alon2: float,
        alat2: float,
        *,
        max_iters: int = 12,
    ) -> Tuple[Optional[List[Tuple[float, float]]], Optional[float]]:
        route: List[Tuple[float, float]] = [(float(alon1), float(alat1)), (float(alon2), float(alat2))]
        if nfz_union is None:
            d0 = haversine_km(float(alat1), float(alon1), float(alat2), float(alon2))
            return route, float(d0)

        def _first_blocked_idx(coords: List[Tuple[float, float]]) -> Optional[int]:
            for ii in range(len(coords) - 1):
                a = coords[ii]
                b = coords[ii + 1]
                if _edge_blocked(a[0], a[1], b[0], b[1]):
                    return ii
            return None

        for _ in range(max(1, int(max_iters))):
            bad_idx = _first_blocked_idx(route)
            if bad_idx is None:
                route = _compress_route_min_length_dp(route) or route
                if _route_is_safe(route):
                    total_km = 0.0
                    for kk in range(len(route) - 1):
                        la, lo_a = route[kk]
                        lb, lo_b = route[kk + 1]
                        total_km += haversine_km(lo_a, la, lo_b, lb)
                    return route, float(total_km)
                return None, None

            a_lon, a_lat = route[bad_idx]
            b_lon, b_lat = route[bad_idx + 1]
            local_coords, _ = boundary_detour_route_for_edge(
                a_lon,
                a_lat,
                b_lon,
                b_lat,
                nfz_union,
                safety_margin_m=0.0,
                boundary_step_m=NO_FLY_BOUNDARY_STEP_M,
            )
            if local_coords is None or len(local_coords) < 2:
                return None, None
            stitched = list(local_coords)
            route = route[:bad_idx] + stitched + route[bad_idx + 2 :]
            if len(route) > 200:
                return None, None

        return None, None

    def _astar_detour(alon1: float, alat1: float, alon2: float, alat2: float):
        if air_graph is None or air_tree is None or air_coords is None or not air_ids:
            return None, None
        H = air_graph
        ctr[0] += 1
        tmp_start = "air_tmp_start_%s" % ctr[0]
        tmp_goal = "air_tmp_goal_%s" % ctr[0]
        out_coords: Optional[List[Tuple[float, float]]] = None
        out_len: Optional[float] = None
        try:
            H.add_node(tmp_start, lon=float(alon1), lat=float(alat1), node_type="air_tmp")
            H.add_node(tmp_goal, lon=float(alon2), lat=float(alat2), node_type="air_tmp")
            grid_spacing_km = float(air_graph.graph.get("__grid_spacing_km", UAV_AIR_GRID_SPACING_KM))
            n_air = int(len(air_ids))
            k_keep = int(min(max(32, 72), n_air))
            k_query = int(min(n_air, max(k_keep * 4, 120)))
            # Уже привязка к сетке — меньше «крюков», ближе к короткой дуге вокруг препятствия.
            max_attach_km = max(2.0, grid_spacing_km * 12.0)

            def _attach_ranked(tmp_nid: str, x0: float, y0: float, ox: float, oy: float) -> None:
                """Рёбра к k_keep узлам с минимальной оценкой w(x0,n)+haversine(n→другой конец) — меньше ложных «крюков»."""
                if air_tree is None or air_coords is None:
                    return
                q = [float(x0), float(y0)]
                kq = min(k_query, n_air)
                res = air_tree.query(q, k=max(1, kq))
                if kq == 1:
                    idxs = [int(res[1])]
                else:
                    idxs = [int(ii) for ii in res[1]]
                scored: List[Tuple[float, float, Any]] = []
                for idx in idxs:
                    nid2 = air_ids[int(idx)]
                    d = H.nodes[nid2]
                    elon = d.get("lon") or d.get("x")
                    elat = d.get("lat") or d.get("y")
                    if elon is None or elat is None:
                        continue
                    if nfz_union is not None and _edge_blocked(float(x0), float(y0), float(elon), float(elat)):
                        continue
                    w_km = haversine_km(float(y0), float(x0), float(elat), float(elon))
                    if w_km <= 0 or w_km > max_attach_km:
                        continue
                    h_km = haversine_km(float(elat), float(elon), float(oy), float(ox))
                    scored.append((float(w_km + h_km), float(w_km), nid2))
                scored.sort(key=lambda t: t[0])
                for _sc, w_km, nid2 in scored[:k_keep]:
                    if not H.has_edge(tmp_nid, nid2):
                        H.add_edge(tmp_nid, nid2, weight=float(w_km), length=float(w_km), edge_type="air")

            _attach_ranked(tmp_start, alon1, alat1, alon2, alat2)
            _attach_ranked(tmp_goal, alon2, alat2, alon1, alat1)
            res = astar_path_safe(
                H,
                tmp_start,
                tmp_goal,
                nfz_union,
                safety_buffer=no_fly_safety_buffer,
                weight_attr="weight",
            )
            if res.get("status") == "success":
                coords = res.get("path_coords") or []
                length_km = res.get("path_length_km")
                if coords and length_km is not None:
                    out_coords = coords
                    out_len = float(length_km)
        except Exception:
            pass
        finally:
            for tn in (tmp_start, tmp_goal):
                try:
                    if H.has_node(tn):
                        H.remove_node(tn)
                except Exception:
                    pass
        if out_coords is None or out_len is None:
            return None, None
        return out_coords, out_len

    if not _edge_blocked(lon1, lat1, lon2, lat2):
        return (straight, float(d_km), "straight")

    log.debug("Сегмент пересекает беспилотную зону %s, подбираем безопасный обход", lbl)
    candidates: List[Tuple[float, List[Tuple[float, float]], str]] = []

    bd_coords, _bd_len = _iterative_boundary_detour(lon1, lat1, lon2, lat2)
    bd_coords = _tighten_detour_route(bd_coords)
    if bd_coords is not None and _route_is_safe(bd_coords):
        bd_km = _route_length_km(bd_coords)
        if bd_km is not None:
            candidates.append((bd_km, bd_coords, "boundary"))
    elif bd_coords is not None:
        log.debug("Гео-обход %s после сжатия небезопасен, пробуем A*", lbl)

    if air_graph is not None and air_tree is not None:
        ast_coords, _al = _astar_detour(lon1, lat1, lon2, lat2)
        ast_coords = _tighten_detour_route(ast_coords)
        if ast_coords is not None and _route_is_safe(ast_coords):
            ast_km = _route_length_km(ast_coords)
            if ast_km is not None:
                candidates.append((ast_km, ast_coords, "air_astar"))
        elif ast_coords is not None:
            log.warning("A* детур %s небезопасен после сжатия и отброшен", lbl)

    if not candidates:
        log.warning(
            "Обход не построен для %s (контур и A* не дали безопасного пути; скругление не при чём — оно ниже по коду)",
            lbl,
        )
        return None

    chord_ref = max(float(d_km), 1e-9)

    candidates.sort(
        key=lambda t: (
            (t[0] / chord_ref) + float(DETOUR_SCORE_EXCESS_PER_KM) * max(0.0, t[0] - chord_ref),
            t[0],
        )
    )
    detour_len, detour_coords, how = candidates[0]
    detour_coords = _tighten_detour_route(detour_coords)
    if detour_coords is not None:
        rl0 = _route_length_km(detour_coords)
        if rl0 is not None:
            detour_len = float(rl0)
    if how != "straight" and len(detour_coords) >= 3:
        if len(detour_coords) <= 6:
            # Для коротких A*/boundary обходов сначала уплотняем вершины, затем скругляем:
            # иначе Chaikin даёт недостаточно «мягкую» линию (мало контрольных точек).
            detour_coords = densify_lonlat_polyline_metric(
                detour_coords,
                float(utm_anchor_lonlat[0]) if utm_anchor_lonlat is not None else float(detour_coords[0][0]),
                float(utm_anchor_lonlat[1]) if utm_anchor_lonlat is not None else float(detour_coords[0][1]),
                max_step_m=320.0,
                max_output_points=36,
            )
        detour_coords = polish_detour_polyline(
            detour_coords,
            nfz_union,
            nfz_metric=nfz_metric,
            utm_anchor_lonlat=utm_anchor_lonlat,
            no_fly_safety_buffer=no_fly_safety_buffer,
        )
        # После Chaikin не делаем Douglas–Peucker simplify: он срезает точки и снова даёт острые углы.
        rl = _route_length_km(detour_coords)
        if rl is not None:
            detour_len = float(rl)
        detour_coords = light_extra_smoothing_lonlat(
            detour_coords,
            nfz_union,
            nfz_metric=nfz_metric,
            utm_anchor_lonlat=utm_anchor_lonlat,
            no_fly_safety_buffer=no_fly_safety_buffer,
        )
        rl2 = _route_length_km(detour_coords)
        if rl2 is not None:
            detour_len = float(rl2)
        # Не вызывать _tighten_detour_route после Chaikin: сжатие снова оставляет 3–4 угла и ломает скругление.
    ratio = float(detour_len) / chord_ref
    log.info(
        "Обход no-fly (%s): %s, хорда %.3f км, путь %.3f км (×%.2f к прямой), точек %s",
        how,
        lbl,
        float(d_km),
        float(detour_len),
        ratio,
        len(detour_coords),
    )
    return (detour_coords, float(detour_len), how)


def prepare_air_detour_auxiliary(
    obstacles: Any,
    air_graph: Optional[nx.Graph],
    utm_anchor_lonlat: Tuple[float, float],
    *,
    no_fly_safety_buffer: float = 0.0,
) -> Tuple[Any, Optional[Tuple[Any, Optional[Transformer]]], Optional[nx.Graph], Optional[np.ndarray], Optional[List[Any]], Optional[Any]]:
    """
    Общая подготовка для route_lonlat_segment_with_nfz_detours: no-fly в UTM, безопасный air_graph, KD-tree.
    Возвращает (nfz_union, nfz_metric_pair, air_work, air_coords, air_ids, air_tree).
    """
    nfz_union = None
    if obstacles is not None and getattr(obstacles, "is_empty", True) is False and getattr(obstacles, "is_valid", True):
        nfz_union = obstacles

    nfz_union_metric = None
    to_metric: Optional[Transformer] = None
    if nfz_union is not None:
        try:
            mid_lon = float(utm_anchor_lonlat[0])
            mid_lat = float(utm_anchor_lonlat[1])
            utm_epsg = _utm_epsg_from_lonlat(mid_lon, mid_lat)
            src_crs = CRS.from_epsg(4326)
            dst_crs = CRS.from_epsg(utm_epsg)
            to_metric = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            nfz_union_metric = shapely_transform(to_metric.transform, nfz_union)
        except Exception:
            nfz_union_metric = None
            to_metric = None

    air_work = air_graph
    if nfz_union is not None and air_work is not None and air_work.number_of_nodes() > 1:
        try:
            air_work = build_safe_graph(air_work, nfz_union, safety_buffer=no_fly_safety_buffer)
        except Exception:
            pass

    air_coords = None
    air_ids = None
    air_tree = None
    if air_work is not None and air_work.number_of_nodes() > 1:
        air_ids = list(air_work.nodes())
        pts_air = []
        for nid in air_ids:
            data = air_work.nodes[nid]
            lon = data.get("lon") or data.get("x")
            lat = data.get("lat") or data.get("y")
            if lon is None or lat is None:
                pts_air.append((np.nan, np.nan))
            else:
                pts_air.append((float(lon), float(lat)))
        air_coords = np.array(pts_air, dtype=float)
        valid_mask = ~np.isnan(air_coords[:, 0]) & ~np.isnan(air_coords[:, 1])
        if not np.any(valid_mask):
            air_coords = None
            air_ids = None
        else:
            air_coords = air_coords[valid_mask]
            air_ids = [air_ids[i] for i, ok in enumerate(valid_mask) if ok]
            try:
                air_tree = cKDTree(air_coords)
            except Exception:
                air_tree = None

    nfz_metric_pair: Optional[Tuple[Any, Optional[Transformer]]] = None
    if nfz_union_metric is not None and to_metric is not None:
        nfz_metric_pair = (nfz_union_metric, to_metric)

    return nfz_union, nfz_metric_pair, air_work, air_coords, air_ids, air_tree


class StationPlacement:
    def __init__(self, data_service, logger=None):
        """Инициализирует сервис размещения станций поверх подготовленных городских данных."""
        self.data_service = data_service
        self.logger = logger or logging.getLogger(__name__)

    def build_demand_points(
        self,
        buildings: gpd.GeoDataFrame,
        road_graph,
        city_boundary,
        *,
        method: str = "dbscan",
        cell_size_m: float = 250.0,
        dbscan_eps_m: float = 180.0,
        dbscan_min_samples: int = 15,
        use_all_buildings: bool = False,
        no_fly_zones=None,
        fill_clusters: bool = False,
        cluster_fill_step_m: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        """Строит точки спроса для размещения станций (DBSCAN/сетка) через DataService."""
        return self.data_service.get_demand_points_weighted(
            buildings,
            road_graph,
            city_boundary,
            method=method,
            cell_size_m=cell_size_m,
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
            use_all_buildings=use_all_buildings,
            no_fly_zones=no_fly_zones,
            fill_clusters=fill_clusters,
            cluster_fill_step_m=cluster_fill_step_m,
        )

    def build_charge_candidates_from_clusters(
        self,
        demand: gpd.GeoDataFrame,
        full_candidates: gpd.GeoDataFrame,
        *,
        max_distance_km: Optional[float] = None,
        candidates_per_cluster: int = 30,
    ) -> gpd.GeoDataFrame:
        """
        Строит кандидатов зарядки относительно кластеров DBSCAN: для каждого кластера (центроид спроса)
        выбираются ближайшие точки из `full_candidates` (крыши/площадки).

        В режиме “без радиусов” ограничение `max_distance_km` обычно выключено
        (тогда выбираются ближайшие без дальностного отсечения).
        Возвращает уникальный набор кандидатов — по одному месту на кластер (лучшая позиция для кластера).
        """
        if demand is None or len(demand) == 0 or full_candidates is None or len(full_candidates) == 0:
            return full_candidates if full_candidates is not None else gpd.GeoDataFrame()

        max_dist = max_distance_km if max_distance_km is not None else float("inf")
        dem_pts = _points_array(demand)
        cand_pts = _points_array(full_candidates)
        chosen_indices = set()
        for i in range(len(dem_pts)):
            lat_d, lon_d = dem_pts[i, 1], dem_pts[i, 0]
            n_for_cluster = max(1, int(candidates_per_cluster))

            dists = []
            for j in range(len(cand_pts)):
                d_km = haversine_km(lat_d, lon_d, cand_pts[j, 1], cand_pts[j, 0])
                if d_km <= max_dist:
                    dists.append((j, d_km))
            if not dists:
                continue
            dists.sort(key=lambda t: t[1])
            added = 0
            for j, _ in dists:
                if j in chosen_indices:
                    continue
                chosen_indices.add(j)
                added += 1
                if added >= n_for_cluster:
                    break
        if not chosen_indices:
            self.logger.warning("Нет кандидатов при заданном max_distance_km — используем все кандидаты")
            return full_candidates
        out = full_candidates.iloc[sorted(chosen_indices)].copy()
        self.logger.info(
            f"Кандидаты зарядки относительно кластеров: {len(out)} из {len(full_candidates)} (по кластерам DBSCAN)"
        )
        return out

    def place_charging_stations(
        self,
        candidates: gpd.GeoDataFrame,
        demand: gpd.GeoDataFrame,
        no_fly_zones,
        *,
        radius_km: float = R_CHARGE_KM,
        max_stations: Optional[int] = None,
        type_a_coverage_ratio: float = 0.95,
        enable_type_b: bool = False,
        type_b_coverage_ratio: float = 1.0,
        min_center_distance_factor_a: float = 1.0,
        min_center_distance_factor_b: float = 2.0,
        min_center_distance_factor_core_a: Optional[float] = None,
        core_dense_radius_factor: float = 5.0,
        core_radius_factor_a: float = 4.0,
        core_weight_quantile_a: float = 0.7,
        min_core_gain_ratio_a: float = 0.01,
        random_state: Optional[int] = None,
        force_type_b_per_cluster: bool = False,
        force_type_a_per_region: bool = False,
        force_type_a_per_cluster: bool = False,
        cluster_id_col: str = "subcluster_id",
        region_id_col: str = "region_id",
        cluster_centroid_is_fill_col: str = "is_cluster_fill",
        mandatory_a_max_distance_km: Optional[float] = None,
        mandatory_b_max_distance_km: Optional[float] = None,
        skip_greedy_after_forced: bool = True,
        # При наличии hull-полигонов кластера выбираем кандидатов строго внутри "своей" геометрии,
        # чтобы станция не оказалась внутри/на пересечении hulls соседних кластеров.
        cluster_hulls: Optional[gpd.GeoDataFrame] = None,
        cluster_hulls_id_col: str = "cluster_id",
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Два типа зарядок:
        - Тип А: покрывают основное «ядро» спроса (type_a_coverage_ratio, по умолчанию 85%).
          Размещаются только в пределах ядра: не далее core_radius_factor_a * radius_km
          от взвешенного центра спроса, и не ближе min_center_distance_factor_a * radius_km
          друг к другу (круги в основном касаются).
        - Тип Б (опционально): добивают покрытие до 100% и помогают при развозке заказов.
        Жадно: на каждом шаге — кандидат с максимальным приростом целевой функции.
        """
        if candidates is None or len(candidates) == 0 or demand is None or len(demand) == 0:
            return gpd.GeoDataFrame(), {
                "placed": 0,
                "placed_type_a": 0,
                "placed_type_b": 0,
                "placed_extra": 0,
                "demand_covered": 0,
                "demand_total": 0,
                "coverage_ratio": 0.0,
            }

        cand_pts = _points_array(candidates)
        dem_pts = _points_array(demand)
        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
        n_demand = len(dem_pts)
        total_weight = float(weights.sum())
        covered = np.zeros(n_demand, dtype=bool)
        selected: List[dict] = []
        selected_indices: List[int] = []
        selected_types: List[str] = []
        rng = np.random.default_rng(random_state)

        core_lon = float(np.average(dem_pts[:, 0], weights=weights)) if n_demand > 0 else 0.0
        core_lat = float(np.average(dem_pts[:, 1], weights=weights)) if n_demand > 0 else 0.0
        max_core_dist_a_km = max(0.0, float(core_radius_factor_a)) * radius_km

        if n_demand > 0:
            if core_weight_quantile_a is None or core_weight_quantile_a <= 0.0:
                is_core_demand = np.ones(n_demand, dtype=bool)
            else:
                if 0.0 < core_weight_quantile_a < 1.0:
                    try:
                        w_thr = float(np.quantile(weights, core_weight_quantile_a))
                    except Exception:
                        w_thr = float(np.median(weights))
                else:
                    w_thr = float(np.median(weights))
                is_core_demand = weights >= w_thr
            total_core_weight = float(weights[is_core_demand].sum())
        else:
            is_core_demand = np.zeros(0, dtype=bool)
            total_core_weight = 0.0

        core_dense_radius_km = max(0.0, float(core_dense_radius_factor)) * radius_km
        core_min_factor_a = (
            float(min_center_distance_factor_core_a)
            if min_center_distance_factor_core_a is not None
            else min_center_distance_factor_a
        )

        def _too_close_to_existing(idx: int, station_type: str) -> bool:
            if not selected_indices:
                return False
            lon_c, lat_c = cand_pts[idx, 0], cand_pts[idx, 1]
            if station_type == "charge_a":
                if core_dense_radius_km > 0:
                    d_to_core = haversine_km(lat_c, lon_c, core_lat, core_lon)
                    min_factor = (
                        max(0.0, float(core_min_factor_a))
                        if d_to_core <= core_dense_radius_km
                        else max(0.0, float(min_center_distance_factor_a))
                    )
                else:
                    min_factor = max(0.0, float(min_center_distance_factor_a))
            else:
                min_factor = max(0.0, float(min_center_distance_factor_b))
            if min_factor <= 0.0:
                return False
            min_allowed = radius_km * min_factor
            for s_idx in selected_indices:
                lon_s, lat_s = cand_pts[s_idx, 0], cand_pts[s_idx, 1]
                if haversine_km(lat_c, lon_c, lat_s, lon_s) < min_allowed:
                    return True
            return False

        def run_phase(station_type: str, stop_at_ratio: float, ignore_covered_in_score: bool = False) -> None:
            nonlocal covered, selected, selected_indices
            while True:
                best_idx = -1
                best_new_weight = 0.0
                top_indices: List[int] = []
                for i in range(len(cand_pts)):
                    if i in selected_indices:
                        continue
                    if station_type in ("charge_a", "charge_b") and "source" in candidates.columns:
                        if candidates.iloc[i].get("source") != "building":
                            continue
                    lon_c, lat_c = cand_pts[i, 0], cand_pts[i, 1]
                    if station_type == "charge_a" and max_core_dist_a_km > 0.0:
                        d_core = haversine_km(lat_c, lon_c, core_lat, core_lon)
                        if d_core > max_core_dist_a_km:
                            continue
                    if _too_close_to_existing(i, station_type):
                        continue
                    new_weight = 0.0
                    for j in range(n_demand):
                        if station_type == "charge_a" and not is_core_demand[j]:
                            continue
                        if not ignore_covered_in_score and covered[j]:
                            continue
                        d_km = haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0])
                        if d_km <= radius_km:
                            new_weight += weights[j]
                    if new_weight <= 0:
                        continue
                    if new_weight > best_new_weight:
                        best_new_weight = new_weight
                        best_idx = i
                        top_indices = [i]
                    elif best_new_weight > 0 and new_weight >= 0.95 * best_new_weight:
                        top_indices.append(i)
                if best_new_weight <= 0:
                    break
                if top_indices:
                    best_idx = int(rng.choice(top_indices))
                elif best_idx < 0:
                    break
                selected_indices.append(best_idx)
                selected_types.append(station_type)
                lon_c, lat_c = cand_pts[best_idx, 0], cand_pts[best_idx, 1]
                for j in range(n_demand):
                    d_km = haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0])
                    if d_km <= radius_km:
                        if station_type == "charge_a" and not is_core_demand[j]:
                            continue
                        if not covered[j]:
                            covered[j] = True
                selected.append(
                    {
                        "geometry": Point(cand_pts[best_idx, 0], cand_pts[best_idx, 1]),
                        "station_type": station_type,
                        "source_index": best_idx,
                    }
                )
                if max_stations is not None and len(selected) >= max_stations:
                    return
                ratio = (weights[covered].sum() / total_weight) if total_weight > 0 else 0.0
                if ratio >= stop_at_ratio:
                    return

        # --- Принудительная логика по районам ---
        # Новая логика пользователя:
        # 1) В каждом районе выделяем ОДИН главный большой кластер (самый тяжёлый).
        # 2) Станция для магистрали trunk ставится только в главный кластер (charge_a).
        # 3) В главный кластер станцию charge_b НЕ ставим (если включён режим charge_b по кластерам).
        # 4) Чтобы не добавлялось “лишнее” жадным алгоритмом — можно пропустить greedy-фазы после forced-размещения.
        cluster_col = None
        if force_type_b_per_cluster:
            if cluster_id_col in demand.columns:
                cluster_col = cluster_id_col
            elif "cluster_id" in demand.columns:
                cluster_col = "cluster_id"

        region_col = None
        if force_type_a_per_region and region_id_col in demand.columns:
            region_col = region_id_col

        forced_activated = bool(region_col is not None and force_type_a_per_region)
        if forced_activated or (force_type_b_per_cluster and cluster_col is not None):
            centroid_df = demand
            if cluster_centroid_is_fill_col in centroid_df.columns:
                centroid_df = centroid_df[~centroid_df[cluster_centroid_is_fill_col].fillna(False).astype(bool)].copy()

            if centroid_df is not None and len(centroid_df) > 0:
                used_candidate_indices: set[int] = set()
                placed_cluster_ids: set[Any] = set()
                # Счётчики случаев, когда внутри hull кластера не нашлось кандидатов.
                # Тогда forced-логика делает fallback и выбирает ближайший кандидат "вообще".
                fallback_allowed_a_from_empty_hull = 0
                fallback_allowed_b_from_empty_hull = 0

                # Предрасчет принадлежности кандидатов hull'ам кластеров.
                # Правило:
                # - если точка попала ровно в один hull -> membership = cluster_id этого hull
                # - если в ноль hull или в несколько hull -> membership = None
                # Это гарантирует, что станция "строго" будет принадлежать hull геометрии своего кластера,
                # а в случае пересечений hull не будет размещена в неоднозначной области.
                candidate_membership: List[Optional[Any]] = [None] * len(cand_pts)
                candidate_covered_any: List[bool] = [False] * len(cand_pts)
                candidates_by_cluster: Dict[Any, List[int]] = {}
                hull_geom_by_cluster: Dict[Any, Any] = {}
                outside_no_hull_indices: List[int] = list(range(len(cand_pts)))

                if (
                    cluster_hulls is not None
                    and len(cluster_hulls) > 0
                    and cluster_hulls_id_col in cluster_hulls.columns
                    and "geometry" in cluster_hulls.columns
                ):
                    try:
                        hulls_gdf = cluster_hulls.copy()
                        hulls_gdf = hulls_gdf[~hulls_gdf["geometry"].isna() & ~hulls_gdf["geometry"].is_empty].copy()
                        if len(hulls_gdf) > 0:
                            for _, hrow in hulls_gdf.iterrows():
                                cid_h = hrow.get(cluster_hulls_id_col)
                                geom_h = hrow.geometry
                                if cid_h is None or geom_h is None or geom_h.is_empty:
                                    continue
                                hull_geom_by_cluster[cid_h] = geom_h

                            sindex = getattr(hulls_gdf, "sindex", None)
                            for ci in range(len(cand_pts)):
                                lon_c, lat_c = cand_pts[ci, 0], cand_pts[ci, 1]
                                pt = Point(lon_c, lat_c)
                                idxs = None
                                if sindex is not None:
                                    try:
                                        # Для spatial index запрос идёт "геометрии индекса относительно точки".
                                        # predicate="covers" здесь почти всегда даёт пусто; берём intersects,
                                        # затем точно проверяем geom.covers(pt) ниже.
                                        idxs = list(sindex.query(pt, predicate="intersects"))
                                    except Exception:
                                        idxs = None
                                if idxs is None:
                                    idxs = range(len(hulls_gdf))

                                match_cids: List[int] = []
                                for hi in idxs:
                                    geom = hulls_gdf.geometry.iloc[int(hi)]
                                    if geom is None or not geom.is_valid:
                                        continue
                                    try:
                                        if bool(geom.covers(pt)):
                                            match_cids.append(int(hulls_gdf.iloc[int(hi)][cluster_hulls_id_col]))
                                    except Exception:
                                        # В редких случаях shapely может падать на covers/contains
                                        try:
                                            if bool(geom.contains(pt)):
                                                match_cids.append(int(hulls_gdf.iloc[int(hi)][cluster_hulls_id_col]))
                                        except Exception:
                                            pass

                                if len(match_cids) == 0:
                                    candidate_covered_any[ci] = False
                                    continue

                                candidate_covered_any[ci] = True
                                unique_cids = set(match_cids)
                                if len(unique_cids) == 1:
                                    cid_val = next(iter(unique_cids))
                                    candidate_membership[ci] = cid_val
                                    candidates_by_cluster.setdefault(cid_val, []).append(ci)

                                # ambiguous (multiple hulls) остаётся membership=None

                            outside_no_hull_indices = [i for i, covered in enumerate(candidate_covered_any) if not covered]
                    except Exception:
                        # Если hull-based принадлежность не сработала — оставляем fallback без строгой геометрии.
                        candidate_membership = [None] * len(cand_pts)
                        outside_no_hull_indices = list(range(len(cand_pts)))
                        candidates_by_cluster = {}

                # Радиусы отсутствуют: покрытие demand-точек делаем по cluster_id.
                # Станция в кластере покрывает все demand-точки этого кластера.
                demand_cluster_vals_for_coverage = None
                if cluster_col is not None and cluster_col in demand.columns:
                    demand_cluster_vals_for_coverage = demand[cluster_col].values

                def _update_covered(served_cluster_id: Any) -> None:
                    if demand_cluster_vals_for_coverage is None:
                        return
                    # Важно: работаем через срез, чтобы не было augmented-assignment на переменную
                    # в замыкании (иначе возможен UnboundLocalError).
                    covered[:] = covered | (demand_cluster_vals_for_coverage == served_cluster_id)

                def _place_one_station(
                    station_type: str, cand_idx: int, *, served_cluster_id: Any
                ) -> None:
                    if served_cluster_id is not None and served_cluster_id in placed_cluster_ids:
                        return
                    selected_indices.append(cand_idx)
                    selected_types.append(station_type)
                    used_candidate_indices.add(cand_idx)
                    selected.append(
                        {
                            "geometry": Point(cand_pts[cand_idx, 0], cand_pts[cand_idx, 1]),
                            "station_type": station_type,
                            "source_index": cand_idx,
                            "cluster_id": served_cluster_id,
                        }
                    )
                    if served_cluster_id is not None:
                        placed_cluster_ids.add(served_cluster_id)
                    _update_covered(served_cluster_id)

                def _place_one_station_at_point(
                    station_type: str, lon: float, lat: float, *, served_cluster_id: Any
                ) -> None:
                    if served_cluster_id is not None and served_cluster_id in placed_cluster_ids:
                        return
                    selected_types.append(station_type)
                    selected.append(
                        {
                            "geometry": Point(float(lon), float(lat)),
                            "station_type": station_type,
                            "source_index": -1,
                            "cluster_id": served_cluster_id,
                        }
                    )
                    if served_cluster_id is not None:
                        placed_cluster_ids.add(served_cluster_id)
                    _update_covered(served_cluster_id)

                def _choose_nearest_candidate(
                    lon_target: float,
                    lat_target: float,
                    candidate_indices: List[int],
                    *,
                    max_dist_km: float,
                ) -> Optional[int]:
                    # Сначала среди неиспользованных в пределах max_dist_km.
                    best = None
                    best_d = float("inf")
                    for ci in candidate_indices:
                        if ci in used_candidate_indices:
                            continue
                        lon_i, lat_i = cand_pts[ci, 0], cand_pts[ci, 1]
                        d = haversine_km(lat_target, lon_target, lat_i, lon_i)
                        if d <= max_dist_km and d < best_d:
                            best_d = d
                            best = ci
                    if best is not None:
                        return int(best)
                    # Fallback: ближайший неиспользованный вообще (без ограничения).
                    for ci in candidate_indices:
                        if ci in used_candidate_indices:
                            continue
                        lon_i, lat_i = cand_pts[ci, 0], cand_pts[ci, 1]
                        d = haversine_km(lat_target, lon_target, lat_i, lon_i)
                        if d < best_d:
                            best_d = d
                            best = ci
                    if best is not None:
                        return int(best)
                    # Последний fallback: ближайший из любых.
                    for ci in candidate_indices:
                        lon_i, lat_i = cand_pts[ci, 0], cand_pts[ci, 1]
                        d = haversine_km(lat_target, lon_target, lat_i, lon_i)
                        if d < best_d:
                            best_d = d
                            best = ci
                    return int(best) if best is not None else None

                def _place_at_cluster_hull_if_possible(station_type: str, cid: Any) -> bool:
                    """Ставит станцию в заведомо внутреннюю точку hull кластера (если геометрия доступна)."""
                    try:
                        geom_h = hull_geom_by_cluster.get(cid)
                        if geom_h is None or getattr(geom_h, "is_empty", True):
                            return False
                        rp = geom_h.representative_point()
                        if rp is None or getattr(rp, "is_empty", True):
                            return False
                        _place_one_station_at_point(
                            station_type,
                            float(rp.x),
                            float(rp.y),
                            served_cluster_id=cid,
                        )
                        return True
                    except Exception:
                        return False

                # Вычисляем главный (A) кластер по району: центральный кластер,
                # т.е. ближайший к взвешенному центру района.
                main_cluster_by_region: Dict[Any, Any] = {}
                if region_col is not None and cluster_col is not None and cluster_col in centroid_df.columns:
                    for rid, sub_r in centroid_df.groupby(region_col):
                        if len(sub_r) == 0:
                            continue
                        # Взвешенный центр района.
                        w_r = sub_r["weight"].values if "weight" in sub_r.columns else np.ones(len(sub_r))
                        coords_r = np.array([_coords_from_geom(g) for g in sub_r.geometry], dtype=float)  # lon, lat
                        if len(coords_r) == 0:
                            continue
                        lon_r = float(np.average(coords_r[:, 0], weights=w_r))
                        lat_r = float(np.average(coords_r[:, 1], weights=w_r))

                        # Главный кластер = ближайший к центру района;
                        # при равенстве расстояния — более тяжёлый.
                        best_cid = None
                        best_d = float("inf")
                        best_w = -1.0
                        for cid, sub_c in sub_r.groupby(cluster_col):
                            if len(sub_c) == 0:
                                continue
                            w_c = float(sub_c["weight"].values[0]) if "weight" in sub_c.columns else float(len(sub_c))
                            # centroid точки кластера:
                            lon_c, lat_c = _coords_from_geom(sub_c.iloc[0].geometry)
                            d_c = haversine_km(lat_r, lon_r, lat_c, lon_c)
                            if d_c < best_d or (d_c == best_d and w_c > best_w):
                                best_d = d_c
                                best_w = w_c
                                best_cid = cid
                        if best_cid is not None:
                            main_cluster_by_region[rid] = best_cid

                # 1a) charge_a: по каждому кластеру ставим ровно 1 станцию в центр кластера.
                if cluster_col is not None and force_type_a_per_cluster:
                    max_a_dist = (
                        mandatory_a_max_distance_km if mandatory_a_max_distance_km is not None else float("inf")
                    )
                    allowed_a = list(range(len(cand_pts)))
                    if "source" in candidates.columns:
                        allowed_a = [i for i in allowed_a if candidates.iloc[i].get("source") == "building"]

                    for cid, sub_c in centroid_df.groupby(cluster_col):
                        if len(sub_c) == 0:
                            continue
                        if cid in placed_cluster_ids:
                            continue
                        lon_c, lat_c = _coords_from_geom(sub_c.iloc[0].geometry)
                        # Берем все точки кандидатов внутри hull этого кластера.
                        target_allowed_a = [i for i in candidates_by_cluster.get(cid, []) if i in allowed_a]
                        if not target_allowed_a:
                            # Если внутри hull нет кандидатов МКД — ставим точку в representative_point hull,
                            # чтобы не "уводить" кластерную станцию в соседний кластер.
                            if _place_at_cluster_hull_if_possible("charge_a", cid):
                                continue
                            # Если hull недоступен, берём ближайший МКД-кандидат.
                            fallback_allowed_a_from_empty_hull += 1
                            target_allowed_a = allowed_a
                            if not target_allowed_a:
                                continue

                        cand_idx = _choose_nearest_candidate(
                            lon_c,
                            lat_c,
                            target_allowed_a,
                            max_dist_km=max_a_dist,
                        )
                        if cand_idx is None:
                            continue
                        _place_one_station("charge_a", cand_idx, served_cluster_id=cid)

                # 1b) charge_a: по району ставим ровно 1 станцию в главный кластер.
                elif region_col is not None and force_type_a_per_region and cluster_col is not None:
                    max_a_dist = (
                        mandatory_a_max_distance_km if mandatory_a_max_distance_km is not None else float("inf")
                    )
                    allowed_a = list(range(len(cand_pts)))
                    if "source" in candidates.columns:
                        allowed_a = [i for i in allowed_a if candidates.iloc[i].get("source") == "building"]

                    for rid, sub_r in centroid_df.groupby(region_col):
                        if len(sub_r) == 0:
                            continue
                        if rid not in main_cluster_by_region:
                            continue
                        main_cid = main_cluster_by_region[rid]
                        if main_cid in placed_cluster_ids:
                            continue

                        # Строго: кандидат должен лежать внутри hull своего main_cid кластера.
                        # Если кандидатов внутри hull не нашлось — станцию не ставим, чтобы не нарушить правило
                        # "нельзя допускать, чтобы станция оказывалась ближе/внутри другого кластера".
                        target_allowed_a = [i for i in candidates_by_cluster.get(main_cid, []) if i in allowed_a]
                        if not target_allowed_a:
                            # Если внутри hull главного кластера нет МКД-кандидатов —
                            # ставим точку в representative_point этого hull.
                            if _place_at_cluster_hull_if_possible("charge_a", main_cid):
                                continue
                            # Если hull недоступен, берём ближайший МКД-кандидат.
                            fallback_allowed_a_from_empty_hull += 1
                            target_allowed_a = allowed_a
                            if not target_allowed_a:
                                continue

                        sub_main = sub_r[sub_r[cluster_col] == main_cid]
                        if len(sub_main) == 0:
                            continue
                        lon_m, lat_m = _coords_from_geom(sub_main.iloc[0].geometry)
                        cand_idx = _choose_nearest_candidate(
                            lon_m,
                            lat_m,
                            target_allowed_a,
                            max_dist_km=max_a_dist,
                        )
                        if cand_idx is None:
                            continue
                        _place_one_station("charge_a", cand_idx, served_cluster_id=main_cid)

                # 2) charge_b: по кластерам, но НЕ в главный кластер каждого района.
                if cluster_col is not None and force_type_b_per_cluster:
                    max_b_dist = (
                        mandatory_b_max_distance_km if mandatory_b_max_distance_km is not None else float("inf")
                    )
                    allowed_b = list(range(len(cand_pts)))
                    if "source" in candidates.columns:
                        allowed_b = [i for i in allowed_b if candidates.iloc[i].get("source") == "building"]
                    for cid, sub_c in centroid_df.groupby(cluster_col):
                        if len(sub_c) == 0:
                            continue
                        if cid in placed_cluster_ids:
                            continue
                        # Главный кластер региона: если A уже поставлен, cid ∈ placed_cluster_ids и мы выше вышли.
                        # Если A не удалось поставить — даём станцию B, иначе кластер остаётся без зарядки.
                        lon_c, lat_c = _coords_from_geom(sub_c.iloc[0].geometry)

                        # Строго: кандидат должен лежать внутри hull своего cid кластера.
                        target_allowed_b = [i for i in candidates_by_cluster.get(cid, []) if i in allowed_b]
                        if not target_allowed_b:
                            # Если внутри hull нет МКД-кандидатов — ставим точку в representative_point hull.
                            if _place_at_cluster_hull_if_possible("charge_b", cid):
                                continue
                            # Если hull недоступен, берём ближайший МКД-кандидат.
                            fallback_allowed_b_from_empty_hull += 1
                            target_allowed_b = allowed_b
                            if not target_allowed_b:
                                continue

                        cand_idx = _choose_nearest_candidate(
                            lon_c,
                            lat_c,
                            target_allowed_b,
                            max_dist_km=max_b_dist,
                        )
                        if cand_idx is None:
                            continue
                        _place_one_station("charge_b", cand_idx, served_cluster_id=cid)

                # Не выходим с пустым результатом: если forced-ветки ничего не поставили
                # (нет cluster_id в demand, пустые allowed_* и т.д.), продолжаем жадные фазы.
                if skip_greedy_after_forced and len(selected) > 0 and (
                    force_type_a_per_region or force_type_b_per_cluster or force_type_a_per_cluster
                ):
                    gdf = gpd.GeoDataFrame(selected, crs=candidates.crs) if selected else gpd.GeoDataFrame()
                    covered_weight = float(weights[covered].sum())
                    n_a = sum(1 for r in selected if r["station_type"] == "charge_a")
                    n_b = sum(1 for r in selected if r["station_type"] == "charge_b")
                    metrics = {
                        "placed": len(selected),
                        "placed_type_a": n_a,
                        "placed_type_b": n_b,
                        "placed_extra": n_b,
                        "demand_covered": covered_weight,
                        "demand_total": total_weight,
                        "coverage_ratio": covered_weight / total_weight if total_weight > 0 else 0.0,
                        "fallback_allowed_a_from_empty_hull": fallback_allowed_a_from_empty_hull,
                        "fallback_allowed_b_from_empty_hull": fallback_allowed_b_from_empty_hull,
                    }
                    self.logger.info(
                        f"[forced] Зарядки: тип А {n_a}, тип Б {n_b}, покрытие {metrics['coverage_ratio']:.2%}, "
                        f"fallback A(empty hull)={fallback_allowed_a_from_empty_hull}, fallback B(empty hull)={fallback_allowed_b_from_empty_hull}"
                    )
                    return gdf, metrics

        # Фаза 1: зарядки типа А — покрывают основную часть спроса.
        # В оценке кандидата берём весь спрос в радиусе (ignore_covered_in_score=True),
        # чтобы станции А «обнимали» основную зону спроса, а не разлетались по мелким островкам.
        run_phase("charge_a", type_a_coverage_ratio, ignore_covered_in_score=True)

        # Фаза 2: станции типа Б (дополняют покрытие оставшегося спроса).
        if enable_type_b and (max_stations is None or len(selected) < max_stations):
            run_phase("charge_b", type_b_coverage_ratio, ignore_covered_in_score=False)

        # Резерв: если жёсткие ограничения/forced ничего не дали, но МКД-кандидаты и спрос есть —
        # ставим минимум одну А в лучшей точке (или у ближайшего МКД к центру спроса).
        if not selected and len(cand_pts) > 0 and n_demand > 0:
            building_indices = list(range(len(cand_pts)))
            if "source" in candidates.columns:
                building_indices = [
                    i for i in building_indices if candidates.iloc[i].get("source") == "building"
                ]
            best_i = -1
            best_cov = -1.0
            for i in building_indices:
                lon_c, lat_c = cand_pts[i, 0], cand_pts[i, 1]
                cov = 0.0
                for j in range(n_demand):
                    if not is_core_demand[j]:
                        continue
                    d_km = haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0])
                    if d_km <= radius_km:
                        cov += float(weights[j])
                if cov > best_cov:
                    best_cov = cov
                    best_i = i
            if best_i < 0:
                best_d = float("inf")
                for i in building_indices:
                    lon_c, lat_c = cand_pts[i, 0], cand_pts[i, 1]
                    d = haversine_km(lat_c, lon_c, core_lat, core_lon)
                    if d < best_d:
                        best_d = d
                        best_i = i
            if best_i >= 0:
                self.logger.warning(
                    "Зарядки: основной алгоритм не поставил станций — резервная одна станция типа А (индекс кандидата %s)",
                    best_i,
                )
                lon_c, lat_c = cand_pts[best_i, 0], cand_pts[best_i, 1]
                selected_indices.append(best_i)
                selected_types.append("charge_a")
                for j in range(n_demand):
                    d_km = haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0])
                    if d_km <= radius_km:
                        if not is_core_demand[j]:
                            continue
                        if not covered[j]:
                            covered[j] = True
                selected.append(
                    {
                        "geometry": Point(cand_pts[best_i, 0], cand_pts[best_i, 1]),
                        "station_type": "charge_a",
                        "source_index": best_i,
                    }
                )

        gdf = gpd.GeoDataFrame(selected, crs=candidates.crs) if selected else gpd.GeoDataFrame()
        covered_weight = float(weights[covered].sum())
        n_a = sum(1 for r in selected if r["station_type"] == "charge_a")
        n_b = sum(1 for r in selected if r["station_type"] == "charge_b")
        metrics = {
            "placed": len(selected),
            "placed_type_a": n_a,
            "placed_type_b": n_b,
            "placed_extra": n_b,
            "demand_covered": covered_weight,
            "demand_total": total_weight,
            "coverage_ratio": covered_weight / total_weight if total_weight > 0 else 0.0,
        }
        self.logger.info(
            f"Зарядки: тип А {n_a}, тип Б {n_b}, покрытие {metrics['coverage_ratio']:.2%}"
        )
        return gdf, metrics

    def build_trunk_graph(
        self,
        charge_stations: gpd.GeoDataFrame,
        garage_points: gpd.GeoDataFrame,
        to_points: gpd.GeoDataFrame,
        obstacles,
        air_graph: Optional[nx.Graph] = None,
        *,
        obstacles_relaxed: Any = None,
        max_edge_km: float = R_GARAGE_TO_KM,
        max_neighbors_a: int = 4,
        topology_mode: str = "mst",
        extra_paths_per_a: int = 2,
        no_fly_safety_buffer: float = 0.0,
    ) -> nx.Graph:
        """
        Магистраль (спина сети): только станции типа А (зарядки А), полёт на эшелонах 4–5.
        Гаражи и ТО в магистраль не входят — к ближайшим А ведут ветки (см. branch_edges в пайплайне).

        Рёбра магистрали соединяют каждую станцию А с ближайшими соседями-А в пределах max_edge_km (км),
        не более 2 рёбер на станцию (даже если max_neighbors_a > 2).

        Если прямая связь между станциями пересекает препятствия (бесполётные зоны / высокие здания
        для эшелонов 4–5), и есть предварительно построенный
        воздушный граф air_graph, то вместо прямого отрезка строится обход по воздушному графу
        (A* по весу рёбер). Полученная полилиния сохраняется в атрибуте geometry_coords ребра.
        """
        nodes_list = []
        trunk_nodes = []
        for i, row in (charge_stations.iterrows() if charge_stations is not None and len(charge_stations) > 0 else []):
            if row.get("station_type") != "charge_a":
                continue
            lon, lat = _coords_from_geom(row.geometry)
            nid = f"charge_{i}"
            nodes_list.append(((lon, lat), "charge", nid))
            trunk_nodes.append((nid, lon, lat))

        G = nx.Graph()
        for (lon, lat), typ, nid in nodes_list:
            G.add_node(nid, lon=lon, lat=lat, node_type=typ)

        if len(trunk_nodes) < 2:
            return G

        nfz_union = None
        if obstacles is not None and getattr(obstacles, "is_empty", True) is False and getattr(obstacles, "is_valid", True):
            nfz_union = obstacles

        nfz_loose_geom = None
        if obstacles_relaxed is not None and getattr(obstacles_relaxed, "is_empty", True) is False and getattr(
            obstacles_relaxed, "is_valid", True
        ):
            nfz_loose_geom = obstacles_relaxed

        # Важно: для корректных проверок пересечений/касаний линии с no-fly зонами
        # используем метрическую СК (UTM). Иначе в EPSG:4326 (градусы) ошибки допусков
        # могут приводить к тому, что визуально маршрут заходит в зону.
        nfz_union_metric = None
        to_metric: Optional[Transformer] = None
        trunk_mid_lon = float(np.mean([lon for _, lon, _ in trunk_nodes]))
        trunk_mid_lat = float(np.mean([lat for _, _, lat in trunk_nodes]))
        if nfz_union is not None and len(trunk_nodes) >= 2:
            try:
                utm_epsg = _utm_epsg_from_lonlat(trunk_mid_lon, trunk_mid_lat)
                src_crs = CRS.from_epsg(4326)
                dst_crs = CRS.from_epsg(utm_epsg)
                to_metric = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                nfz_union_metric = shapely_transform(to_metric.transform, nfz_union)
            except Exception:
                nfz_union_metric = None
                to_metric = None

        air_original = air_graph.copy() if air_graph is not None and air_graph.number_of_nodes() > 1 else None
        air_strict: Optional[nx.Graph] = None
        air_loose: Optional[nx.Graph] = None
        if air_original is not None:
            try:
                if nfz_union is not None:
                    air_strict = build_safe_graph(
                        air_original.copy(),
                        nfz_union,
                        safety_buffer=no_fly_safety_buffer,
                    )
                else:
                    air_strict = air_original.copy()
            except Exception:
                air_strict = air_original.copy()
            try:
                if nfz_loose_geom is not None:
                    air_loose = build_safe_graph(air_original.copy(), nfz_loose_geom, safety_buffer=0.0)
                else:
                    air_loose = None
            except Exception:
                air_loose = None

        detour_counter = [0]
        nfz_metric_pair: Optional[Tuple[Any, Optional[Transformer]]] = None
        if nfz_union_metric is not None and to_metric is not None:
            nfz_metric_pair = (nfz_union_metric, to_metric)

        pts = np.array([[lon, lat] for _, lon, lat in trunk_nodes])
        ids = [nid for nid, _, _ in trunk_nodes]

        forced_straight_trunk = [0]

        def _air_pick(ag: Optional[nx.Graph]):
            if ag is None or ag.number_of_nodes() < 2:
                return None, None, None, None
            ids_l = list(ag.nodes())
            pts_l: List[Tuple[float, float]] = []
            for nid in ids_l:
                data = ag.nodes[nid]
                lon = data.get("lon") or data.get("x")
                lat = data.get("lat") or data.get("y")
                if lon is None or lat is None:
                    pts_l.append((float("nan"), float("nan")))
                else:
                    pts_l.append((float(lon), float(lat)))
            ac = np.array(pts_l, dtype=float)
            vm = ~np.isnan(ac[:, 0]) & ~np.isnan(ac[:, 1])
            if not np.any(vm):
                return None, None, None, None
            ac = ac[vm]
            ids_l = [ids_l[i] for i, ok in enumerate(vm) if ok]
            try:
                tr = cKDTree(ac)
            except Exception:
                tr = None
            return ag, ac, ids_l, tr

        def _add_trunk_edge(nid_i: str, nid_j: str, lon_i: float, lat_i: float, lon_j: float, lat_j: float) -> None:
            if G.has_edge(nid_i, nid_j):
                return
            seg = LineString([(float(lon_i), float(lat_i)), (float(lon_j), float(lat_j))])
            d_km = haversine_km(float(lat_i), float(lon_i), float(lat_j), float(lon_j))

            if nfz_union is None and nfz_loose_geom is None:
                G.add_edge(nid_i, nid_j, weight=float(d_km), length=float(d_km))
                return

            routed = None
            ags, acs, ids_s, trs = _air_pick(air_strict)
            if nfz_union is not None:
                routed = route_lonlat_segment_with_nfz_detours(
                    lon_i,
                    lat_i,
                    lon_j,
                    lat_j,
                    nfz_union,
                    air_graph=ags,
                    air_coords=acs,
                    air_ids=ids_s,
                    air_tree=trs,
                    no_fly_safety_buffer=no_fly_safety_buffer,
                    nfz_metric=nfz_metric_pair,
                    utm_anchor_lonlat=(trunk_mid_lon, trunk_mid_lat),
                    logger=self.logger,
                    detour_counter=detour_counter,
                    log_edge_label="%s–%s" % (nid_i, nid_j),
                )

            if routed is None and nfz_loose_geom is not None:
                agl, acl, idl, trl = _air_pick(air_loose)
                routed = route_lonlat_segment_with_nfz_detours(
                    lon_i,
                    lat_i,
                    lon_j,
                    lat_j,
                    nfz_loose_geom,
                    air_graph=agl,
                    air_coords=acl,
                    air_ids=idl,
                    air_tree=trl,
                    no_fly_safety_buffer=0.0,
                    nfz_metric=None,
                    utm_anchor_lonlat=(trunk_mid_lon, trunk_mid_lat),
                    logger=self.logger,
                    detour_counter=detour_counter,
                    log_edge_label="%s–%s(loose)" % (nid_i, nid_j),
                )

            if routed is not None:
                coords, w_km, how = routed
                ek: Dict[str, Any] = {
                    "weight": float(w_km),
                    "length": float(w_km),
                    "trunk_route_mode": "routed",
                }
                if how != "straight":
                    ek["geometry_coords"] = coords
                G.add_edge(nid_i, nid_j, **ek)
                return

            if nfz_union is not None and not is_edge_blocked(seg, nfz_union, safety_buffer=0.0):
                G.add_edge(
                    nid_i,
                    nid_j,
                    weight=float(d_km),
                    length=float(d_km),
                    geometry_coords=[(float(lon_i), float(lat_i)), (float(lon_j), float(lat_j))],
                    trunk_route_mode="straight_clear_buffer",
                )
                return

            if nfz_loose_geom is not None and not is_edge_blocked(seg, nfz_loose_geom, safety_buffer=0.0):
                G.add_edge(
                    nid_i,
                    nid_j,
                    weight=float(d_km),
                    length=float(d_km),
                    geometry_coords=[(float(lon_i), float(lat_i)), (float(lon_j), float(lat_j))],
                    trunk_route_mode="straight_clear_core_nfz",
                )
                return

            G.add_edge(
                nid_i,
                nid_j,
                weight=float(d_km),
                length=float(d_km),
                geometry_coords=[(float(lon_i), float(lat_i)), (float(lon_j), float(lat_j))],
                trunk_route_mode="forced_straight",
            )
            forced_straight_trunk[0] += 1
            self.logger.warning(
                "Магистраль %s–%s: обход не найден — добавлена прямая линия (может пересекать зоны).",
                nid_i,
                nid_j,
            )

        use_mst = str(topology_mode or "").strip().lower() == "mst"
        if use_mst and len(trunk_nodes) >= 2:
            full = nx.Graph()
            for nid, lon, lat in trunk_nodes:
                full.add_node(nid, lon=lon, lat=lat)
            for i in range(len(trunk_nodes)):
                nid_i, lon_i, lat_i = trunk_nodes[i]
                for j in range(i + 1, len(trunk_nodes)):
                    nid_j, lon_j, lat_j = trunk_nodes[j]
                    d_km = haversine_km(lat_i, lon_i, lat_j, lon_j)
                    full.add_edge(nid_i, nid_j, weight=d_km)
            mst = nx.minimum_spanning_tree(full, weight="weight")
            mst_edge_list = list(mst.edges())
            for u, v in mst_edge_list:
                du = full.nodes[u]
                dv = full.nodes[v]
                _add_trunk_edge(
                    u,
                    v,
                    float(du["lon"]),
                    float(du["lat"]),
                    float(dv["lon"]),
                    float(dv["lat"]),
                )
        else:
            k = min(max(1, int(max_neighbors_a)), 2, len(trunk_nodes) - 1)
            if k <= 0:
                return G
            for i in range(len(trunk_nodes)):
                nid_i = ids[i]
                lat_i, lon_i = pts[i, 1], pts[i, 0]
                cand = []
                for j in range(len(trunk_nodes)):
                    if j == i:
                        continue
                    d_km = haversine_km(lat_i, lon_i, pts[j, 1], pts[j, 0])
                    if d_km <= max_edge_km:
                        cand.append((d_km, j))
                cand.sort(key=lambda x: x[0])
                for _, j in cand[:k]:
                    nid_j = ids[j]
                    lon_j, lat_j = pts[j, 0], pts[j, 1]
                    _add_trunk_edge(nid_i, nid_j, lon_i, lat_i, lon_j, lat_j)

        trunk_node_ids = list(G.nodes())
        if len(trunk_node_ids) > 1:
            sub = G.subgraph(trunk_node_ids).copy()
            components = list(nx.connected_components(sub))
            if len(components) > 1:
                coords = {
                    n: (G.nodes[n].get("lon"), G.nodes[n].get("lat"))
                    for n in trunk_node_ids
                    if G.nodes[n].get("lon") is not None
                }
                base_comp = set(components[0])
                remaining = [set(c) for c in components[1:]]
                # Соединяем оставшиеся компоненты с растущим base_comp: перебираем пары по длине,
                # пока не удастся _add_trunk_edge. Раньше бралась только ближайшая пара и компонента
                # выкидывалась из очереди даже при провале маршрута — рёбра терялись.
                while remaining:
                    cand_pairs: List[Tuple[float, str, str, int]] = []
                    for idx, comp in enumerate(remaining):
                        for a in base_comp:
                            if a not in coords:
                                continue
                            lon_a, lat_a = coords[a]
                            for b in comp:
                                if b not in coords:
                                    continue
                                lon_b, lat_b = coords[b]
                                d_km = haversine_km(lat_a, lon_a, lat_b, lon_b)
                                cand_pairs.append((float(d_km), a, b, idx))
                    cand_pairs.sort(key=lambda t: t[0])
                    progressed = False
                    for _d, a, b, idx in cand_pairs:
                        u, v = a, b
                        lon_u, lat_u = coords[u]
                        lon_v, lat_v = coords[v]
                        if not G.has_edge(u, v):
                            _add_trunk_edge(u, v, lon_u, lat_u, lon_v, lat_v)
                        if G.has_edge(u, v):
                            base_comp |= remaining[idx]
                            del remaining[idx]
                            progressed = True
                            break
                    if not progressed:
                        self.logger.error(
                            "Магистраль: не удалось соединить компоненты (неожиданно; ожидалось принудительное ребро)."
                        )
                        break

        # Добавляем до N дополнительных путей между станциями A,
        # чтобы сеть была более устойчивой (резервные маршруты).
        extra_n = max(0, int(extra_paths_per_a))
        if extra_n > 0:
            charge_nodes = [(nid, lon, lat) for nid, lon, lat in trunk_nodes if str(nid).startswith("charge_")]
            if len(charge_nodes) >= 2:
                added_cnt: Dict[str, int] = {nid: 0 for nid, _, _ in charge_nodes}
                cand_edges: List[Tuple[float, str, float, float, str, float, float]] = []
                for i in range(len(charge_nodes)):
                    ni, loni, lati = charge_nodes[i]
                    for j in range(i + 1, len(charge_nodes)):
                        nj, lonj, latj = charge_nodes[j]
                        if G.has_edge(ni, nj):
                            continue
                        d_km = haversine_km(lati, loni, latj, lonj)
                        if d_km <= max_edge_km:
                            cand_edges.append((d_km, ni, loni, lati, nj, lonj, latj))
                cand_edges.sort(key=lambda x: x[0])
                for _, ni, loni, lati, nj, lonj, latj in cand_edges:
                    if added_cnt.get(ni, 0) >= extra_n or added_cnt.get(nj, 0) >= extra_n:
                        continue
                    if G.has_edge(ni, nj):
                        continue
                    _add_trunk_edge(ni, nj, loni, lati, lonj, latj)
                    if G.has_edge(ni, nj):
                        added_cnt[ni] = added_cnt.get(ni, 0) + 1
                        added_cnt[nj] = added_cnt.get(nj, 0) + 1
        if forced_straight_trunk[0] > 0:
            self.logger.warning(
                "Магистраль: %s рёбер построены прямой линией через препятствие (forced_straight) — проверьте на карте.",
                forced_straight_trunk[0],
            )
        self.logger.info(
            f"Магистраль (только А, макс. {max_neighbors_a} соседей): {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер"
        )
        return G

    def _route_branch_segment_lonlat(
        self,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
        nfz_union: Any,
        nfz_metric_pair: Optional[Tuple[Any, Optional[Transformer]]],
        air_work: Optional[nx.Graph],
        air_coords: Optional[np.ndarray],
        air_ids: Optional[List[Any]],
        air_tree: Optional[Any],
        anchor: Tuple[float, float],
        detour_counter: List[int],
        log_label: str,
        *,
        no_fly_safety_buffer: float = 0.0,
    ) -> Tuple[List[List[float]], float]:
        """Полилиния ветки с обходом NFZ (контур / A*), иначе прямой отрезок."""
        routed = route_lonlat_segment_with_nfz_detours(
            lon1,
            lat1,
            lon2,
            lat2,
            nfz_union,
            air_graph=air_work,
            air_coords=air_coords,
            air_ids=air_ids,
            air_tree=air_tree,
            no_fly_safety_buffer=no_fly_safety_buffer,
            nfz_metric=nfz_metric_pair,
            utm_anchor_lonlat=anchor,
            logger=self.logger,
            detour_counter=detour_counter,
            log_edge_label=log_label,
        )
        d0 = haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))
        if routed is None:
            self.logger.warning("Ветка %s: обход не построен (конец в NFZ или ошибка) — прямая линия.", log_label)
            return [[float(lon1), float(lat1)], [float(lon2), float(lat2)]], round(float(d0), 4)
        coords, w_km, _how = routed
        line_coords = [[float(c[0]), float(c[1])] for c in coords]
        return line_coords, round(float(w_km), 4)

    def build_branch_edges(
        self,
        charge_stations: gpd.GeoDataFrame,
        *,
        max_b_per_type_a: int = MAX_TYPE_B_BRANCHES_PER_TYPE_A,
        max_branch_km: float = R_GARAGE_TO_KM,
        obstacles: Any = None,
        air_graph: Optional[nx.Graph] = None,
        no_fly_safety_buffer: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Ветки только Б→А (полёт на эшелонах 4–5): у каждой Б ровно одна ветка. Min-cost flow только по рёбрам «до k ближайших Б к каждой А» (k=BRANCH_NEAREST_B_PER_STATION_A),
        не более max_b_per_type_a Б на одну А; в весах d + штраф за превышение max_branch_km (квадрат излишка).
        При заданных obstacles / air_graph геометрия ветки — обход NFZ и высоких зданий (контур и/или A*), как у магистрали.
        Returns: список GeoJSON-like Feature (LineString, edge_type="branch", source_id, target_id, weight_km).
        """
        if charge_stations is None or len(charge_stations) == 0:
            return []
        type_a = (
            charge_stations["station_type"] == "charge_a"
            if "station_type" in charge_stations.columns
            else pd.Series([True] * len(charge_stations))
        )
        type_b = (
            charge_stations["station_type"] == "charge_b"
            if "station_type" in charge_stations.columns
            else pd.Series([False] * len(charge_stations))
        )
        a_indices = [i for i in charge_stations.index[type_a]]
        b_indices = [i for i in charge_stations.index[type_b]]
        if not b_indices or not a_indices:
            return []
        cap = max(1, int(max_b_per_type_a))
        a_pts = {i: _coords_from_geom(charge_stations.loc[i].geometry) for i in a_indices}
        b_pts = np.array([_coords_from_geom(charge_stations.loc[i].geometry) for i in b_indices])
        n_b = len(b_indices)
        n_a = len(a_indices)
        dist = np.zeros((n_b, n_a), dtype=float)
        for bi in range(n_b):
            lon_b, lat_b = float(b_pts[bi, 0]), float(b_pts[bi, 1])
            for aj in range(n_a):
                ia = a_indices[aj]
                lon_a, lat_a = a_pts[ia]
                dist[bi, aj] = haversine_km(lat_b, lon_b, lat_a, lon_a)

        # Рёбра назначения: только до BRANCH_NEAREST_B_PER_STATION_A ближайших Б к каждой А (+ хотя бы одно ребро на каждую Б).
        k_near = min(int(BRANCH_NEAREST_B_PER_STATION_A), n_b)
        allowed = np.zeros((n_b, n_a), dtype=bool)
        for aj in range(n_a):
            order = np.argsort(dist[:, aj])
            for t in range(k_near):
                allowed[int(order[t]), aj] = True
        for bi in range(n_b):
            if not np.any(allowed[bi, :]):
                jn = int(np.argmin(dist[bi, :]))
                allowed[bi, jn] = True

        # Веса целочисленные; штраф за переполнение квоты >> любой суммы «нормальных» расстояний.
        scale = 1_000_000
        overflow_penalty = int(50_000 * scale)
        pen_k = float(BRANCH_ASSIGN_OVER_MAX_KM_PENALTY)
        mbrk = float(max_branch_km)

        def _decode_flow(flow: Dict[str, Dict[str, int]]) -> List[int]:
            chosen_aj: List[int] = []
            for bi in range(n_b):
                bnode = f"__b{bi}"
                aj_pick = -1
                for v, amt in (flow.get(bnode) or {}).items():
                    if amt > 0 and v.startswith("__a") and v.endswith("_in"):
                        aj_pick = int(v[len("__a") : -len("_in")])
                        break
                chosen_aj.append(aj_pick)
            return chosen_aj

        def _greedy_by_margin() -> List[int]:
            """Запасной вариант: сначала Б с наименьшим «зазором» между 1-й и 2-й по расстоянию А."""
            def _margin_bi(bi: int) -> float:
                row = dist[bi]
                if n_a <= 1:
                    return 0.0
                rs = np.sort(row)
                return float(rs[1] - rs[0])

            order = sorted(range(n_b), key=_margin_bi)
            used = [0] * n_a
            pick: Dict[int, int] = {}
            for bi in order:
                js = np.argsort(dist[bi])
                chosen_j = int(js[0])
                for j in js:
                    jj = int(j)
                    if used[jj] < cap:
                        chosen_j = jj
                        break
                used[chosen_j] += 1
                pick[bi] = chosen_j
            return [pick[i] for i in range(n_b)]

        G = nx.DiGraph()
        G.add_node("__S", demand=-n_b)
        G.add_node("__T", demand=n_b)
        for bi in range(n_b):
            bnode = f"__b{bi}"
            G.add_edge("__S", bnode, capacity=1, weight=0)
            for aj in range(n_a):
                if not allowed[bi, aj]:
                    continue
                ain = f"__a{aj}_in"
                d_km = float(dist[bi, aj])
                excess = max(0.0, d_km - mbrk)
                w = int(scale * (d_km + pen_k * excess * excess))
                G.add_edge(bnode, ain, capacity=1, weight=w)
        for aj in range(n_a):
            ain, aout, aov = f"__a{aj}_in", f"__a{aj}_out", f"__a{aj}_ov"
            G.add_edge(ain, aout, capacity=cap, weight=0)
            G.add_edge(aout, "__T", capacity=cap, weight=0)
            G.add_edge(ain, aov, capacity=n_b, weight=overflow_penalty)
            G.add_edge(aov, "__T", capacity=n_b, weight=0)

        chosen_aj: List[int]
        n_overflow = 0
        try:
            flow = nx.min_cost_flow(G)
            chosen_aj = _decode_flow(flow)
            if any(j < 0 for j in chosen_aj):
                raise nx.NetworkXError("incomplete B→A flow decode")
            for aj in range(n_a):
                ain, aov = f"__a{aj}_in", f"__a{aj}_ov"
                ov = int((flow.get(ain) or {}).get(aov, 0))
                n_overflow += ov
        except (nx.NetworkXError, nx.NetworkXUnfeasible, ValueError, KeyError) as e:
            self.logger.warning("Ветки Б→А: min-cost flow не удался (%s), запасной жадный разбор по зазору.", e)
            chosen_aj = _greedy_by_margin()
            n_overflow = 0

        lons_anchor: List[float] = [float(x) for x in b_pts[:, 0]]
        lats_anchor: List[float] = [float(x) for x in b_pts[:, 1]]
        for bi in range(n_b):
            ia = a_indices[chosen_aj[bi]]
            la, lata = a_pts[ia]
            lons_anchor.append(float(la))
            lats_anchor.append(float(lata))
        anchor = (float(np.mean(lons_anchor)), float(np.mean(lats_anchor)))
        nfz_union, nfz_metric_pair, air_work, air_coords, air_ids, air_tree = prepare_air_detour_auxiliary(
            obstacles,
            air_graph,
            anchor,
            no_fly_safety_buffer=no_fly_safety_buffer,
        )
        detour_counter = [0]

        features: List[Dict[str, Any]] = []
        for bi, idx_b in enumerate(b_indices):
            aj = chosen_aj[bi]
            ia = a_indices[aj]
            lon_b, lat_b = float(b_pts[bi, 0]), float(b_pts[bi, 1])
            lon_a, lat_a = float(a_pts[ia][0]), float(a_pts[ia][1])
            line_coords, w_km = self._route_branch_segment_lonlat(
                lon_b,
                lat_b,
                lon_a,
                lat_a,
                nfz_union,
                nfz_metric_pair,
                air_work,
                air_coords,
                air_ids,
                air_tree,
                anchor,
                detour_counter,
                "branch_B %s→A %s" % (idx_b, ia),
                no_fly_safety_buffer=no_fly_safety_buffer,
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": line_coords},
                    "properties": {
                        "edge_type": "branch",
                        "source_id": str(idx_b),
                        "target_id": f"charge_{ia}",
                        "weight_km": w_km,
                    },
                }
            )
        if n_overflow > 0:
            self.logger.warning(
                "Ветки Б→А: %s соединений с превышением квоты %s Б на А (min-cost с штрафом).",
                n_overflow,
                cap,
            )
        self.logger.info("Ветки Б→А (макс. %s Б на одну А, min-cost): %s рёбер", cap, len(features))
        return features

    def build_facility_branch_edges(
        self,
        charge_stations: gpd.GeoDataFrame,
        garage_points: Optional[gpd.GeoDataFrame] = None,
        to_points: Optional[gpd.GeoDataFrame] = None,
        *,
        max_branch_km: float = R_GARAGE_TO_KM,
        obstacles: Any = None,
        air_graph: Optional[nx.Graph] = None,
        no_fly_safety_buffer: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Ветки гараж и ТО только к ближайшей зарядке А (полёт на эшелонах 4–5; по одной на объект; в пределах max_branch_km, иначе ближайшая без порога).
        Геометрия — обход NFZ и высоких зданий при заданных obstacles / air_graph.
        """
        if charge_stations is None or len(charge_stations) == 0:
            return []
        type_a = (
            charge_stations["station_type"] == "charge_a"
            if "station_type" in charge_stations.columns
            else pd.Series([True] * len(charge_stations))
        )
        a_indices = [i for i in charge_stations.index[type_a]]
        if not a_indices:
            return []
        a_pts = {i: _coords_from_geom(charge_stations.loc[i].geometry) for i in a_indices}
        features: List[Dict[str, Any]] = []

        def _nearest_a(lon_f: float, lat_f: float) -> Tuple[Any, float, float, float]:
            best_ia: Any = a_indices[0]
            best_d = math.inf
            for ia in a_indices:
                lon_a, lat_a = a_pts[ia]
                d_km = haversine_km(lat_f, lon_f, lat_a, lon_a)
                if d_km < best_d:
                    best_d = d_km
                    best_ia = ia
            la, lata = a_pts[best_ia]
            return best_ia, float(la), float(lata), float(best_d)

        pending: List[Tuple[str, Any, float, float, float, float, str]] = []
        if garage_points is not None:
            for i, row in garage_points.iterrows():
                lon_f, lat_f = _coords_from_geom(row.geometry)
                ia, lon_a, lat_a, d_km = _nearest_a(lon_f, lat_f)
                if d_km > max_branch_km:
                    self.logger.info(
                        "Гараж %s: ближайшая А на %.2f км (порог %.2f км) — ветка к ближайшей А.",
                        i,
                        d_km,
                        max_branch_km,
                    )
                pending.append((f"garage_{i}", ia, lon_f, lat_f, lon_a, lat_a, "branch_garage %s→A %s" % (i, ia)))
        if to_points is not None:
            for i, row in to_points.iterrows():
                lon_f, lat_f = _coords_from_geom(row.geometry)
                ia, lon_a, lat_a, d_km = _nearest_a(lon_f, lat_f)
                if d_km > max_branch_km:
                    self.logger.info(
                        "ТО %s: ближайшая А на %.2f км (порог %.2f км) — ветка к ближайшей А.",
                        i,
                        d_km,
                        max_branch_km,
                    )
                pending.append((f"to_{i}", ia, lon_f, lat_f, lon_a, lat_a, "branch_to %s→A %s" % (i, ia)))

        if not pending:
            return []

        lons_a = [t[2] for t in pending] + [t[4] for t in pending]
        lats_a = [t[3] for t in pending] + [t[5] for t in pending]
        anchor = (float(np.mean(lons_a)), float(np.mean(lats_a)))
        nfz_union, nfz_metric_pair, air_work, air_coords, air_ids, air_tree = prepare_air_detour_auxiliary(
            obstacles,
            air_graph,
            anchor,
            no_fly_safety_buffer=no_fly_safety_buffer,
        )
        detour_counter = [0]

        for sid, ia, lon_f, lat_f, lon_a, lat_a, lbl in pending:
            line_coords, w_km = self._route_branch_segment_lonlat(
                lon_f,
                lat_f,
                lon_a,
                lat_a,
                nfz_union,
                nfz_metric_pair,
                air_work,
                air_coords,
                air_ids,
                air_tree,
                anchor,
                detour_counter,
                lbl,
                no_fly_safety_buffer=no_fly_safety_buffer,
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": line_coords},
                    "properties": {
                        "edge_type": "branch",
                        "source_id": sid,
                        "target_id": f"charge_{ia}",
                        "weight_km": w_km,
                    },
                }
            )
        self.logger.info("Ветки гараж/ТО→А: %s рёбер", len(features))
        return features

    def build_local_edges(
        self,
        charge_stations: gpd.GeoDataFrame,
        obstacles=None,
        air_graph: Optional[nx.Graph] = None,
        *,
        max_edge_km: float = R_GARAGE_TO_KM,
        max_neighbors_b: int = 2,
        no_fly_safety_buffer: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Локальные связи между станциями типа Б (полёт на эшелонах 4–5). Каждая Б соединяется с макс. max_neighbors_b ближайшими Б в пределах max_edge_km.
        При наличии препятствий и воздушного графа строит тот же обход, что и магистраль (контур + A*).
        Returns: список GeoJSON-like Feature (LineString, edge_type="local", weight_km).
        """
        if charge_stations is None or len(charge_stations) == 0:
            return []
        type_b = (
            charge_stations["station_type"] == "charge_b"
            if "station_type" in charge_stations.columns
            else pd.Series([False] * len(charge_stations))
        )
        b_indices = [i for i in charge_stations.index[type_b]]
        if len(b_indices) < 2:
            return []
        b_pts = np.array([_coords_from_geom(charge_stations.loc[i].geometry) for i in b_indices])
        anchor = (float(np.mean(b_pts[:, 0])), float(np.mean(b_pts[:, 1])))
        nfz_union, nfz_metric_pair, air_work, air_coords, air_ids, air_tree = prepare_air_detour_auxiliary(
            obstacles,
            air_graph,
            anchor,
            no_fly_safety_buffer=no_fly_safety_buffer,
        )

        detour_counter = [0]
        features = []
        k = min(max_neighbors_b, len(b_indices) - 1)
        seen_edges = set()
        for i in range(len(b_indices)):
            cand = []
            for j in range(len(b_indices)):
                if j == i:
                    continue
                d_km = haversine_km(b_pts[i, 1], b_pts[i, 0], b_pts[j, 1], b_pts[j, 0])
                if d_km <= max_edge_km:
                    cand.append((d_km, j))
            cand.sort(key=lambda x: x[0])
            for _d_unused, j in cand[:k]:
                ui, vi = b_indices[i], b_indices[j]
                edge_key = (min(ui, vi), max(ui, vi))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                lon_i, lat_i = float(b_pts[i, 0]), float(b_pts[i, 1])
                lon_j, lat_j = float(b_pts[j, 0]), float(b_pts[j, 1])
                routed = route_lonlat_segment_with_nfz_detours(
                    lon_i,
                    lat_i,
                    lon_j,
                    lat_j,
                    nfz_union,
                    air_graph=air_work,
                    air_coords=air_coords,
                    air_ids=air_ids,
                    air_tree=air_tree,
                    no_fly_safety_buffer=no_fly_safety_buffer,
                    nfz_metric=nfz_metric_pair,
                    utm_anchor_lonlat=anchor,
                    logger=self.logger,
                    detour_counter=detour_counter,
                    log_edge_label="local_B %s–%s" % (ui, vi),
                )
                if routed is None:
                    continue
                coords, w_km, _how = routed
                line_coords = [[float(c[0]), float(c[1])] for c in coords]
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": line_coords},
                        "properties": {
                            "edge_type": "local",
                            "source_id": str(ui),
                            "target_id": str(vi),
                            "weight_km": round(float(w_km), 4),
                        },
                    }
                )
        self.logger.info(f"Локальные рёбра Б↔Б (макс. {max_neighbors_b} на станцию): {len(features)}")
        return features

    def place_garages(
        self,
        demand: gpd.GeoDataFrame,
        garage_candidates: gpd.GeoDataFrame,
        *,
        k: int = 3,
        radius_km: float = R_GARAGE_TO_KM,
        coverage_ratio_target: Optional[float] = None,
        min_distance_factor: float = 2.0,
        max_garages: int = 100,
    ) -> gpd.GeoDataFrame:
        """Размещение гаражей: либо k-median (фиксированное k), либо жадно по покрытию спроса до coverage_ratio_target."""
        if garage_candidates is None or len(garage_candidates) == 0:
            return gpd.GeoDataFrame()
        if demand is None or len(demand) == 0:
            out = garage_candidates.iloc[:k].copy()
            out["station_type"] = "garage"
            return out

        dem_pts = _points_array(demand)
        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
        cand_pts = _points_array(garage_candidates)
        n_cand = len(cand_pts)
        total_weight = float(weights.sum()) if len(weights) > 0 else 0.0

        if coverage_ratio_target is not None and total_weight > 0:
            min_dist_km = max(0.0, float(min_distance_factor)) * radius_km
            covered = np.zeros(len(dem_pts), dtype=bool)
            chosen = []
            for _ in range(max_garages):
                if (weights[covered].sum() / total_weight) >= coverage_ratio_target:
                    break
                best_cand = -1
                best_new_weight = 0.0
                for c in range(n_cand):
                    if c in chosen:
                        continue
                    if chosen and min_dist_km > 0:
                        if any(
                            haversine_km(cand_pts[c, 1], cand_pts[c, 0], cand_pts[prev, 1], cand_pts[prev, 0])
                            < min_dist_km
                            for prev in chosen
                        ):
                            continue
                    lat_c, lon_c = cand_pts[c, 1], cand_pts[c, 0]
                    new_weight = 0.0
                    for j in range(len(dem_pts)):
                        if covered[j]:
                            continue
                        if haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0]) <= radius_km:
                            new_weight += weights[j]
                    if new_weight > best_new_weight:
                        best_new_weight = new_weight
                        best_cand = c
                if best_cand < 0 or best_new_weight <= 0:
                    break
                chosen.append(best_cand)
                lat_c, lon_c = cand_pts[best_cand, 1], cand_pts[best_cand, 0]
                for j in range(len(dem_pts)):
                    if not covered[j] and haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0]) <= radius_km:
                        covered[j] = True
            rows = [garage_candidates.iloc[idx].copy() for idx in chosen]
            for r in rows:
                r["station_type"] = "garage"
            out = gpd.GeoDataFrame(rows, crs=garage_candidates.crs)
            self.logger.info(
                f"Гаражи (покрытие): размещено {len(out)}, покрыто спроса {float(weights[covered].sum()):.1f} из {total_weight:.1f}"
            )
            return out

        chosen: List[int] = []
        nearest = np.full(len(dem_pts), -1)
        best_dist = np.full(len(dem_pts), np.inf)

        for _ in range(min(k, n_cand)):
            best_cand = -1
            best_cost = np.inf
            for c in range(n_cand):
                if c in chosen:
                    continue
                if chosen:
                    if any(
                        haversine_km(cand_pts[c, 1], cand_pts[c, 0], cand_pts[prev, 1], cand_pts[prev, 0])
                        <= radius_km
                        for prev in chosen
                    ):
                        continue
                lat_c, lon_c = cand_pts[c, 1], cand_pts[c, 0]
                cost = 0.0
                for j in range(len(dem_pts)):
                    d = haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0])
                    if d > radius_km:
                        cost += weights[j] * radius_km * 2
                    else:
                        cost += weights[j] * min(d, best_dist[j])
                if cost < best_cost:
                    best_cost = cost
                    best_cand = c
            if best_cand < 0:
                break
            chosen.append(best_cand)
            lat_c, lon_c = cand_pts[best_cand, 1], cand_pts[best_cand, 0]
            for j in range(len(dem_pts)):
                d = haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0])
                if d < best_dist[j]:
                    best_dist[j] = d
                    nearest[j] = len(chosen) - 1

        rows = []
        total_weight = float(weights.sum()) if len(weights) > 0 else 0.0
        served_weight = np.zeros(len(chosen), dtype=float) if chosen else np.array([])
        if chosen and total_weight > 0:
            for j in range(len(dem_pts)):
                idx_pos = nearest[j]
                if 0 <= idx_pos < len(chosen) and best_dist[j] <= radius_km:
                    served_weight[idx_pos] += float(weights[j])
            if len(chosen) > 1:
                max_served = float(served_weight.max()) if served_weight.size > 0 else 0.0
                keep_positions = []
                min_share_total = 0.05
                for pos in range(len(chosen)):
                    share_total = (served_weight[pos] / total_weight) if total_weight > 0 else 0.0
                    share_rel = (served_weight[pos] / max_served) if max_served > 0 else 0.0
                    if share_total >= min_share_total or share_rel >= 0.3:
                        keep_positions.append(pos)
                if not keep_positions:
                    best_pos = int(np.argmax(served_weight)) if served_weight.size > 0 else 0
                    keep_positions = [best_pos]
                chosen = [chosen[pos] for pos in keep_positions]

        for idx in chosen:
            row = garage_candidates.iloc[idx].copy()
            row["station_type"] = "garage"
            rows.append(row)
        out = gpd.GeoDataFrame(rows, crs=garage_candidates.crs)
        self.logger.info(f"Гаражи: размещено {len(out)}")
        return out

    def place_to_stations(
        self,
        trunk_graph: nx.Graph,
        to_candidates: gpd.GeoDataFrame,
        demand: Optional[gpd.GeoDataFrame] = None,
        *,
        k: int = 2,
        radius_km: float = R_GARAGE_TO_KM,
        coverage_ratio_target: Optional[float] = None,
        min_distance_factor: float = 2.0,
        max_to_stations: int = 100,
    ) -> gpd.GeoDataFrame:
        """Размещает станции ТО рядом с магистралью и покрывает спрос в заданном радиусе."""
        if to_candidates is None or len(to_candidates) == 0:
            return gpd.GeoDataFrame()
        if trunk_graph is None:
            trunk_graph = nx.Graph()

        has_demand = demand is not None and len(demand) > 0
        if has_demand:
            dem_pts = _points_array(demand)
            weights = (
                demand["weight"].values
                if isinstance(demand, gpd.GeoDataFrame) and "weight" in demand.columns
                else np.ones(len(demand))
            )
            cand_pts = _points_array(to_candidates)
            n_demand = len(dem_pts)
            total_weight = float(weights.sum())
            covered = np.zeros(n_demand, dtype=bool)

            if coverage_ratio_target is not None and total_weight > 0:
                min_dist_km = max(0.0, float(min_distance_factor)) * radius_km
                chosen_idx: List[int] = []
                for _ in range(max_to_stations):
                    if (weights[covered].sum() / total_weight) >= coverage_ratio_target:
                        break
                    best_i = -1
                    best_new_weight = 0.0
                    for i in range(len(cand_pts)):
                        if i in chosen_idx:
                            continue
                        if chosen_idx and min_dist_km > 0:
                            if any(
                                haversine_km(
                                    cand_pts[i, 1],
                                    cand_pts[i, 0],
                                    cand_pts[j, 1],
                                    cand_pts[j, 0],
                                )
                                < min_dist_km
                                for j in chosen_idx
                            ):
                                continue
                        lat_i, lon_i = cand_pts[i, 1], cand_pts[i, 0]
                        new_weight = 0.0
                        for d_idx in range(n_demand):
                            if not covered[d_idx] and haversine_km(
                                lat_i, lon_i, dem_pts[d_idx, 1], dem_pts[d_idx, 0]
                            ) <= radius_km:
                                new_weight += weights[d_idx]
                        if new_weight > best_new_weight:
                            best_new_weight = new_weight
                            best_i = i
                    if best_i < 0 or best_new_weight <= 0:
                        break
                    chosen_idx.append(best_i)
                    lat_i, lon_i = cand_pts[best_i, 1], cand_pts[best_i, 0]
                    for d_idx in range(n_demand):
                        if not covered[d_idx] and haversine_km(
                            lat_i, lon_i, dem_pts[d_idx, 1], dem_pts[d_idx, 0]
                        ) <= radius_km:
                            covered[d_idx] = True
                rows = [to_candidates.iloc[i].copy() for i in chosen_idx]
                for r in rows:
                    r["station_type"] = "to"
                out = gpd.GeoDataFrame(rows, crs=to_candidates.crs)
                self.logger.info(
                    f"ТО (покрытие): размещено {len(out)}; покрыто спроса {float(weights[covered].sum()):.1f} из {total_weight:.1f}"
                )
                return out

            chosen_idx = []
            for _ in range(min(k, len(cand_pts))):
                best_i = -1
                best_new_weight = 0.0
                for i in range(len(cand_pts)):
                    if i in chosen_idx:
                        continue
                    if chosen_idx and any(
                        haversine_km(
                            cand_pts[i, 1],
                            cand_pts[i, 0],
                            cand_pts[j, 1],
                            cand_pts[j, 0],
                        )
                        <= radius_km
                        for j in chosen_idx
                    ):
                        continue
                    lat_i, lon_i = cand_pts[i, 1], cand_pts[i, 0]
                    new_weight = 0.0
                    for d_idx in range(n_demand):
                        if not covered[d_idx] and haversine_km(
                            lat_i, lon_i, dem_pts[d_idx, 1], dem_pts[d_idx, 0]
                        ) <= radius_km:
                            new_weight += weights[d_idx]
                    if new_weight > best_new_weight:
                        best_new_weight = new_weight
                        best_i = i
                if best_i < 0 or best_new_weight <= 0:
                    break
                chosen_idx.append(best_i)
                lat_i, lon_i = cand_pts[best_i, 1], cand_pts[best_i, 0]
                for d_idx in range(n_demand):
                    if not covered[d_idx] and haversine_km(
                        lat_i, lon_i, dem_pts[d_idx, 1], dem_pts[d_idx, 0]
                    ) <= radius_km:
                        covered[d_idx] = True

            rows = [to_candidates.iloc[i].copy() for i in chosen_idx]
            for r in rows:
                r["station_type"] = "to"
            out = gpd.GeoDataFrame(rows, crs=to_candidates.crs)
            self.logger.info(
                f"ТО (coverage-first): размещено {len(out)}; покрыто спроса {float(weights[covered].sum()):.1f} из {float(weights.sum()):.1f}"
            )
            if len(out) > 0:
                return out

        try:
            between = nx.betweenness_centrality(trunk_graph, weight="weight")
        except Exception:
            between = {n: 0.0 for n in trunk_graph.nodes()}

        cand_pts = _points_array(to_candidates)
        trunk_node_ids = []
        trunk_pts = []
        for n in trunk_graph.nodes():
            data = trunk_graph.nodes[n]
            lon = data.get("lon") or data.get("x")
            lat = data.get("lat") or data.get("y")
            if lon is None or lat is None:
                continue
            trunk_node_ids.append(n)
            trunk_pts.append((lon, lat))
        if not trunk_pts:
            out = to_candidates.iloc[:k].copy()
            out["station_type"] = "to"
            return out

        trunk_pts = np.array(trunk_pts)
        tree = cKDTree(trunk_pts)
        scores: List[Tuple[int, float]] = []
        for i in range(len(cand_pts)):
            res = tree.query(cand_pts[i], k=1)
            idx = int(res[1]) if hasattr(res[1], "__int__") else res[1]
            if 0 <= idx < len(trunk_node_ids):
                nid = trunk_node_ids[idx]
                scores.append((i, between.get(nid, 0.0)))
            else:
                scores.append((i, 0.0))
        scores.sort(key=lambda x: -x[1])
        chosen_idx = []
        for i, _ in scores:
            if len(chosen_idx) >= k:
                break
            lat_i, lon_i = cand_pts[i, 1], cand_pts[i, 0]
            if chosen_idx and any(
                haversine_km(lat_i, lon_i, cand_pts[j, 1], cand_pts[j, 0]) <= radius_km
                for j in chosen_idx
            ):
                continue
            chosen_idx.append(i)
        rows = [to_candidates.iloc[i].copy() for i in chosen_idx]
        for r in rows:
            r["station_type"] = "to"
        out = gpd.GeoDataFrame(rows, crs=to_candidates.crs)
        self.logger.info(f"ТО: размещено {len(out)}")
        return out

    def compute_metrics(
        self,
        demand: gpd.GeoDataFrame,
        charge_stations: gpd.GeoDataFrame,
        trunk_graph: nx.Graph,
    ) -> Dict[str, Any]:
        """Покрытие demand точек зарядками по `cluster_id`, связность trunk."""
        metrics = {"coverage_ratio": 0.0, "trunk_connected": False}
        if demand is None or len(demand) == 0 or charge_stations is None or len(charge_stations) == 0:
            return metrics
        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
        total_weight = float(weights.sum())

        demand_cluster_cols: List[str] = []
        if "cluster_id" in demand.columns:
            demand_cluster_cols.append("cluster_id")
        if "subcluster_id" in demand.columns:
            demand_cluster_cols.append("subcluster_id")

        if not demand_cluster_cols or "cluster_id" not in charge_stations.columns:
            # Фоллбэк: раз радиусы отключены, считаем, что наличие станций достаточно.
            metrics["coverage_ratio"] = 1.0 if total_weight > 0 else 0.0
        else:
            station_ids = charge_stations["cluster_id"].dropna().values
            best_ratio = 0.0
            for demand_cluster_col in demand_cluster_cols:
                demand_ids = demand[demand_cluster_col].values
                covered_mask = np.isin(demand_ids, station_ids)
                covered_weight = float(weights[covered_mask].sum())
                ratio = covered_weight / total_weight if total_weight > 0 else 0.0
                if ratio > best_ratio:
                    best_ratio = ratio
            metrics["coverage_ratio"] = best_ratio

        if trunk_graph is not None:
            metrics["trunk_connected"] = (
                nx.is_connected(trunk_graph) if trunk_graph.number_of_nodes() > 0 else False
            )
        return metrics


# Этап 5-11: полный конвейер размещения, графов, маршрутов и эшелонов.
def run_full_pipeline(
    data_service,
    city_name: str,
    network_type: str = "drive",
    simplify: bool = True,
    *,
    demand_method: str = "dbscan",
    demand_cell_m: float = 250.0,
    dbscan_eps_m: float = 180.0,
    dbscan_min_samples: int = 15,
    candidates_per_cluster: int = 10,
    use_all_buildings: bool = False,
    max_charge_stations: Optional[int] = None,
    num_garages: int = 1,
    num_to: int = 1,
    a_by_admin_districts: bool = False,
    voronoi_buildings_per_centroid: int = 60,
    inter_cluster_max_hull_gap_m: float = 2000.0,
    inter_cluster_max_edge_length_m: float = 2000.0,
    voronoi_intra_component_bridge_max_m: float = 600.0,
) -> Dict[str, Any]:
    """
    Запускает полный пайплайн: данные города → спрос → зарядки → гаражи/ТО → магистраль (эшелоны 4–5) → ветки Б→А / гараж и ТО→А (4–5) → локальные Б↔Б (4–5) → слой Вороного (эшелоны 1–3, с сайтами станций).
    """
    logger = logging.getLogger(__name__)
    data = data_service.get_city_data(city_name, network_type=network_type, simplify=simplify, load_no_fly_zones=True)
    buildings = data.get("buildings")
    road_graph = data.get("road_graph")
    city_boundary = data.get("city_boundary")
    no_fly_zones = data.get("no_fly_zones")

    placement = StationPlacement(data_service, logger=logger)

    # Важно: если demand_method=dbscan и в get_city_data уже есть demand_points/demand_hulls,
    # то не пересчитываем demand заново (иначе логически будет "DBSCAN районов" второй раз).
    demand = None
    if demand_method == "dbscan":
        try:
            cached = data.get("demand_points") if isinstance(data, dict) else None
            if cached is not None and len(cached) > 0:
                demand = cached
        except Exception:
            demand = None

    if demand is None or len(demand) == 0:
        demand = placement.build_demand_points(
            buildings,
            road_graph,
            city_boundary,
            method=demand_method,
            cell_size_m=demand_cell_m,
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
            use_all_buildings=use_all_buildings,
            no_fly_zones=no_fly_zones,
            fill_clusters=(demand_method == "dbscan"),
            cluster_fill_step_m=None,
        )
    if demand is None or len(demand) == 0:
        return {
            "error": "Нет точек спроса",
            "demand": None,
            "charge_stations": None,
            "garages": None,
            "to_stations": None,
            "trunk_graph": None,
            "metrics": {},
        }

    if a_by_admin_districts and demand is not None and len(demand) > 0:
        def _repack_regions_by_nearest_clusters(
            d: gpd.GeoDataFrame, group_size: int = 5
        ) -> gpd.GeoDataFrame:
            """
            Формирует region_id из групп ближайших кластеров.
            Затем в каждой группе forced-логика ставит одну A в самый тяжёлый кластер.
            """
            out = d.copy()
            if len(out) == 0:
                return out

            cluster_col = None
            if "subcluster_id" in out.columns:
                cluster_col = "subcluster_id"
            elif "cluster_id" in out.columns:
                cluster_col = "cluster_id"
            if cluster_col is None:
                out["region_id"] = "nearest_0"
                return out

            centroid_df = out
            if "is_cluster_fill" in centroid_df.columns:
                centroid_df = centroid_df[~centroid_df["is_cluster_fill"].fillna(False).astype(bool)].copy()
            if len(centroid_df) == 0:
                centroid_df = out.copy()

            cluster_rows = centroid_df[[cluster_col, "geometry"]].copy()
            if "weight" in centroid_df.columns:
                cluster_rows["weight"] = centroid_df["weight"].astype(float)
            else:
                cluster_rows["weight"] = 1.0
            cluster_rows = cluster_rows[cluster_rows[cluster_col].notna()].copy()
            if len(cluster_rows) == 0:
                out["region_id"] = "nearest_0"
                return out
            cluster_rows = cluster_rows.sort_values(by="weight", ascending=False).drop_duplicates(subset=[cluster_col])

            # Список уникальных кластеров.
            clus = []
            for _, row in cluster_rows.iterrows():
                lon, lat = _coords_from_geom(row.geometry)
                clus.append(
                    {
                        "cid": row[cluster_col],
                        "lon": float(lon),
                        "lat": float(lat),
                        "weight": float(row.get("weight", 1.0)),
                    }
                )
            if not clus:
                out["region_id"] = "nearest_0"
                return out

            group_size = max(1, int(group_size))
            unassigned = set(range(len(clus)))
            region_for_cid: Dict[Any, str] = {}
            region_idx = 0

            while unassigned:
                # seed: самый тяжёлый ещё не назначенный кластер
                seed = max(unassigned, key=lambda i: clus[i]["weight"])
                seed_lon = clus[seed]["lon"]
                seed_lat = clus[seed]["lat"]

                dist_items = []
                for j in unassigned:
                    d_km = haversine_km(seed_lat, seed_lon, clus[j]["lat"], clus[j]["lon"])
                    dist_items.append((d_km, j))
                dist_items.sort(key=lambda x: x[0])
                picked = [j for _, j in dist_items[:group_size]]
                rid = f"nearest_{region_idx}"
                region_idx += 1
                for j in picked:
                    region_for_cid[clus[j]["cid"]] = rid
                    if j in unassigned:
                        unassigned.remove(j)

            out["region_id"] = out[cluster_col].map(region_for_cid)
            na_mask = out["region_id"].isna()
            if na_mask.any():
                out.loc[na_mask, "region_id"] = "nearest_0"
            return out

        def _assign_synthetic_regions(d: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            """Fallback: если районов 0-1, делим кластеры на синтетические регионы по главной оси."""
            out = d.copy()
            if "region_id" not in out.columns:
                out["region_id"] = None
            if len(out) <= 1:
                out["region_id"] = "region_0"
                return out

            # Берем только реальные центроиды кластеров (без fill-точек), если колонка есть.
            work_idx = out.index
            if "is_cluster_fill" in out.columns:
                mask = ~out["is_cluster_fill"].fillna(False).astype(bool)
                work_idx = out.index[mask]
                if len(work_idx) == 0:
                    work_idx = out.index

            coords = np.array([_coords_from_geom(out.loc[i].geometry) for i in work_idx], dtype=float)
            if len(coords) < 2:
                out["region_id"] = "region_0"
                return out

            centered = coords - coords.mean(axis=0, keepdims=True)
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                axis = vt[0]
            except Exception:
                axis = np.array([1.0, 0.0], dtype=float)
            proj = centered @ axis

            n_points = len(work_idx)
            # Обычно даёт 4-8 "районов", чтобы A не схлопывались в одну станцию.
            k_regions = int(np.clip(round(np.sqrt(n_points)), 2, 8))
            edges = np.quantile(proj, np.linspace(0.0, 1.0, k_regions + 1))
            labels = np.digitize(proj, edges[1:-1], right=False)

            for pos, idx in enumerate(work_idx):
                out.at[idx, "region_id"] = f"synthetic_{int(labels[pos])}"
            # На всякий случай проставим тем, кто не попал в work_idx.
            na_mask = out["region_id"].isna()
            if na_mask.any():
                out.loc[na_mask, "region_id"] = "synthetic_0"
            return out

        try:
            admin_districts = data_service.get_admin_districts(city_name, city_boundary=city_boundary)
        except Exception:
            admin_districts = gpd.GeoDataFrame()

        if admin_districts is not None and len(admin_districts) > 0:
            try:
                d = demand.copy()
                if d.crs is None:
                    d = d.set_crs("EPSG:4326", allow_override=True)
                districts = admin_districts.to_crs(d.crs)
                if "district_name" not in districts.columns:
                    districts["district_name"] = districts.index.astype(str)
                districts = districts[["district_name", "geometry"]].copy()
                districts = districts.rename(columns={"district_name": "region_id"})
                joined = gpd.sjoin(d, districts, how="left", predicate="within")
                if "region_id" in joined.columns:
                    d["region_id"] = joined["region_id"].values
                    no_region = d["region_id"].isna()
                    if no_region.any():
                        d.loc[no_region, "region_id"] = (
                            "unassigned_" + d.index.astype(str)
                        )[no_region]
                else:
                    d["region_id"] = "region_default"
                n_regions = int(pd.Series(d["region_id"]).nunique()) if "region_id" in d.columns else 0
                if n_regions <= 1:
                    d = _assign_synthetic_regions(d)
                    n_regions = int(pd.Series(d["region_id"]).nunique())
                # Пользовательская логика: группы ближайших кластеров (по 5), одна A в самый большой кластер группы.
                d = _repack_regions_by_nearest_clusters(d, group_size=5)
                n_regions = int(pd.Series(d["region_id"]).nunique()) if "region_id" in d.columns else n_regions
                demand = d
                logger.info(
                    "Привязка кластеров к административным районам/группам: %s групп",
                    n_regions,
                )
            except Exception:
                demand = demand.copy()
                demand["region_id"] = "region_default"
                demand = _assign_synthetic_regions(demand)
                demand = _repack_regions_by_nearest_clusters(demand, group_size=5)
        else:
            demand = _assign_synthetic_regions(demand.copy())
            demand = _repack_regions_by_nearest_clusters(demand, group_size=5)

    candidates_rooftop = data_service.get_station_candidates(
        buildings, city_boundary, no_fly_zones, road_graph, station_type="rooftop"
    )
    parts = []
    if candidates_rooftop is not None and len(candidates_rooftop) > 0:
        parts.append(candidates_rooftop)
    # Зарядные станции: только rooftop-кандидаты (МКД).
    charge_candidates_full = (
        gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=(parts[0].crs if parts else "EPSG:4326"))
        if parts
        else gpd.GeoDataFrame()
    )
    if charge_candidates_full is None or len(charge_candidates_full) == 0:
        logger.warning(
            "Кандидатов для зарядок на МКД нет (пустой список после фильтрации OSM). "
            "Проверьте теги building и границу города; слой зарядок будет пустым."
        )

    if demand_method == "dbscan" and charge_candidates_full is not None and len(charge_candidates_full) > 0:
        charge_candidates = placement.build_charge_candidates_from_clusters(
            demand,
            charge_candidates_full,
            candidates_per_cluster=candidates_per_cluster,
        )
    else:
        charge_candidates = charge_candidates_full

    # Hull-полигоны кластеров нужны для строгой и устойчивой привязки станций к "своей" геометрии.
    cluster_hulls_gdf = None
    if demand_method == "dbscan":
        try:
            cluster_hulls_gdf = data.get("demand_hulls") if isinstance(data, dict) else None
        except Exception:
            cluster_hulls_gdf = None
        if cluster_hulls_gdf is None:
            try:
                _, cluster_hulls_gdf = data_service.get_demand_points_weighted(
                    buildings,
                    road_graph,
                    city_boundary,
                    method=demand_method,
                    cell_size_m=demand_cell_m,
                    dbscan_eps_m=dbscan_eps_m,
                    dbscan_min_samples=dbscan_min_samples,
                    use_all_buildings=use_all_buildings,
                    no_fly_zones=no_fly_zones,
                    return_hulls=True,
                    fill_clusters=True,
                    cluster_fill_step_m=None,
                )
            except Exception:
                cluster_hulls_gdf = None

    candidates_for_placement = (
        charge_candidates_full
        if (charge_candidates_full is not None and len(charge_candidates_full) > 0)
        else charge_candidates
    )
    if candidates_for_placement is None or len(candidates_for_placement) == 0:
        candidates_for_placement = charge_candidates

    charge_stations, charge_metrics = placement.place_charging_stations(
        candidates_for_placement,
        demand,
        no_fly_zones,
        radius_km=R_CHARGE_KM,
        cluster_hulls=cluster_hulls_gdf,
        max_stations=max_charge_stations,
        enable_type_b=True,
        type_a_coverage_ratio=0.95,
        type_b_coverage_ratio=1.0,
        min_center_distance_factor_a=2.0,
        min_center_distance_factor_core_a=2.0,
        core_dense_radius_factor=5.0,
        core_radius_factor_a=0.0,
        core_weight_quantile_a=0.0,
        min_center_distance_factor_b=0.75,
        force_type_b_per_cluster=True,
        force_type_a_per_region=True,
        force_type_a_per_cluster=False,
    )
    # --- Корректная привязка зарядок к "своей" геометрии кластера ---
    # Мы делаем hull-first привязку:
    # - если точка станции попадает ровно в один hull -> cluster_id = cluster_id этого hull
    # - если точка попала в 0 или несколько hull -> cluster_id ставим через fallback ближайшего demand-центроида
    cluster_col = None
    if demand is not None and len(demand) > 0:
        if "cluster_id" in demand.columns:
            cluster_col = "cluster_id"
        elif "subcluster_id" in demand.columns:
            cluster_col = "subcluster_id"

    if cluster_col is not None and charge_stations is not None and len(charge_stations) > 0:
        def _as_int_or_none(v: Any) -> Optional[int]:
            try:
                if v is None:
                    return None
                if isinstance(v, float) and np.isnan(v):
                    return None
                return int(v)
            except Exception:
                return None

        assigned: List[Optional[int]] = [None] * len(charge_stations)
        # Если forced-логика уже записала cluster_id, используем это как начальное значение.
        if "cluster_id" in charge_stations.columns:
            try:
                assigned = [_as_int_or_none(charge_stations.iloc[pos].get("cluster_id")) for pos in range(len(charge_stations))]
            except Exception:
                assigned = [None] * len(charge_stations)

        # Верификация по геометрии hull'ов:
        # - если точка станции попала ровно в один hull -> cluster_id ставим по этому hull
        # - иначе -> cluster_id=None (не привязываем к соседнему кластеру)
        if cluster_hulls_gdf is not None and len(cluster_hulls_gdf) > 0 and "cluster_id" in cluster_hulls_gdf.columns:
            try:
                hulls_gdf = cluster_hulls_gdf
                hulls_gdf = hulls_gdf[~hulls_gdf["geometry"].isna() & ~hulls_gdf["geometry"].is_empty].copy()
                if len(hulls_gdf) > 0:
                    sindex = getattr(hulls_gdf, "sindex", None)
                    for pos, (_, srow) in enumerate(charge_stations.iterrows()):
                        # Если cluster_id уже задан forced-логикой, сохраняем его как "истину":
                        # иначе последующая пере-привязка по геометрии может схлопнуть разные кластеры
                        # в один и потерять станции после dedup.
                        if assigned[pos] is not None:
                            continue

                        lon_s, lat_s = _coords_from_geom(srow.geometry)
                        pt = Point(lon_s, lat_s)

                        idxs = None
                        if sindex is not None:
                            try:
                                # См. комментарий выше: через spatial index используем intersects,
                                # а точную проверку принадлежности делаем geom.covers(pt).
                                idxs = list(sindex.query(pt, predicate="intersects"))
                            except Exception:
                                idxs = None
                        if idxs is None:
                            idxs = range(len(hulls_gdf))

                        match_cids: List[int] = []
                        for hi in idxs:
                            geom = hulls_gdf.geometry.iloc[int(hi)]
                            if geom is None or not geom.is_valid:
                                continue
                            try:
                                if bool(geom.covers(pt)):
                                    match_cids.append(int(hulls_gdf.iloc[int(hi)]["cluster_id"]))
                            except Exception:
                                try:
                                    if bool(geom.contains(pt)):
                                        match_cids.append(int(hulls_gdf.iloc[int(hi)]["cluster_id"]))
                                except Exception:
                                    pass

                        unique_cids = set(match_cids)
                        if len(unique_cids) == 1:
                            assigned[pos] = next(iter(unique_cids))
                        else:
                            # Неоднозначно (0 или несколько hull покрывают точку).
                            # Раньше мы намеренно ставили `None`, чтобы не допустить ошибочной привязки,
                            # но это приводит к тому, что целые кластеры выглядят "без станций".
                            # Поэтому в любом неоднозначном случае назначаем ближайший hull по расстоянию.
                            try:
                                best_cid: Optional[int] = None
                                best_d: Optional[float] = None

                                # Если несколько hulls совпали, ограничим выбор только ими.
                                # Если 0 совпадений — ищем ближайший среди всех hulls.
                                cids_to_check: Optional[set] = unique_cids if len(unique_cids) > 0 else None

                                for hi in range(len(hulls_gdf)):
                                    cid_h = int(hulls_gdf.iloc[int(hi)]["cluster_id"])
                                    if cids_to_check is not None and cid_h not in cids_to_check:
                                        continue

                                    geom = hulls_gdf.geometry.iloc[int(hi)]
                                    if geom is None or not geom.is_valid:
                                        continue

                                    d = float(geom.distance(pt))
                                    if best_d is None or d < best_d:
                                        best_d = d
                                        best_cid = cid_h

                                # Если даже это не получилось — оставляем текущую (initial) привязку.
                                assigned[pos] = best_cid if best_cid is not None else assigned[pos]
                            except Exception:
                                # На худой случай оставляем initial привязку.
                                assigned[pos] = assigned[pos]
            except Exception:
                pass

        # Если точка станции не однозначно попала в hull — cluster_id=None, чтобы не допускать "привязку к соседнему кластеру".

        charge_stations = charge_stations.copy()
        charge_stations["cluster_id"] = assigned

        # Гарантия: максимум одна зарядка на один cluster_id.
        # При конфликте оставляем приоритетно charge_a, затем charge_b.
        if "cluster_id" in charge_stations.columns and len(charge_stations) > 0:
            stype = (
                charge_stations["station_type"]
                if "station_type" in charge_stations.columns
                else pd.Series([""] * len(charge_stations), index=charge_stations.index)
            )
            priority = np.where(stype == "charge_a", 0, np.where(stype == "charge_b", 1, 2))
            charge_stations = charge_stations.assign(_cluster_priority=priority)
            non_null = charge_stations[charge_stations["cluster_id"].notna()]
            nulls = charge_stations[~charge_stations["cluster_id"].notna()]
            non_null = non_null.sort_values(by="_cluster_priority").drop_duplicates(
                subset=["cluster_id"], keep="first"
            )
            charge_stations = pd.concat([non_null, nulls], ignore_index=True).drop(
                columns=["_cluster_priority"], errors="ignore"
            )

        # Сохраняем станции даже при None cluster_id:
        # - иначе можно полностью потерять размещение, если hull'ы не покрывают точные координаты кандидатов
        # - UI/линии привязки при None просто не смогут построиться корректно, что безопаснее неверного cluster_id.

    # Проставляем region_id станциям через cluster_id demand-кластеров.
    if (
        charge_stations is not None
        and len(charge_stations) > 0
        and demand is not None
        and len(demand) > 0
        and "region_id" in demand.columns
    ):
        try:
            cluster_key = None
            if "cluster_id" in demand.columns:
                cluster_key = "cluster_id"
            elif "subcluster_id" in demand.columns:
                cluster_key = "subcluster_id"
            if cluster_key is not None and "cluster_id" in charge_stations.columns:
                cluster_to_region: Dict[Any, Any] = {}
                for _, row in demand.iterrows():
                    cid = row.get(cluster_key)
                    rid = row.get("region_id")
                    if pd.isna(cid) or pd.isna(rid):
                        continue
                    cluster_to_region[cid] = rid
                charge_stations = charge_stations.copy()
                charge_stations["region_id"] = charge_stations["cluster_id"].map(cluster_to_region)
        except Exception:
            pass

    # --- Информация о кластерах и “зданиях зарядки” ---
    cluster_count = 0
    charging_buildings_in_clusters = 0
    if demand is not None and len(demand) > 0:
        if "cluster_id" in demand.columns:
            cluster_count = int(demand["cluster_id"].nunique())
        elif "subcluster_id" in demand.columns:
            cluster_count = int(demand["subcluster_id"].nunique())
        else:
            cluster_count = int(len(demand))

        if "n_buildings" in demand.columns:
            if "is_cluster_fill" in demand.columns:
                mask = ~demand["is_cluster_fill"].fillna(False).astype(bool)
                charging_buildings_in_clusters = int(demand.loc[mask, "n_buildings"].sum())
            else:
                charging_buildings_in_clusters = int(demand["n_buildings"].sum())

    # --- Гаражи и ТО: ceil(N_a/3) каждого типа; связь с ближайшими A как у B→A (ветка на карте) ---
    obstacles_union = None
    if no_fly_zones is not None:
        try:
            if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                nz = no_fly_zones.to_crs("EPSG:4326")
                obstacles_union = unary_union([g for g in nz.geometry if g is not None and not g.is_empty])
            elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                obstacles_union = unary_union([g for g in no_fly_zones if g is not None and not g.is_empty])
        except Exception:
            obstacles_union = None

    # Маршрутизация и обходы — по геометрии no-fly из данных, без дополнительного метрического буфера.
    # Эшелоны 4–5: магистраль А–А, ветки Б→А, локальные Б↔Б, гараж→А, ТО→А — высокие здания как NFZ (см. DataService).
    obstacles_routing = obstacles_union
    try:
        from data_service import DataService

        mag_min_h = DataService.high_altitude_station_routing_obstacle_min_height_m(data.get("flight_levels"))
        tall_union = buildings_footprint_union_min_height_wgs84(buildings, mag_min_h)
        if tall_union is not None and not getattr(tall_union, "is_empty", True):
            if obstacles_routing is not None and not getattr(obstacles_routing, "is_empty", True):
                obstacles_routing = unary_union([obstacles_routing, tall_union])
            else:
                obstacles_routing = tall_union
    except Exception as e:
        logger.warning("Объединение высоких зданий с NFZ для маршрутизации пропущено: %s", e)

    n_a_for_facilities = 0
    if (
        charge_stations is not None
        and len(charge_stations) > 0
        and "station_type" in charge_stations.columns
    ):
        n_a_for_facilities = int((charge_stations["station_type"] == "charge_a").sum())
    n_garage_to = (n_a_for_facilities + 2) // 3 if n_a_for_facilities > 0 else 0

    garage_candidates_fc = data_service.get_station_candidates(
        buildings, city_boundary, no_fly_zones, road_graph, station_type="garage"
    )
    to_candidates_fc = data_service.get_station_candidates(
        buildings, city_boundary, no_fly_zones, road_graph, station_type="to"
    )
    if n_garage_to > 0:
        busy_lonlat = _lonlat_exclusions_from_gdf(charge_stations)
        garages = _place_n_facilities_by_demand_coverage(
            demand,
            garage_candidates_fc,
            n=n_garage_to,
            radius_km=R_GARAGE_TO_KM,
            station_type_label="garage",
            log=logger,
            min_separation_km=MIN_FACILITY_SEPARATION_KM,
            exclude_lonlat=busy_lonlat,
        )
        busy_lonlat = busy_lonlat + _lonlat_exclusions_from_gdf(garages)
        # Тот же принцип, что и гараж; плюс разнос от зарядок и от уже выбранных гаражей.
        to_stations = _place_n_facilities_by_demand_coverage(
            demand,
            to_candidates_fc,
            n=n_garage_to,
            radius_km=R_GARAGE_TO_KM,
            station_type_label="to",
            log=logger,
            min_separation_km=MIN_FACILITY_SEPARATION_KM,
            exclude_lonlat=busy_lonlat,
        )
    else:
        garages = gpd.GeoDataFrame()
        to_stations = gpd.GeoDataFrame()

    # Магистраль на карте — только между зарядками A. Ветки спутник→A в пайплайне не строятся.
    # Воздушный граф БПЛА строим отдельно как сетку свободного пространства,
    # чтобы detour обходил no-fly зоны "по воздуху", а не по дорогам.
    uav_air_graph = None
    try:
        nfz_for_air = obstacles_routing if obstacles_routing is not None else no_fly_zones
        b_union_air = buildings_footprint_union_wgs84(buildings)
        uav_air_graph = build_uav_air_graph_grid(
            city_boundary,
            nfz_for_air,
            building_union=b_union_air,
            grid_spacing_km=UAV_AIR_GRID_SPACING_KM,
            connect_diagonal=True,
            max_nodes=UAV_AIR_GRID_MAX_NODES,
            safety_buffer=0.0,
        )
        if uav_air_graph is None or uav_air_graph.number_of_nodes() < 2:
            uav_air_graph = None
    except Exception as e:
        logger.warning("Не удалось построить воздушный граф для обходов: %s", e)
        uav_air_graph = None

    trunk = placement.build_trunk_graph(
        charge_stations,
        gpd.GeoDataFrame(),
        gpd.GeoDataFrame(),
        obstacles_routing,
        air_graph=uav_air_graph,
        obstacles_relaxed=obstacles_union,
        max_edge_km=R_GARAGE_TO_KM,
        max_neighbors_a=4,
        topology_mode="mst",
        extra_paths_per_a=2,
    )

    if charge_stations is not None and len(charge_stations) > 0:
        charge_stations = charge_stations.copy()
        charge_stations["trunk_degree"] = 0
        if trunk is not None and trunk.number_of_nodes() > 0:
            for pos, idx in enumerate(charge_stations.index):
                nid = f"charge_{idx}"
                if trunk.has_node(nid):
                    charge_stations.iloc[pos, charge_stations.columns.get_loc("trunk_degree")] = int(trunk.degree[nid])
    cs_for_edges = charge_stations if charge_stations is not None else gpd.GeoDataFrame()
    branch_edges = placement.build_branch_edges(
        cs_for_edges,
        max_b_per_type_a=MAX_TYPE_B_BRANCHES_PER_TYPE_A,
        max_branch_km=R_GARAGE_TO_KM,
        obstacles=obstacles_routing,
        air_graph=uav_air_graph,
        no_fly_safety_buffer=0.0,
    )
    branch_edges.extend(
        placement.build_facility_branch_edges(
            cs_for_edges,
            garages if garages is not None and len(garages) > 0 else None,
            to_stations if to_stations is not None and len(to_stations) > 0 else None,
            max_branch_km=R_GARAGE_TO_KM,
            obstacles=obstacles_routing,
            air_graph=uav_air_graph,
            no_fly_safety_buffer=0.0,
        )
    )
    local_edges = placement.build_local_edges(
        cs_for_edges,
        obstacles_routing,
        uav_air_graph,
        max_edge_km=R_GARAGE_TO_KM,
        max_neighbors_b=2,
        no_fly_safety_buffer=0.0,
    )

    from voronoi_paths import build_voronoi_local_paths_fc, charging_station_gdfs_to_features

    st_voronoi_features = charging_station_gdfs_to_features(
        charge_stations if charge_stations is not None and len(charge_stations) > 0 else None,
        garages if garages is not None and len(garages) > 0 else None,
        to_stations if to_stations is not None and len(to_stations) > 0 else None,
    )
    st_voronoi_arg: Optional[List[Dict[str, Any]]] = st_voronoi_features if st_voronoi_features else None
    try:
        voronoi_edges = build_voronoi_local_paths_fc(
            data_service,
            city_name,
            network_type,
            simplify,
            dbscan_eps_m,
            dbscan_min_samples,
            use_all_buildings,
            buildings_per_centroid=voronoi_buildings_per_centroid,
            charging_station_features=st_voronoi_arg,
            inter_cluster_max_hull_gap_m=inter_cluster_max_hull_gap_m,
            inter_cluster_max_edge_length_m=inter_cluster_max_edge_length_m,
            voronoi_intra_component_bridge_max_m=voronoi_intra_component_bridge_max_m,
            city_data=data,
        )
    except Exception as e:
        logger.warning("Вороной в placement: не удалось построить (%s)", e)
        voronoi_edges = {"type": "FeatureCollection", "features": []}

    metrics = placement.compute_metrics(demand, charge_stations, trunk)
    metrics["charge"] = charge_metrics
    metrics["cluster_count"] = cluster_count
    metrics["charging_buildings_in_clusters"] = charging_buildings_in_clusters
    vf = list((voronoi_edges or {}).get("features") or [])
    metrics["voronoi_edges_total"] = len(vf)
    vbe = (voronoi_edges or {}).get("voronoi_by_echelon") or {}
    if isinstance(vbe, dict) and vbe:
        metrics["voronoi_edges_by_echelon"] = {
            str(k): len((sub or {}).get("features") or []) for k, sub in vbe.items()
        }
    fe_thr = (voronoi_edges or {}).get("flight_echelon_building_obstacle_min_height_m")
    if isinstance(fe_thr, dict) and fe_thr:
        metrics["flight_echelon_building_obstacle_min_height_m"] = {
            int(k): float(v) for k, v in fe_thr.items()
        }
    metrics["voronoi_clusters_with_paths"] = int((voronoi_edges or {}).get("clusters_with_paths") or 0)
    metrics["voronoi_clusters_total"] = int((voronoi_edges or {}).get("clusters_total") or 0)

    return {
        "buildings": buildings,
        "demand": demand,
        "charge_stations": charge_stations,
        "garages": garages,
        "to_stations": to_stations,
        "trunk_graph": trunk,
        "branch_edges": branch_edges,
        "local_edges": local_edges,
        "voronoi_edges": voronoi_edges,
        "flight_levels": data.get("flight_levels") or [],
        "metrics": metrics,
        "city_boundary": city_boundary,
        "no_fly_zones": no_fly_zones,
        "demand_hulls": cluster_hulls_gdf,
        "params": {
            "voronoi_buildings_per_centroid": int(voronoi_buildings_per_centroid),
            "voronoi_includes_station_sites": bool(st_voronoi_arg),
            "inter_cluster_max_hull_gap_m": float(inter_cluster_max_hull_gap_m),
            "inter_cluster_max_edge_length_m": float(inter_cluster_max_edge_length_m),
            "voronoi_mode": "full_in_placement",
            "voronoi_intra_component_bridge_max_m": float(voronoi_intra_component_bridge_max_m),
        },
    }


def _empty_fc() -> Dict[str, Any]:
    """Пустой GeoJSON FeatureCollection."""
    return {"type": "FeatureCollection", "features": []}


def _gdf_to_fc(gdf: Optional[gpd.GeoDataFrame]) -> Dict[str, Any]:
    """GeoDataFrame → GeoJSON FeatureCollection (сериализуемые свойства)."""
    if gdf is None or len(gdf) == 0:
        return _empty_fc()
    from shapely.geometry import mapping

    features: List[Dict[str, Any]] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or getattr(geom, "is_empty", True):
            continue
        props: Dict[str, Any] = {}
        for col in gdf.columns:
            if col == "geometry":
                continue
            val = row[col]
            if pd.isna(val):
                props[col] = None
            elif hasattr(val, "item"):
                try:
                    props[col] = val.item()
                except Exception:
                    props[col] = str(val)
            else:
                props[col] = val
        features.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return {"type": "FeatureCollection", "features": features}


def _list_features_to_fc(features: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Оборачивает список GeoJSON-фич в FeatureCollection."""
    if not features:
        return _empty_fc()
    return {"type": "FeatureCollection", "features": list(features)}


def trunk_graph_to_geojson_features(trunk_graph: Optional[nx.Graph]) -> List[Dict[str, Any]]:
    """Рёбра магистрали → GeoJSON Features (LineString); при наличии geometry_coords — полилиния."""
    if trunk_graph is None or trunk_graph.number_of_edges() == 0:
        return []
    feats: List[Dict[str, Any]] = []
    for u, v, data in trunk_graph.edges(data=True):
        coords = data.get("geometry_coords")
        if coords is not None and len(coords) >= 2:
            line_coords = [[float(c[0]), float(c[1])] for c in coords]
        else:
            du = trunk_graph.nodes[u]
            dv = trunk_graph.nodes[v]
            lon1 = du.get("lon")
            lat1 = du.get("lat")
            lon2 = dv.get("lon")
            lat2 = dv.get("lat")
            if lon1 is None or lat1 is None or lon2 is None or lat2 is None:
                continue
            line_coords = [[float(lon1), float(lat1)], [float(lon2), float(lat2)]]
        w = data.get("weight", data.get("length"))
        trm = data.get("trunk_route_mode")
        props: Dict[str, Any] = {
            "source": str(u),
            "target": str(v),
            "weight_km": float(w) if w is not None else None,
        }
        if trm is not None:
            props["trunk_route_mode"] = str(trm)
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": line_coords},
                "properties": props,
            }
        )
    return feats


def _annotate_station_fc_vertical_echelon_links(fc: Dict[str, Any]) -> None:
    """Для карты/клиента: зарядки A/B — узлы стыковки маршрутов по разным эшелонам."""
    if not fc or not isinstance(fc, dict):
        return
    for f in fc.get("features") or []:
        if not isinstance(f, dict):
            continue
        p = f.setdefault("properties", {})
        if not isinstance(p, dict):
            continue
        p["vertical_echelons_connected"] = [1, 2, 3, 4, 5]
        p["vertical_planning_note"] = (
            "Стыковка эшелонов 1–5: на станции можно планировать плавный вертикальный спуск/подъём между "
            "локальными маршрутами (эшелоны 1–3) и магистралью с ветками (4–5)."
        )


# Этап 12-13: упаковка результата в GeoJSON-слои для карты.
def pipeline_result_to_geojson(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ответ API `/api/stations/placement`: слои для карты + метрики и параметры."""
    err = raw.get("error")
    charge = raw.get("charge_stations")
    charging_b = gpd.GeoDataFrame()
    if charge is not None and len(charge) > 0 and "station_type" in charge.columns:
        charging_a = charge[charge["station_type"] == "charge_a"]
        charging_b = charge[charge["station_type"] == "charge_b"]
    elif charge is not None and len(charge) > 0:
        # На случай старых схем — если станций A/B нет, считаем что пришедший GeoDataFrame и есть А.
        charging_a = charge
    else:
        charging_a = gpd.GeoDataFrame()

    trunk_feats = trunk_graph_to_geojson_features(raw.get("trunk_graph"))
    empty_fc = _empty_fc()
    demand = raw.get("demand")
    cluster_centroids = None
    if demand is not None and len(demand) > 0:
        demand_for_link = demand
        if "is_cluster_fill" in demand_for_link.columns:
            demand_for_link = demand_for_link[~demand_for_link["is_cluster_fill"].fillna(False).astype(bool)].copy()

        cid_col = None
        if "cluster_id" in demand_for_link.columns:
            cid_col = "cluster_id"
        elif "subcluster_id" in demand_for_link.columns:
            cid_col = "subcluster_id"

        if cid_col is not None and cid_col in demand_for_link.columns and len(demand_for_link) > 0:
            demand_for_link = demand_for_link[demand_for_link[cid_col].notna()].copy()
            if len(demand_for_link) > 0:
                if "weight" in demand_for_link.columns:
                    demand_for_link = demand_for_link.sort_values(by="weight", ascending=False)
                # Оставляем по 1 точке на кластер (для UI достаточно координат центра).
                demand_for_link = demand_for_link.drop_duplicates(subset=[cid_col])
                if cid_col != "cluster_id":
                    demand_for_link = demand_for_link.rename(columns={cid_col: "cluster_id"})

                keep_cols = ["cluster_id"]
                for extra in ("weight", "n_buildings", "region_id"):
                    if extra in demand_for_link.columns:
                        keep_cols.append(extra)
                cluster_centroids = demand_for_link[[*keep_cols, "geometry"]]

    ve = raw.get("voronoi_edges")
    if isinstance(ve, dict) and ve.get("type") == "FeatureCollection":
        voronoi_fc: Dict[str, Any] = {
            "type": "FeatureCollection",
            "features": list(ve.get("features") or []),
        }
    else:
        voronoi_fc = empty_fc

    voronoi_by_echelon_out: Dict[str, Any] = {}
    if isinstance(ve, dict):
        vbe = ve.get("voronoi_by_echelon")
        if isinstance(vbe, dict):
            for sk, sub in vbe.items():
                if not isinstance(sub, dict):
                    continue
                voronoi_by_echelon_out[str(sk)] = {
                    "type": "FeatureCollection",
                    "features": list(sub.get("features") or []),
                    "voronoi_echelon_level": sub.get("voronoi_echelon_level"),
                    "voronoi_echelon_altitude_m": sub.get("voronoi_echelon_altitude_m"),
                    "voronoi_building_obstacle_min_height_m": sub.get(
                        "voronoi_building_obstacle_min_height_m"
                    ),
                }

    out: Dict[str, Any] = {
        "charging_type_a": _gdf_to_fc(charging_a),
        "charging_type_b": _gdf_to_fc(charging_b),
        "garages": _gdf_to_fc(raw.get("garages")),
        "to_stations": _gdf_to_fc(raw.get("to_stations")),
        "trunk": {"type": "FeatureCollection", "features": trunk_feats},
        "branch_edges": _list_features_to_fc(raw.get("branch_edges")),
        "local_edges": _list_features_to_fc(raw.get("local_edges")),
        "voronoi_edges": voronoi_fc,
        **(
            {"voronoi_edges_by_echelon": voronoi_by_echelon_out}
            if voronoi_by_echelon_out
            else {}
        ),
        **(
            {
                "flight_echelon_building_obstacle_min_height_m": ve.get(
                    "flight_echelon_building_obstacle_min_height_m"
                )
            }
            if isinstance(ve, dict)
            and isinstance(ve.get("flight_echelon_building_obstacle_min_height_m"), dict)
            else {}
        ),
        "flight_levels": raw.get("flight_levels") or [],
        "metrics": raw.get("metrics") or {},
        "params": raw.get("params") or {},
        "cluster_centroids": _gdf_to_fc(cluster_centroids),
    }
    _annotate_station_fc_vertical_echelon_links(out["charging_type_a"])
    _annotate_station_fc_vertical_echelon_links(out["charging_type_b"])
    if err:
        out["error"] = err
    return out
