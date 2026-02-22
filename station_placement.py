# -*- coding: utf-8 -*-
"""
Размещение станций зарядки, гаражей и ТО по правилам:
- Зарядка+ожидание (туда-обратно): R_charge = 0.4 * D_max = 6.4 км (FlyCart 30, D_max=16 км).
- Гаражи/ТО (в одну сторону): R_garage/TO = 0.8 * D_max = 12.8 км.

Шаги:
  1) Точки спроса (кластеры/сетка 250×250 м с весом).
  2) Зарядки: weighted maximum coverage / set cover (жадный).
  3) Магистральный слой: узлы Charge_A + гаражи/ТО, рёбра kNN/Delaunay.
  4) Гаражи: k-median по спросу, только промзоны.
  5) ТО: betweenness centrality по trunk + промзона.
  6) Локальная сеть и метрики.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from scipy.spatial import Delaunay, cKDTree

# Радиус Земли в км
EARTH_R_KM = 6371.0

# FlyCart 30 с грузом: D_max = 16 км (радиусы уменьшены в 4 раза для отображения/расчёта)
D_MAX_KM = 16.0
# Зарядка + ожидание (туда-обратно): R = 0.4 * D_max / 4 = 1.6 км
R_CHARGE_KM = 0.4 * D_MAX_KM / 4.0  # 1.6 км
# Гаражи и ТО (в одну сторону): R = 0.8 * D_max / 4 = 3.2 км
R_GARAGE_TO_KM = 0.8 * D_MAX_KM / 4.0  # 3.2 км


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
    """(lon, lat) из геометрии (Point или centroid)."""
    if g is None:
        return (0.0, 0.0)
    if hasattr(g, "x") and hasattr(g, "y"):
        return (float(g.x), float(g.y))
    c = getattr(g, "centroid", None)
    if c is not None:
        return (float(c.x), float(c.y))
    return (0.0, 0.0)


def _points_array(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Массив (N, 2) lon, lat из GeoDataFrame."""
    if gdf is None or len(gdf) == 0:
        return np.empty((0, 2))
    coords = []
    for _, row in gdf.iterrows():
        g = row.geometry
        coords.append(_coords_from_geom(g))
    return np.array(coords)


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


class StationPlacement:
    def __init__(self, data_service, graph_service, logger=None):
        self.data_service = data_service
        self.graph_service = graph_service
        self.logger = logger or logging.getLogger(__name__)

    # --- Шаг 1: точки спроса ---
    def build_demand_points(
        self,
        buildings: gpd.GeoDataFrame,
        road_graph,
        city_boundary,
        *,
        method: str = "grid",
        cell_size_m: float = 250.0,
    ) -> gpd.GeoDataFrame:
        """Точки спроса: кластеры или ячейки плотности с весом."""
        return self.data_service.get_demand_points_weighted(
            buildings,
            road_graph,
            city_boundary,
            method=method,
            cell_size_m=cell_size_m,
            use_delivery_points=True,
        )

    # --- Шаг 2: зарядки (weighted set cover) ---
    def place_charging_stations(
        self,
        candidates: gpd.GeoDataFrame,
        demand: gpd.GeoDataFrame,
        no_fly_zones,
        *,
        radius_km: float = R_CHARGE_KM,
        max_stations: Optional[int] = None,
        min_coverage_ratio: float = 0.99,
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Жадно выбирает зарядки: на каждом шаге — кандидат, покрывающий максимум ещё непокрытого веса спроса.
        Возвращает (GeoDataFrame выбранных зарядок, метрики).
        """
        if candidates is None or len(candidates) == 0 or demand is None or len(demand) == 0:
            return gpd.GeoDataFrame(), {"placed": 0, "demand_covered": 0, "demand_total": 0, "coverage_ratio": 0.0}

        cand_pts = _points_array(candidates)
        dem_pts = _points_array(demand)
        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))

        n_demand = len(dem_pts)
        covered = np.zeros(n_demand, dtype=bool)
        total_weight = float(weights.sum())
        selected = []
        selected_indices = []

        while True:
            best_idx = -1
            best_new_weight = 0.0
            for i in range(len(cand_pts)):
                lon_c, lat_c = cand_pts[i, 0], cand_pts[i, 1]
                # Не ставим зарядку в радиус другой уже выбранной зарядки
                if selected_indices:
                    if any(
                        haversine_km(lat_c, lon_c, cand_pts[s, 1], cand_pts[s, 0]) <= radius_km
                        for s in selected_indices
                    ):
                        continue
                new_weight = 0.0
                for j in range(n_demand):
                    if covered[j]:
                        continue
                    d_km = haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0])
                    if d_km <= radius_km:
                        new_weight += weights[j]
                if new_weight > best_new_weight:
                    best_new_weight = new_weight
                    best_idx = i
            if best_idx < 0 or best_new_weight <= 0:
                break
            selected_indices.append(best_idx)
            lon_c, lat_c = cand_pts[best_idx, 0], cand_pts[best_idx, 1]
            for j in range(n_demand):
                if covered[j]:
                    continue
                if haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0]) <= radius_km:
                    covered[j] = True
            selected.append({
                "geometry": Point(cand_pts[best_idx, 0], cand_pts[best_idx, 1]),
                "station_type": "charge",
                "source_index": best_idx,
            })
            if max_stations is not None and len(selected) >= max_stations:
                break
            covered_weight = weights[covered].sum()
            if total_weight > 0 and covered_weight / total_weight >= min_coverage_ratio:
                break

        gdf = gpd.GeoDataFrame(selected, crs=candidates.crs) if selected else gpd.GeoDataFrame()
        covered_weight = weights[covered].sum()
        metrics = {
            "placed": len(selected),
            "demand_covered": float(covered_weight),
            "demand_total": float(total_weight),
            "coverage_ratio": float(covered_weight / total_weight) if total_weight > 0 else 0.0,
        }
        self.logger.info(f"Зарядки: размещено {len(selected)}, покрытие спроса {metrics['coverage_ratio']:.2%}")
        return gdf, metrics

    # --- Шаг 3: магистральный граф (узлы: Charge_A + гаражи/ТО; рёбра: Delaunay/kNN) ---
    def build_trunk_graph(
        self,
        charge_stations: gpd.GeoDataFrame,
        garage_points: gpd.GeoDataFrame,
        to_points: gpd.GeoDataFrame,
        no_fly_zones,
        *,
        max_edge_km: float = R_GARAGE_TO_KM,
        k_neighbors: int = 5,
    ) -> nx.Graph:
        """Строит граф магистрали: узлы — зарядки (Charge_A), гаражи, ТО; рёбра — Delaunay/kNN, длина не больше max_edge_km."""
        nodes_list = []
        node_ids = []
        # Зарядки
        for i, row in (charge_stations.iterrows() if charge_stations is not None and len(charge_stations) > 0 else []):
            g = row.geometry
            nodes_list.append((_coords_from_geom(g), "charge", f"charge_{i}"))
            node_ids.append(f"charge_{i}")
        # Гаражи
        for i, row in (garage_points.iterrows() if garage_points is not None and len(garage_points) > 0 else []):
            g = row.geometry
            nodes_list.append((_coords_from_geom(g), "garage", f"garage_{i}"))
            node_ids.append(f"garage_{i}")
        # ТО
        for i, row in (to_points.iterrows() if to_points is not None and len(to_points) > 0 else []):
            g = row.geometry
            nodes_list.append((_coords_from_geom(g), "to", f"to_{i}"))
            node_ids.append(f"to_{i}")

        if len(nodes_list) < 2:
            G = nx.Graph()
            for (lon, lat), typ, nid in nodes_list:
                G.add_node(nid, lon=lon, lat=lat, node_type=typ)
            return G

        pts = np.array([[c[0][0], c[0][1]] for c in nodes_list])
        id_by_idx = [c[2] for c in nodes_list]

        # Рёбра: Delaunay + фильтр по длине
        G = nx.Graph()
        for (lon, lat), typ, nid in nodes_list:
            G.add_node(nid, lon=lon, lat=lat, node_type=typ)

        try:
            tri = Delaunay(pts)
            for simplex in tri.simplices:
                for i in range(3):
                    u, v = simplex[i], simplex[(i + 1) % 3]
                    uid, vid = id_by_idx[u], id_by_idx[v]
                    if G.has_edge(uid, vid):
                        continue
                    lat1, lon1 = pts[u, 1], pts[u, 0]
                    lat2, lon2 = pts[v, 1], pts[v, 0]
                    d_km = haversine_km(lat1, lon1, lat2, lon2)
                    if d_km <= max_edge_km:
                        G.add_edge(uid, vid, weight=d_km, length=d_km)
        except Exception as e:
            self.logger.warning(f"Delaunay trunk: {e}, fallback to kNN")
            tree = cKDTree(pts)
            for i in range(len(pts)):
                uid = id_by_idx[i]
                lat1, lon1 = pts[i, 1], pts[i, 0]
                dists, idxs = tree.query(pts, k=min(k_neighbors + 1, len(pts)))
                if not hasattr(dists, "__len__"):
                    dists, idxs = [dists], [idxs]
                for j, (d_deg, jj) in enumerate(zip(dists, idxs)):
                    if jj == i or jj >= len(id_by_idx):
                        continue
                    d_km = haversine_km(lat1, lon1, pts[jj, 1], pts[jj, 0])
                    if d_km <= max_edge_km:
                        vid = id_by_idx[jj]
                        if not G.has_edge(uid, vid):
                            G.add_edge(uid, vid, weight=d_km, length=d_km)

        self.logger.info(f"Trunk: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
        return G

    # --- Шаг 4: гаражи (k-median по спросу, только промзоны) ---
    def place_garages(
        self,
        demand: gpd.GeoDataFrame,
        garage_candidates: gpd.GeoDataFrame,
        *,
        k: int = 3,
        radius_km: float = R_GARAGE_TO_KM,
    ) -> gpd.GeoDataFrame:
        """k-median: выбираем k кандидатов (промзоны), минимизируя взвешенную сумму расстояний от спроса до ближайшего гаража."""
        if garage_candidates is None or len(garage_candidates) == 0:
            return gpd.GeoDataFrame()
        if demand is None or len(demand) == 0:
            return garage_candidates.iloc[:k].copy()

        dem_pts = _points_array(demand)
        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
        cand_pts = _points_array(garage_candidates)

        # Жадно: по одному добавляем гараж, который сильнее всего уменьшает целевую функцию
        n_cand = len(cand_pts)
        chosen = []
        # Ближайший выбранный гараж для каждого спроса (индекс в chosen)
        nearest = np.full(len(dem_pts), -1)
        best_dist = np.full(len(dem_pts), np.inf)

        for _ in range(min(k, n_cand)):
            best_cand = -1
            best_cost = np.inf
            for c in range(n_cand):
                if c in chosen:
                    continue
                # Гараж не должен попадать в радиус другого уже выбранного гаража
                if chosen:
                    if any(
                        haversine_km(cand_pts[c, 1], cand_pts[c, 0], cand_pts[prev, 1], cand_pts[prev, 0]) <= radius_km
                        for prev in chosen
                    ):
                        continue
                lat_c, lon_c = cand_pts[c, 1], cand_pts[c, 0]
                cost = 0.0
                for j in range(len(dem_pts)):
                    d = haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0])
                    if d > radius_km:
                        cost += weights[j] * radius_km * 2  # штраф за непокрытие
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
        for idx in chosen:
            row = garage_candidates.iloc[idx].copy()
            row["station_type"] = "garage"
            rows.append(row)
        out = gpd.GeoDataFrame(rows, crs=garage_candidates.crs)
        self.logger.info(f"Гаражи: размещено {len(out)}")
        return out

    # --- Шаг 5: станции ТО (betweenness по trunk + только промзона) ---
    def place_to_stations(
        self,
        trunk_graph: nx.Graph,
        to_candidates: gpd.GeoDataFrame,
        *,
        k: int = 2,
        radius_km: float = R_GARAGE_TO_KM,
    ) -> gpd.GeoDataFrame:
        """Выбираем до k станций ТО среди кандидатов в промзоне, максимизируя betweenness centrality на trunk-графе.
        Станция ТО не попадает в радиус другой уже выбранной станции ТО."""
        if to_candidates is None or len(to_candidates) == 0 or trunk_graph is None or trunk_graph.number_of_nodes() == 0:
            return gpd.GeoDataFrame()

        # Betweenness по trunk (на узлах charge/garage)
        try:
            between = nx.betweenness_centrality(trunk_graph, weight="weight")
        except Exception:
            between = {n: 0.0 for n in trunk_graph.nodes()}

        # Привязываем кандидатов ТО к ближайшим узлам trunk и наследуют их betweenness
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
            return to_candidates.iloc[:k].copy()

        trunk_pts = np.array(trunk_pts)
        tree = cKDTree(trunk_pts)
        scores = []
        for i in range(len(cand_pts)):
            res = tree.query(cand_pts[i], k=1)
            idx = int(res[1]) if hasattr(res[1], "__int__") else res[1]
            if 0 <= idx < len(trunk_node_ids):
                nid = trunk_node_ids[idx]
                scores.append((i, between.get(nid, 0.0)))
            else:
                scores.append((i, 0.0))
        scores.sort(key=lambda x: -x[1])
        # Жадно берём по убыванию betweenness, но не добавляем ТО в радиус уже выбранной ТО
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

    # --- Шаг 6: метрики (покрытие, средняя стоимость, связность) ---
    def compute_metrics(
        self,
        demand: gpd.GeoDataFrame,
        charge_stations: gpd.GeoDataFrame,
        trunk_graph: nx.Graph,
        radius_charge_km: float = R_CHARGE_KM,
    ) -> Dict[str, Any]:
        """Покрытие спроса зарядками, средняя взвешенная дистанция до ближайшей зарядки, связность trunk."""
        metrics = {"coverage_ratio": 0.0, "avg_distance_to_charge_km": None, "trunk_connected": False}
        if demand is None or len(demand) == 0 or charge_stations is None or len(charge_stations) == 0:
            return metrics
        dem_pts = _points_array(demand)
        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
        charge_pts = _points_array(charge_stations)
        total_weight = float(weights.sum())
        covered_weight = 0.0
        sum_dist = 0.0
        for j in range(len(dem_pts)):
            d_min = np.min([haversine_km(charge_pts[i, 1], charge_pts[i, 0], dem_pts[j, 1], dem_pts[j, 0]) for i in range(len(charge_pts))])
            if d_min <= radius_charge_km:
                covered_weight += weights[j]
            sum_dist += weights[j] * d_min
        metrics["coverage_ratio"] = covered_weight / total_weight if total_weight > 0 else 0.0
        metrics["avg_distance_to_charge_km"] = sum_dist / total_weight if total_weight > 0 else None
        if trunk_graph is not None:
            metrics["trunk_connected"] = nx.is_connected(trunk_graph) if trunk_graph.number_of_nodes() > 0 else False
        return metrics


def run_full_pipeline(
    data_service,
    graph_service,
    city_name: str,
    network_type: str = "drive",
    simplify: bool = True,
    *,
    demand_cell_m: float = 250.0,
    max_charge_stations: Optional[int] = None,
    num_garages: int = 3,
    num_to: int = 2,
) -> Dict[str, Any]:
    """
    Запускает полный пайплайн: данные города → спрос → зарядки → гаражи/ТО кандидаты → trunk → гаражи/ТО → метрики.
    """
    logger = logging.getLogger(__name__)
    data = data_service.get_city_data(city_name, network_type=network_type, simplify=simplify, load_no_fly_zones=True)
    buildings = data.get("buildings")
    road_graph = data.get("road_graph")
    city_boundary = data.get("city_boundary")
    no_fly_zones = data.get("no_fly_zones")

    placement = StationPlacement(data_service, graph_service, logger=logger)

    # Шаг 1: спрос
    demand = placement.build_demand_points(buildings, road_graph, city_boundary, cell_size_m=demand_cell_m)
    if demand is None or len(demand) == 0:
        return {"error": "Нет точек спроса", "demand": None, "charge_stations": None, "garages": None, "to_stations": None, "trunk_graph": None, "metrics": {}}

    # Кандидаты: крыши + земля для зарядок; отдельно гаражи и ТО (промзоны)
    candidates_rooftop = data_service.get_station_candidates(
        buildings, city_boundary, no_fly_zones, road_graph, station_type="rooftop"
    )
    candidates_ground = data_service.get_station_candidates(
        buildings, city_boundary, no_fly_zones, road_graph, station_type="ground"
    )
    parts = []
    if candidates_rooftop is not None and len(candidates_rooftop) > 0:
        parts.append(candidates_rooftop)
    if candidates_ground is not None and len(candidates_ground) > 0:
        parts.append(candidates_ground)
    charge_candidates = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=(parts[0].crs if parts else "EPSG:4326")) if parts else gpd.GeoDataFrame()

    # Шаг 2: зарядки
    charge_stations, charge_metrics = placement.place_charging_stations(
        charge_candidates, demand, no_fly_zones, radius_km=R_CHARGE_KM, max_stations=max_charge_stations
    )

    # Кандидаты гаражи и ТО (промзоны)
    garage_candidates = data_service.get_station_candidates(
        buildings, city_boundary, no_fly_zones, road_graph, station_type="garage"
    )
    to_candidates = data_service.get_station_candidates(
        buildings, city_boundary, no_fly_zones, road_graph, station_type="to"
    )
    if garage_candidates is None or len(garage_candidates) == 0:
        garage_candidates = gpd.GeoDataFrame()
    if to_candidates is None or len(to_candidates) == 0:
        to_candidates = gpd.GeoDataFrame()

    # Заглушки гаражи/ТО для trunk (если ещё не размещены — используем первых кандидатов)
    garage_placeholder = garage_candidates.iloc[: max(1, num_garages)].copy() if garage_candidates is not None and len(garage_candidates) > 0 else gpd.GeoDataFrame()
    to_placeholder = to_candidates.iloc[: max(1, num_to)].copy() if to_candidates is not None and len(to_candidates) > 0 else gpd.GeoDataFrame()

    # Шаг 3: trunk (с заглушками)
    trunk = placement.build_trunk_graph(charge_stations, garage_placeholder, to_placeholder, no_fly_zones, max_edge_km=R_GARAGE_TO_KM)

    # Шаг 4–5: размещаем гаражи и ТО по правилам
    garages = placement.place_garages(demand, garage_candidates, k=num_garages, radius_km=R_GARAGE_TO_KM)
    to_stations = placement.place_to_stations(trunk, to_candidates, k=num_to)

    # Пересобираем trunk с уже выбранными гаражами и ТО
    trunk = placement.build_trunk_graph(charge_stations, garages, to_stations, no_fly_zones, max_edge_km=R_GARAGE_TO_KM)

    # Шаг 6: метрики
    metrics = placement.compute_metrics(demand, charge_stations, trunk, radius_charge_km=R_CHARGE_KM)
    metrics["charge"] = charge_metrics

    return {
        "demand": demand,
        "charge_stations": charge_stations,
        "garages": garages,
        "to_stations": to_stations,
        "trunk_graph": trunk,
        "metrics": metrics,
        "city_boundary": city_boundary,
        "no_fly_zones": no_fly_zones,
        "params": {
            "R_charge_km": R_CHARGE_KM,
            "R_garage_to_km": R_GARAGE_TO_KM,
            "D_max_km": D_MAX_KM,
        },
    }
