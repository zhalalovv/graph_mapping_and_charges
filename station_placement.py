# -*- coding: utf-8 -*-
"""
Размещение станций зарядки, гаражей и ТО по правилам:
- Зарядка+ожидание (туда-обратно): R_charge = 0.4 * D_max = 6.4 км (FlyCart 30, D_max=16 км).
- Гаражи/ТО (в одну сторону): R_garage/TO = 0.8 * D_max = 12.8 км.

Шаги:
  1) Точки спроса (кластеры/сетка 250×250 м с весом).
  2) Зарядки: weighted maximum coverage / set cover (жадный).
  3) Магистральный слой: узлы Charge_A + гаражи/ТО, рёбра А–А (макс. 4 на станцию).
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
from shapely.ops import unary_union
from scipy.spatial import cKDTree

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
    def __init__(self, data_service, logger=None):
        self.data_service = data_service
        self.logger = logger or logging.getLogger(__name__)

    # --- Шаг 1: точки спроса ---
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
        # При method="dbscan": дополнительно заполнять полигоны кластеров сеткой точек спроса,
        # чтобы размещение станций стремилось покрыть всю область кластера.
        fill_clusters: bool = False,
        # Пользовательский шаг сетки для заполнения кластеров (в метрах); по умолчанию = dbscan_eps_m.
        cluster_fill_step_m: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        """Точки спроса: DBSCAN-кластеры или сеточные ячейки с весом. При use_all_buildings=True — по всем зданиям вне беспилотных зон."""
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

    # --- Кандидаты зарядок относительно кластеров DBSCAN ---
    def build_charge_candidates_from_clusters(
        self,
        demand: gpd.GeoDataFrame,
        full_candidates: gpd.GeoDataFrame,
        *,
        radius_km: float = R_CHARGE_KM,
        max_distance_km: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        """
        Строит кандидатов зарядки относительно кластеров DBSCAN: для каждого кластера (центроид спроса)
        выбирается ближайшая точка из full_candidates (крыши/площадки) в пределах max_distance_km.
        Возвращает уникальный набор кандидатов — по одному месту на кластер (лучшая позиция для кластера).
        """
        if demand is None or len(demand) == 0 or full_candidates is None or len(full_candidates) == 0:
            return full_candidates if full_candidates is not None else gpd.GeoDataFrame()

        # На большие кластеры можно ставить несколько кандидатов:
        # считаем квантиль по весу и даём больше ближайших точек крупным кластерам.
        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
        # Адаптивные пороги: нижняя/верхняя медиана по весу
        w_med = float(np.median(weights)) if len(weights) > 0 else 1.0
        w_p75 = float(np.percentile(weights, 75)) if len(weights) > 0 else w_med

        max_dist = max_distance_km if max_distance_km is not None else radius_km * 2.0
        dem_pts = _points_array(demand)
        cand_pts = _points_array(full_candidates)
        chosen_indices = set()
        for i in range(len(dem_pts)):
            lat_d, lon_d = dem_pts[i, 1], dem_pts[i, 0]
            # Сколько кандидатов хотим для этого кластера в зависимости от веса
            w = float(weights[i]) if i < len(weights) else 1.0
            if w >= w_p75:
                n_for_cluster = 3
            elif w >= w_med:
                n_for_cluster = 2
            else:
                n_for_cluster = 1

            # Список (j, dist) кандидатов в пределах max_dist
            dists = []
            for j in range(len(cand_pts)):
                d_km = haversine_km(lat_d, lon_d, cand_pts[j, 1], cand_pts[j, 0])
                if d_km <= max_dist:
                    dists.append((j, d_km))
            if not dists:
                continue
            # Берём ближайшие n_for_cluster ещё не выбранных кандидатов
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
            self.logger.warning("Нет кандидатов в радиусе от кластеров — используем все кандидаты")
            return full_candidates
        out = full_candidates.iloc[sorted(chosen_indices)].copy()
        self.logger.info(f"Кандидаты зарядки относительно кластеров: {len(out)} из {len(full_candidates)} (по кластерам DBSCAN)")
        return out

    # --- Шаг 2: зарядки — тип А (основное покрытие) + дополнительные (до 100%, поддержка развозки) ---
    def place_charging_stations(
        self,
        candidates: gpd.GeoDataFrame,
        demand: gpd.GeoDataFrame,
        no_fly_zones,
        *,
        radius_km: float = R_CHARGE_KM,
        max_stations: Optional[int] = None,
        # Для типа А целимся покрыть почти весь «тяжёлый» спрос ядра
        type_a_coverage_ratio: float = 0.95,
        # По умолчанию расставляем только станции типа А; тип Б можно включить отдельно.
        enable_type_b: bool = False,
        type_b_coverage_ratio: float = 1.0,
        # Минимальное расстояние между центрами станций (в радиусах).
        # Для типа А по умолчанию допускаем частичное перекрытие (1.0 * R),
        # чтобы на большой зоне спроса можно было поставить несколько станций.
        min_center_distance_factor_a: float = 1.0,
        min_center_distance_factor_b: float = 2.0,
        # В ядре города (в радиусе core_dense_radius_factor * radius_km от центра спроса)
        # разрешаем ещё плотнее: min_center_distance_factor_core_a * radius_km.
        min_center_distance_factor_core_a: Optional[float] = None,
        core_dense_radius_factor: float = 5.0,
        # Станции типа А ограничиваем «ядром» города: не дальше core_radius_factor_a * radius_km
        core_radius_factor_a: float = 4.0,
        # Только "тяжёлый" спрос (квантили по весу) считается ядром для станций типа А.
        core_weight_quantile_a: float = 0.7,
        # Минимальный прирост покрытия ядра (доля от total_core_weight),
        # при котором ещё есть смысл добавлять новую станцию типа А.
        min_core_gain_ratio_a: float = 0.01,
        # Стохастичность: random_state=None -> каждый запуск чуть разный;
        # при фиксированном random_state результат воспроизводим.
        random_state: Optional[int] = None,
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
                "demand_covered": 0, "demand_total": 0, "coverage_ratio": 0.0,
            }

        cand_pts = _points_array(candidates)
        dem_pts = _points_array(demand)
        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
        n_demand = len(dem_pts)
        total_weight = float(weights.sum())
        covered = np.zeros(n_demand, dtype=bool)
        selected = []
        selected_indices: List[int] = []  # индексы кандидатов (и тип А, и доп)
        selected_types: List[str] = []    # типы уже выбранных станций

        # ГПСЧ для стохастического выбора среди почти лучших кандидатов
        rng = np.random.default_rng(random_state)

        # Взвешенный центр спроса — используется как «ядро» города для станций типа А.
        core_lon = float(np.average(dem_pts[:, 0], weights=weights)) if n_demand > 0 else 0.0
        core_lat = float(np.average(dem_pts[:, 1], weights=weights)) if n_demand > 0 else 0.0
        max_core_dist_a_km = max(0.0, float(core_radius_factor_a)) * radius_km

        # Ядро спроса по весу: по умолчанию — самые «тяжёлые» точки.
        # При core_weight_quantile_a <= 0 берём весь спрос как ядро (тип А по всему городу).
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
            """
            Проверка геометрического ограничения: станции не должны перекрываться.
            Центры окружностей должны быть не ближе, чем min_center_distance_factor * radius_km
            друг к другу. В ядре города (около центра спроса) разрешено плотнее.
            """
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
                # Для стохастики храним несколько почти лучших кандидатов
                top_indices: List[int] = []
                for i in range(len(cand_pts)):
                    if i in selected_indices:
                        continue
                    # Станции типа А — только крыши МКД (source == "building"); тип Б — любые кандидаты
                    if station_type == "charge_a" and "source" in candidates.columns:
                        if candidates.iloc[i].get("source") != "building":
                            continue
                    lon_c, lat_c = cand_pts[i, 0], cand_pts[i, 1]
                    # Для станций типа А жёстко ограничиваемся ядром: далеко в сёла не уходим.
                    if station_type == "charge_a" and max_core_dist_a_km > 0.0:
                        d_core = haversine_km(lat_c, lon_c, core_lat, core_lon)
                        if d_core > max_core_dist_a_km:
                            continue
                    # Геометрическое ограничение: станции не должны "впадать" друг в друга.
                    # Разрешаем только такие кандидаты, которые достаточно далеко от уже выбранных.
                    if _too_close_to_existing(i, station_type):
                        continue
                    new_weight = 0.0
                    for j in range(n_demand):
                        # Фаза А: учитываем только «ядро» спроса (is_core_demand),
                        # фаза Б: весь оставшийся спрос.
                        if station_type == "charge_a" and not is_core_demand[j]:
                            continue
                        if not ignore_covered_in_score and covered[j]:
                            continue
                        d_km = haversine_km(lat_c, lon_c, dem_pts[j, 1], dem_pts[j, 0])
                        if d_km <= radius_km:
                            new_weight += weights[j]
                    # Для станций типа Б даём приоритет крышам МКД:
                    # кандидаты с source == "building" получают небольшой бонус к "выигрышу" по спросу.
                    if (
                        station_type == "charge_b"
                        and "source" in candidates.columns
                        and candidates.iloc[i].get("source") == "building"
                        and new_weight > 0
                    ):
                        new_weight *= 1.2
                    if new_weight <= 0:
                        continue
                    if new_weight > best_new_weight:
                        best_new_weight = new_weight
                        best_idx = i
                        top_indices = [i]
                    else:
                        # Кандидаты, которые дают почти такой же прирост (например, не хуже 95% от лучшего),
                        # тоже добавляем в список для случайного выбора.
                        if best_new_weight > 0 and new_weight >= 0.95 * best_new_weight:
                            top_indices.append(i)
                if best_new_weight <= 0:
                    break
                # Если есть несколько почти эквивалентных по весу кандидатов — выбираем случайно один из них.
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
                        # В фазе А помечаем покрытие только для точек ядра спроса.
                        if station_type == "charge_a" and not is_core_demand[j]:
                            continue
                        if not covered[j]:
                            covered[j] = True
                selected.append({
                    "geometry": Point(cand_pts[best_idx, 0], cand_pts[best_idx, 1]),
                    "station_type": station_type,
                    "source_index": best_idx,
                })
                if max_stations is not None and len(selected) >= max_stations:
                    return
                ratio = (weights[covered].sum() / total_weight) if total_weight > 0 else 0.0
                if ratio >= stop_at_ratio:
                    return

        # Фаза 1: зарядки типа А — покрывают основную часть спроса.
        # В оценке кандидата берём весь спрос в радиусе (ignore_covered_in_score=True),
        # чтобы станции А “обнима́ли” основную зону спроса, а не разлетались по мелким островкам.
        run_phase("charge_a", type_a_coverage_ratio, ignore_covered_in_score=True)

        # Фаза 2: станции типа Б (дополняют покрытие оставшегося спроса).
        # Для них действует то же ограничение на неперекрытие окружностей.
        if enable_type_b and (max_stations is None or len(selected) < max_stations):
            run_phase("charge_b", type_b_coverage_ratio, ignore_covered_in_score=False)

        gdf = gpd.GeoDataFrame(selected, crs=candidates.crs) if selected else gpd.GeoDataFrame()
        covered_weight = float(weights[covered].sum())
        n_a = sum(1 for r in selected if r["station_type"] == "charge_a")
        n_b = sum(1 for r in selected if r["station_type"] == "charge_b")
        metrics = {
            "placed": len(selected),
            "placed_type_a": n_a,
            "placed_type_b": n_b,
            # Совместимость со старым ключом: считаем тип Б "дополнительными" станциями
            "placed_extra": n_b,
            "demand_covered": covered_weight,
            "demand_total": total_weight,
            "coverage_ratio": covered_weight / total_weight if total_weight > 0 else 0.0,
        }
        self.logger.info(
            f"Зарядки: тип А {n_a}, тип Б {n_b}, покрытие {metrics['coverage_ratio']:.2%}"
        )
        return gdf, metrics

    # --- Шаг 3: магистральный граф (только А–А, макс. 4 связи на станцию, без Delaunay) ---
    def build_trunk_graph(
        self,
        charge_stations: gpd.GeoDataFrame,
        garage_points: gpd.GeoDataFrame,
        to_points: gpd.GeoDataFrame,
        obstacles,
        air_graph: Optional[nx.Graph] = None,
        *,
        max_edge_km: float = R_GARAGE_TO_KM,
        max_neighbors_a: int = 4,
    ) -> nx.Graph:
        """
        Магистраль: узлы — все станции типа А:
        - зарядки типа А (основное покрытие),
        - гаражи,
        - станции ТО.

        Рёбра магистрали соединяют каждую станцию с ближайшими соседями в пределах max_edge_km (км),
        не более 2 рёбер на станцию (даже если max_neighbors_a > 2).

        Если прямая связь между станциями пересекает препятствия (бесполётные зоны / высокие здания),
        и есть предварительно построенный
        воздушный граф air_graph, то вместо прямого отрезка строится обход по воздушному графу
        (A* по весу рёбер). Полученная полилиния сохраняется в атрибуте geometry_coords ребра.
        """
        nodes_list = []
        # Зарядки типа А, гаражи и ТО — все считаются станциями типа А для магистрали
        trunk_nodes = []
        for i, row in (charge_stations.iterrows() if charge_stations is not None and len(charge_stations) > 0 else []):
            if row.get("station_type") != "charge_a":
                continue
            lon, lat = _coords_from_geom(row.geometry)
            nid = f"charge_{i}"
            nodes_list.append(((lon, lat), "charge", nid))
            trunk_nodes.append((nid, lon, lat))
        for i, row in (garage_points.iterrows() if garage_points is not None and len(garage_points) > 0 else []):
            lon, lat = _coords_from_geom(row.geometry)
            nid = f"garage_{i}"
            nodes_list.append(((lon, lat), "garage", nid))
            trunk_nodes.append((nid, lon, lat))
        for i, row in (to_points.iterrows() if to_points is not None and len(to_points) > 0 else []):
            lon, lat = _coords_from_geom(row.geometry)
            nid = f"to_{i}"
            nodes_list.append(((lon, lat), "to", nid))
            trunk_nodes.append((nid, lon, lat))

        G = nx.Graph()
        for (lon, lat), typ, nid in nodes_list:
            G.add_node(nid, lon=lon, lat=lat, node_type=typ)

        if len(trunk_nodes) < 2:
            return G

        # Геометрия беспилотных зон (WGS84) для проверки: проходит ли магистральный отрезок через зону
        nfz_union = None
        if obstacles is not None and getattr(obstacles, "is_empty", True) is False and getattr(obstacles, "is_valid", True):
            nfz_union = obstacles

        # Подготовка структур для A* по воздушному графу
        air_coords = None
        air_ids = None
        air_tree = None
        if air_graph is not None and air_graph.number_of_nodes() > 1:
            air_ids = list(air_graph.nodes())
            pts = []
            for nid in air_ids:
                data = air_graph.nodes[nid]
                lon = data.get("lon") or data.get("x")
                lat = data.get("lat") or data.get("y")
                if lon is None or lat is None:
                    pts.append((np.nan, np.nan))
                else:
                    pts.append((float(lon), float(lat)))
            air_coords = np.array(pts, dtype=float)
            # Фильтруем узлы без координат
            valid_mask = ~np.isnan(air_coords[:, 0]) & ~np.isnan(air_coords[:, 1])
            if not np.any(valid_mask):
                air_coords = None
                air_ids = None
            else:
                air_coords_valid = air_coords[valid_mask]
                air_ids_valid = [air_ids[i] for i, ok in enumerate(valid_mask) if ok]
                air_coords = air_coords_valid
                air_ids = air_ids_valid
                try:
                    air_tree = cKDTree(air_coords)
                except Exception:
                    air_tree = None

        def _edge_intersects_nfz(lon1: float, lat1: float, lon2: float, lat2: float) -> bool:
            """Проверка: проходит ли отрезок магистрали через беспилотную зону (пересекает или внутри)."""
            if nfz_union is None:
                return False
            from shapely.geometry import LineString as _Line

            try:
                line = _Line([(float(lon1), float(lat1)), (float(lon2), float(lat2))])
                if not line.is_valid:
                    return False
                # пересекает границу или лежит внутри зоны
                return nfz_union.intersects(line)
            except Exception:
                return False

        def _astar_detour(lon1: float, lat1: float, lon2: float, lat2: float):
            """
            Обход по воздушному графу между двумя точками.
            Возвращает (coords: List[(lon, lat)], length_km) или (None, None) при неудаче.
            """
            if air_graph is None or air_tree is None or air_coords is None or not air_ids:
                return None, None
            try:
                # Находим ближайшие узлы воздушного графа к концам магистрали
                d_start, idx_start = air_tree.query([lon1, lat1], k=1)
                d_end, idx_end = air_tree.query([lon2, lat2], k=1)
                start_id = air_ids[int(idx_start)]
                end_id = air_ids[int(idx_end)]

                def _heuristic(a, b):
                    da = air_graph.nodes[a]
                    db = air_graph.nodes[b]
                    lon_a = da.get("lon") or da.get("x")
                    lat_a = da.get("lat") or da.get("y")
                    lon_b = db.get("lon") or db.get("x")
                    lat_b = db.get("lat") or db.get("y")
                    if lon_a is None or lat_a is None or lon_b is None or lat_b is None:
                        return 0.0
                    return haversine_km(lat_a, lon_a, lat_b, lon_b)

                path = nx.astar_path(air_graph, start_id, end_id, heuristic=_heuristic, weight="weight")
                if not path or len(path) < 2:
                    return None, None
                coords = []
                total_len = 0.0
                prev = None
                for nid in path:
                    d = air_graph.nodes[nid]
                    lon = d.get("lon") or d.get("x")
                    lat = d.get("lat") or d.get("y")
                    if lon is None or lat is None:
                        continue
                    lon_f, lat_f = float(lon), float(lat)
                    coords.append((lon_f, lat_f))
                    if prev is not None:
                        total_len += haversine_km(prev[1], prev[0], lat_f, lon_f)
                    prev = (lon_f, lat_f)
                if len(coords) < 2:
                    return None, None
                return coords, total_len
            except Exception:
                return None, None

        # Рёбра между всеми станциями типа А: каждая станция соединяется с макс. max_neighbors_a
        # ближайшими соседями в пределах max_edge_km.
        pts = np.array([[lon, lat] for _, lon, lat in trunk_nodes])
        ids = [nid for nid, _, _ in trunk_nodes]
        # Не более 2 соседей на станцию, даже если max_neighbors_a больше.
        k = min(max(1, int(max_neighbors_a)), 2, len(trunk_nodes) - 1)
        if k <= 0:
            return G
        for i in range(len(trunk_nodes)):
            nid_i = ids[i]
            lat_i, lon_i = pts[i, 1], pts[i, 0]
            candidates = []
            for j in range(len(trunk_nodes)):
                if j == i:
                    continue
                d_km = haversine_km(lat_i, lon_i, pts[j, 1], pts[j, 0])
                if d_km <= max_edge_km:
                    candidates.append((d_km, j))
            candidates.sort(key=lambda x: x[0])
            for d_km, j in candidates[:k]:
                nid_j = ids[j]
                if not G.has_edge(nid_i, nid_j):
                    lon_j, lat_j = pts[j, 0], pts[j, 1]
                    # Проверяем: проходит ли прямая между станциями через беспилотную зону
                    if not _edge_intersects_nfz(lon_i, lat_i, lon_j, lat_j):
                        G.add_edge(nid_i, nid_j, weight=d_km, length=d_km)
                        continue
                    # Отрезок пересекает беспилотную зону — строим обход по воздушному графу (A*)
                    self.logger.debug("Магистраль пересекает беспилотную зону %s–%s, строим обход A*", nid_i, nid_j)
                    if air_graph is not None and air_tree is not None:
                        detour_coords, detour_len = _astar_detour(lon_i, lat_i, lon_j, lat_j)
                        if detour_coords is not None and detour_len is not None:
                            G.add_edge(
                                nid_i,
                                nid_j,
                                weight=detour_len,
                                length=detour_len,
                                geometry_coords=detour_coords,
                            )
                            self.logger.info("Обход беспилотной зоны: %s–%s, точек маршрута %s", nid_i, nid_j, len(detour_coords))
                            continue
                    self.logger.warning("Обход не построен для %s–%s, оставлена прямая", nid_i, nid_j)
                    G.add_edge(nid_i, nid_j, weight=d_km, length=d_km)
        # Связность: объединяем компоненты всех узлов магистрали
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
                while remaining:
                    best_pair, best_dist, best_idx = None, float("inf"), -1
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
                                if d_km < best_dist:
                                    best_dist, best_pair, best_idx = d_km, (a, b), idx
                    if best_pair is None:
                        break
                    u, v = best_pair
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=best_dist, length=best_dist)
                    base_comp |= remaining[best_idx]
                    del remaining[best_idx]
        self.logger.info(f"Trunk (А–А, макс. {max_neighbors_a} соседей): {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
        return G

    def build_branch_edges(
        self,
        charge_stations: gpd.GeoDataFrame,
        garage_points: Optional[gpd.GeoDataFrame] = None,
        to_points: Optional[gpd.GeoDataFrame] = None,
        *,
        k_nearest: int = 3,
        max_branch_km: float = R_GARAGE_TO_KM,
    ) -> List[Dict[str, Any]]:
        """
        Ветки от станций типа Б к ближайшим узлам: зарядки А, гаражи, ТО (любые). Макс. k_nearest связей на станцию Б.
        Returns: список GeoJSON-like Feature (LineString, edge_type="branch", source_id, target_id, weight_km).
        """
        if charge_stations is None or len(charge_stations) == 0:
            return []
        type_a = charge_stations["station_type"] == "charge_a" if "station_type" in charge_stations.columns else pd.Series([True] * len(charge_stations))
        type_b = charge_stations["station_type"] == "charge_b" if "station_type" in charge_stations.columns else pd.Series([False] * len(charge_stations))
        a_indices = [i for i in charge_stations.index[type_a]]
        b_indices = [i for i in charge_stations.index[type_b]]
        if not b_indices:
            return []
        # Цели: тип А + гаражи + ТО (id: charge_i, garage_i, to_i)
        targets_pts: List[Tuple[float, float, str]] = []
        for i in a_indices:
            lon, lat = _coords_from_geom(charge_stations.loc[i].geometry)
            targets_pts.append((lon, lat, f"charge_{i}"))
        if garage_points is not None:
            for i, row in garage_points.iterrows():
                lon, lat = _coords_from_geom(row.geometry)
                targets_pts.append((lon, lat, f"garage_{i}"))
        if to_points is not None:
            for i, row in to_points.iterrows():
                lon, lat = _coords_from_geom(row.geometry)
                targets_pts.append((lon, lat, f"to_{i}"))
        if not targets_pts:
            return []
        b_pts = np.array([_coords_from_geom(charge_stations.loc[i].geometry) for i in b_indices])
        features = []
        k = min(k_nearest, len(targets_pts))
        for bi, idx_b in enumerate(b_indices):
            lon_b, lat_b = b_pts[bi, 0], b_pts[bi, 1]
            candidates = []
            for lon_t, lat_t, tid in targets_pts:
                d_km = haversine_km(lat_b, lon_b, lat_t, lon_t)
                if d_km <= max_branch_km:
                    candidates.append((d_km, lon_t, lat_t, tid))
            candidates.sort(key=lambda x: x[0])
            for d_km, lon_t, lat_t, tid in candidates[:k]:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[lon_b, lat_b], [lon_t, lat_t]]},
                    "properties": {
                        "edge_type": "branch",
                        "source_id": str(idx_b),
                        "target_id": tid,
                        "weight_km": round(d_km, 4),
                    },
                })
        self.logger.info(f"Ветки Б→А/гараж/ТО (макс. {k} на станцию): {len(features)} рёбер")
        return features

    def build_local_edges(
        self,
        charge_stations: gpd.GeoDataFrame,
        *,
        max_edge_km: float = R_GARAGE_TO_KM,
        max_neighbors_b: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Локальные связи между станциями типа Б. Каждая Б соединяется с макс. max_neighbors_b ближайшими Б в пределах max_edge_km.
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
        features = []
        k = min(max_neighbors_b, len(b_indices) - 1)
        seen_edges = set()
        for i in range(len(b_indices)):
            candidates = []
            for j in range(len(b_indices)):
                if j == i:
                    continue
                d_km = haversine_km(b_pts[i, 1], b_pts[i, 0], b_pts[j, 1], b_pts[j, 0])
                if d_km <= max_edge_km:
                    candidates.append((d_km, j))
            candidates.sort(key=lambda x: x[0])
            for d_km, j in candidates[:k]:
                ui, vi = b_indices[i], b_indices[j]
                edge_key = (min(ui, vi), max(ui, vi))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[b_pts[i, 0], b_pts[i, 1]], [b_pts[j, 0], b_pts[j, 1]]],
                    },
                    "properties": {
                        "edge_type": "local",
                        "source_id": str(ui),
                        "target_id": str(vi),
                        "weight_km": round(d_km, 4),
                    },
                })
        self.logger.info(f"Локальные рёбра Б↔Б (макс. {max_neighbors_b} на станцию): {len(features)}")
        return features

    # --- Шаг 4: гаражи (k-median или жадно по покрытию спроса) ---
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
            return garage_candidates.iloc[:k].copy()

        dem_pts = _points_array(demand)
        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
        cand_pts = _points_array(garage_candidates)
        n_cand = len(cand_pts)
        total_weight = float(weights.sum()) if len(weights) > 0 else 0.0

        # Режим покрытия: жадно добавляем гаражи, пока не покроем долю спроса (как зарядки типа А)
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
                            haversine_km(cand_pts[c, 1], cand_pts[c, 0], cand_pts[prev, 1], cand_pts[prev, 0]) < min_dist_km
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

        # Режим k-median: фиксированное число гаражей
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
        # Пост-фильтр: убираем "лишние" гаражи, которые почти не обслуживают спрос.
        # Считаем, какой вес спроса реально обслуживает каждый выбранный гараж
        # (точки спроса, для которых он является ближайшим и в радиусе radius_km).
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
                # Минимальная доля спроса для "осмысленного" гаража (от общего веса)
                min_share_total = 0.05  # 5% от всего спроса
                for pos in range(len(chosen)):
                    share_total = (served_weight[pos] / total_weight) if total_weight > 0 else 0.0
                    share_rel = (served_weight[pos] / max_served) if max_served > 0 else 0.0
                    # Оставляем гараж, если он покрывает заметную часть спроса сам по себе
                    # или хотя бы не намного хуже лучшего (чтобы не удалить все).
                    if share_total >= min_share_total or share_rel >= 0.3:
                        keep_positions.append(pos)
                if not keep_positions:
                    # Если порог слишком строгий и никого не оставили — оставляем лучший по покрытию.
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

    # --- Шаг 5: станции ТО (покрытие спроса или betweenness по trunk) ---
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
        """Размещение ТО: либо до k станций, либо жадно по покрытию спроса до coverage_ratio_target (как зарядки/гаражи)."""
        if to_candidates is None or len(to_candidates) == 0:
            return gpd.GeoDataFrame()
        # Trunk может быть пустым при режиме покрытия
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

            # Режим покрытия: жадно до целевой доли спроса (как гаражи)
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
                                    cand_pts[i, 1], cand_pts[i, 0],
                                    cand_pts[j, 1], cand_pts[j, 0],
                                ) < min_dist_km
                                for j in chosen_idx
                            ):
                                continue
                        lat_i, lon_i = cand_pts[i, 1], cand_pts[i, 0]
                        new_weight = 0.0
                        for d_idx in range(n_demand):
                            if not covered[d_idx] and haversine_km(lat_i, lon_i, dem_pts[d_idx, 1], dem_pts[d_idx, 0]) <= radius_km:
                                new_weight += weights[d_idx]
                        if new_weight > best_new_weight:
                            best_new_weight = new_weight
                            best_i = i
                    if best_i < 0 or best_new_weight <= 0:
                        break
                    chosen_idx.append(best_i)
                    lat_i, lon_i = cand_pts[best_i, 1], cand_pts[best_i, 0]
                    for d_idx in range(n_demand):
                        if not covered[d_idx] and haversine_km(lat_i, lon_i, dem_pts[d_idx, 1], dem_pts[d_idx, 0]) <= radius_km:
                            covered[d_idx] = True
                rows = [to_candidates.iloc[i].copy() for i in chosen_idx]
                for r in rows:
                    r["station_type"] = "to"
                out = gpd.GeoDataFrame(rows, crs=to_candidates.crs)
                self.logger.info(
                    f"ТО (покрытие): размещено {len(out)}; покрыто спроса {float(weights[covered].sum()):.1f} из {total_weight:.1f}"
                )
                return out

            # Обычный режим: до k станций по покрытию
            chosen_idx = []
            for _ in range(min(k, len(cand_pts))):
                best_i = -1
                best_new_weight = 0.0
                for i in range(len(cand_pts)):
                    if i in chosen_idx:
                        continue
                    if chosen_idx and any(
                        haversine_km(
                            cand_pts[i, 1], cand_pts[i, 0],
                            cand_pts[j, 1], cand_pts[j, 0],
                        ) <= radius_km
                        for j in chosen_idx
                    ):
                        continue
                    lat_i, lon_i = cand_pts[i, 1], cand_pts[i, 0]
                    new_weight = 0.0
                    for d_idx in range(n_demand):
                        if not covered[d_idx] and haversine_km(lat_i, lon_i, dem_pts[d_idx, 1], dem_pts[d_idx, 0]) <= radius_km:
                            new_weight += weights[d_idx]
                    if new_weight > best_new_weight:
                        best_new_weight = new_weight
                        best_i = i
                if best_i < 0 or best_new_weight <= 0:
                    break
                chosen_idx.append(best_i)
                lat_i, lon_i = cand_pts[best_i, 1], cand_pts[best_i, 0]
                for d_idx in range(n_demand):
                    if not covered[d_idx] and haversine_km(lat_i, lon_i, dem_pts[d_idx, 1], dem_pts[d_idx, 0]) <= radius_km:
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

        # Фоллбэк: betweenness по trunk, если спроса нет или граф не пустой
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
    city_name: str,
    network_type: str = "drive",
    simplify: bool = True,
    *,
    demand_method: str = "dbscan",
    demand_cell_m: float = 250.0,
    dbscan_eps_m: float = 180.0,
    dbscan_min_samples: int = 15,
    use_all_buildings: bool = False,
    max_charge_stations: Optional[int] = None,
    num_garages: int = 1,
    num_to: int = 1,
) -> Dict[str, Any]:
    """
    Запускает полный пайплайн: данные города → спрос (DBSCAN или сетка) → зарядки → гаражи/ТО → метрики.
    """
    logger = logging.getLogger(__name__)
    from graph_service import GraphService

    data = data_service.get_city_data(city_name, network_type=network_type, simplify=simplify, load_no_fly_zones=True)
    buildings = data.get("buildings")
    road_graph = data.get("road_graph")
    city_boundary = data.get("city_boundary")
    no_fly_zones = data.get("no_fly_zones")

    placement = StationPlacement(data_service, logger=logger)

    # Шаг 1: спрос (DBSCAN по умолчанию)
    demand = placement.build_demand_points(
        buildings, road_graph, city_boundary,
        method=demand_method,
        cell_size_m=demand_cell_m,
        dbscan_eps_m=dbscan_eps_m,
        dbscan_min_samples=dbscan_min_samples,
        use_all_buildings=use_all_buildings,
        no_fly_zones=no_fly_zones,
        # При method="dbscan" дополнительно заполняем полигоны кластеров сеткой точек спроса,
        # чтобы размещение станций стремилось закрывать весь кластер.
        fill_clusters=(demand_method == "dbscan"),
        cluster_fill_step_m=None,
    )
    if demand is None or len(demand) == 0:
        return {"error": "Нет точек спроса", "demand": None, "charge_stations": None, "garages": None, "to_stations": None, "trunk_graph": None, "metrics": {}}

    # Кандидаты зарядок: крыши МКД (для типа А) + наземные (для типа Б — полное покрытие, в т.ч. отдалённые зоны)
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
    charge_candidates_full = (
        gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=(parts[0].crs if parts else "EPSG:4326"))
        if parts
        else gpd.GeoDataFrame()
    )

    # При спросе по DBSCAN: кандидаты зарядки строим относительно кластеров (ближайшая крыша/площадка к каждому кластеру)
    if demand_method == "dbscan" and charge_candidates_full is not None and len(charge_candidates_full) > 0:
        charge_candidates = placement.build_charge_candidates_from_clusters(
            demand, charge_candidates_full, radius_km=R_CHARGE_KM, max_distance_km=R_CHARGE_KM * 2.0
        )
    else:
        charge_candidates = charge_candidates_full

    # Для вспомогательных станций (тип Б) нужен расширенный набор кандидатов, чтобы покрыть незанятые зоны спроса
    candidates_for_placement = (
        charge_candidates_full
        if (charge_candidates_full is not None and len(charge_candidates_full) > 0)
        else charge_candidates
    )
    if candidates_for_placement is None or len(candidates_for_placement) == 0:
        candidates_for_placement = charge_candidates

    # Шаг 2: зарядки тип А (R=1.6 км), в радиусе 3.2 км от А — только тип Б; тип Б добивают остальное.
    charge_stations, charge_metrics = placement.place_charging_stations(
        candidates_for_placement,
        demand,
        no_fly_zones,
        radius_km=R_CHARGE_KM,
        max_stations=max_charge_stations,
        enable_type_b=True,
        type_a_coverage_ratio=0.95,
        type_b_coverage_ratio=1.0,
        min_center_distance_factor_a=2.0,    # А не ближе 2*1.6=3.2 км — в 3.2 км только Б
        min_center_distance_factor_core_a=2.0,
        core_dense_radius_factor=5.0,
        core_radius_factor_a=0.0,
        core_weight_quantile_a=0.0,
        min_center_distance_factor_b=0.75,   # Б плотнее — дозаполняют слабо заполненные кластеры
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

    # Жёстко убираем из кандидатов все точки внутри буфера подстанций (гаражи и ТО не должны там стоять)
    power_buffer_union = data_service.get_power_substation_buffer_union(buildings)
    if power_buffer_union is not None:
        def _drop_inside_power_buffer(cand_gdf):
            if cand_gdf is None or len(cand_gdf) == 0:
                return cand_gdf
            keep = []
            for idx, row in cand_gdf.iterrows():
                pt = getattr(row.geometry, "centroid", row.geometry)
                if pt is None:
                    keep.append(True)
                    continue
                try:
                    keep.append(not (power_buffer_union.contains(pt) or power_buffer_union.intersects(pt)))
                except Exception:
                    keep.append(True)
            if not any(keep):
                return gpd.GeoDataFrame(crs=cand_gdf.crs)
            return cand_gdf.iloc[[i for i, k in enumerate(keep) if k]].copy()

        garage_candidates = _drop_inside_power_buffer(garage_candidates)
        to_candidates = _drop_inside_power_buffer(to_candidates)

    # Заглушки гаражи/ТО для trunk (если num_garages/num_to > 0 — берём первых кандидатов; при 0 — пусто)
    garage_placeholder = (garage_candidates.iloc[:num_garages].copy() if num_garages > 0 and garage_candidates is not None and len(garage_candidates) > 0 else gpd.GeoDataFrame())
    to_placeholder = (to_candidates.iloc[:num_to].copy() if num_to > 0 and to_candidates is not None and len(to_candidates) > 0 else gpd.GeoDataFrame())

    # Воздушный граф для обхода бесполётных зон и препятствий (используется в магистрали при необходимости обхода)
    air_graph = None
    try:
        if city_boundary is not None:
            gs = GraphService()
            air_graph = gs.build_air_graph(
                city_boundary=city_boundary,
                buildings=buildings,
                no_fly_zones=no_fly_zones,
                building_buffer_deg=0.00025,
                min_distance_deg=0.0003,
                max_points=3000,
                k_neighbors=6,
            )
    except Exception:
        air_graph = None

    # Объединённая геометрия беспилотных зон (только они) для проверки магистрали и A* обхода
    no_fly_zones_union = None
    try:
        geoms = []
        if no_fly_zones is not None:
            if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                try:
                    nfz_4326 = no_fly_zones.to_crs("EPSG:4326")
                except Exception:
                    nfz_4326 = no_fly_zones
                for g in nfz_4326.geometry:
                    if g is not None and not getattr(g, "is_empty", True):
                        geoms.append(g)
            elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                for z in no_fly_zones:
                    g = getattr(z, "geometry", z)
                    if g is not None and hasattr(g, "is_valid") and g.is_valid and not getattr(g, "is_empty", True):
                        geoms.append(g)
        if geoms:
            no_fly_zones_union = unary_union(geoms)
            if no_fly_zones_union is not None and (getattr(no_fly_zones_union, "is_empty", True) or not getattr(no_fly_zones_union, "is_valid", True)):
                no_fly_zones_union = None
    except Exception:
        no_fly_zones_union = None

    # Шаг 3: trunk (с заглушками)
    trunk = placement.build_trunk_graph(
        charge_stations,
        garage_placeholder,
        to_placeholder,
        no_fly_zones_union,
        air_graph=air_graph,
        max_edge_km=R_GARAGE_TO_KM,
    )

    # Шаг 4–5: гаражи и ТО размещаем по покрытию спроса (как зарядки) до 95%
    garages = placement.place_garages(
        demand,
        garage_candidates,
        k=num_garages,
        radius_km=R_GARAGE_TO_KM,
        coverage_ratio_target=0.95,
        min_distance_factor=1.2,  # плотнее: гаражи могут быть ближе друг к другу
        max_garages=100,
    )

    # Чтобы станция ТО не попадала в то же здание/точку, что и гараж,
    # отфильтруем кандидатов ТО, которые слишком близко к уже выбранным гаражам.
    filtered_to_candidates = to_candidates
    if (
        to_candidates is not None
        and len(to_candidates) > 0
        and garages is not None
        and len(garages) > 0
    ):
        try:
            to_pts = _points_array(to_candidates)
            gar_pts = _points_array(garages)
            if len(to_pts) > 0 and len(gar_pts) > 0:
                keep_indices = []
                # Минимальное расстояние между станцией ТО и гаражом (км)
                min_sep_km = 0.2  # ~200 м, чтобы точно не одно и то же здание
                for i in range(len(to_pts)):
                    lon_t, lat_t = to_pts[i, 0], to_pts[i, 1]
                    too_close = False
                    for j in range(len(gar_pts)):
                        lon_g, lat_g = gar_pts[j, 0], gar_pts[j, 1]
                        d_km = haversine_km(lat_t, lon_t, lat_g, lon_g)
                        if d_km < min_sep_km:
                            too_close = True
                            break
                    if not too_close:
                        keep_indices.append(i)
                if keep_indices:
                    filtered_to_candidates = to_candidates.iloc[keep_indices].copy()
                else:
                    # Если всех отфильтровали — оставляем исходных кандидатов, чтобы не потерять ТО вообще.
                    filtered_to_candidates = to_candidates
        except Exception:
            filtered_to_candidates = to_candidates

    to_stations = placement.place_to_stations(
        trunk,
        filtered_to_candidates,
        demand,
        k=num_to,
        radius_km=R_GARAGE_TO_KM,
        coverage_ratio_target=0.95,
        min_distance_factor=1.2,  # плотнее: станции ТО могут быть ближе друг к другу
        max_to_stations=100,
    )

    # Пересобираем trunk с уже выбранными гаражами и ТО
    trunk = placement.build_trunk_graph(
        charge_stations,
        garages,
        to_stations,
        no_fly_zones_union,
        air_graph=air_graph,
        max_edge_km=R_GARAGE_TO_KM,
    )

    # Ветки: Б → (А / гараж / ТО), макс. 3 связи на станцию Б
    branch_edges = placement.build_branch_edges(
        charge_stations,
        garages,
        to_stations,
        k_nearest=3,
        max_branch_km=R_GARAGE_TO_KM,
    )
    # Локальная сеть Б↔Б, макс. 2 связи на станцию
    local_edges = placement.build_local_edges(
        charge_stations, max_edge_km=R_GARAGE_TO_KM, max_neighbors_b=2
    )

    # Шаг 6: метрики
    metrics = placement.compute_metrics(demand, charge_stations, trunk, radius_charge_km=R_CHARGE_KM)
    metrics["charge"] = charge_metrics

    return {
        "buildings": buildings,
        "demand": demand,
        "charge_stations": charge_stations,
        "garages": garages,
        "to_stations": to_stations,
        "trunk_graph": trunk,
        "branch_edges": branch_edges,
        "local_edges": local_edges,
        "metrics": metrics,
        "city_boundary": city_boundary,
        "no_fly_zones": no_fly_zones,
        "params": {
            "R_charge_km": R_CHARGE_KM,
            "R_garage_to_km": R_GARAGE_TO_KM,
            "D_max_km": D_MAX_KM,
        },
    }