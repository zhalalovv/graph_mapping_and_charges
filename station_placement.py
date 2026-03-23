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
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

# Радиус Земли в км
EARTH_R_KM = 6371.0

# FlyCart 30 с грузом: D_max = 16 км (радиусы уменьшены в 4 раза для отображения/расчёта)
D_MAX_KM = 16.0
# Зарядка + ожидание (туда-обратно): R = 0.4 * D_max / 4 = 1.6 км
R_CHARGE_KM = 0.4 * D_MAX_KM / 4.0
# Гаражи и ТО (в одну сторону): R = 0.8 * D_max / 4 = 3.2 км
R_GARAGE_TO_KM = 0.8 * D_MAX_KM / 4.0


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
    if g is None:
        return (0.0, 0.0)
    if hasattr(g, "x") and hasattr(g, "y"):
        return (float(g.x), float(g.y))
    c = getattr(g, "centroid", None)
    if c is not None:
        return (float(c.x), float(c.y))
    return (0.0, 0.0)


def _points_array(gdf: gpd.GeoDataFrame) -> np.ndarray:
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

        weights = demand["weight"].values if "weight" in demand.columns else np.ones(len(demand))
        w_med = float(np.median(weights)) if len(weights) > 0 else 1.0
        w_p75 = float(np.percentile(weights, 75)) if len(weights) > 0 else w_med

        max_dist = max_distance_km if max_distance_km is not None else float("inf")
        dem_pts = _points_array(demand)
        cand_pts = _points_array(full_candidates)
        chosen_indices = set()
        for i in range(len(dem_pts)):
            lat_d, lon_d = dem_pts[i, 1], dem_pts[i, 0]
            w = float(weights[i]) if i < len(weights) else 1.0
            if w >= w_p75:
                n_for_cluster = 3
            elif w >= w_med:
                n_for_cluster = 2
            else:
                n_for_cluster = 1

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
                    if station_type == "charge_a" and "source" in candidates.columns:
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

                # Предрасчет принадлежности кандидатов hull'ам кластеров.
                # Правило:
                # - если точка попала ровно в один hull -> membership = cluster_id этого hull
                # - если в ноль hull или в несколько hull -> membership = None
                # Это гарантирует, что станция "строго" будет принадлежать hull геометрии своего кластера,
                # а в случае пересечений hull не будет размещена в неоднозначной области.
                candidate_membership: List[Optional[Any]] = [None] * len(cand_pts)
                candidate_covered_any: List[bool] = [False] * len(cand_pts)
                candidates_by_cluster: Dict[Any, List[int]] = {}
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
                            sindex = getattr(hulls_gdf, "sindex", None)
                            for ci in range(len(cand_pts)):
                                lon_c, lat_c = cand_pts[ci, 0], cand_pts[ci, 1]
                                pt = Point(lon_c, lat_c)
                                idxs = None
                                if sindex is not None:
                                    try:
                                        idxs = list(sindex.query(pt, predicate="covers"))
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

                # Вычисляем главный кластер по району: максимальный weight у кластер-центроида.
                # Текущие данные demand включают по одной centroid-точке на кластер, поэтому weight уже агрегированный.
                main_cluster_by_region: Dict[Any, Any] = {}
                if region_col is not None and cluster_col is not None and cluster_col in centroid_df.columns:
                    for rid, sub_r in centroid_df.groupby(region_col):
                        if len(sub_r) == 0:
                            continue
                        # центр района (для tie-break по “центральности”)
                        w_r = sub_r["weight"].values if "weight" in sub_r.columns else np.ones(len(sub_r))
                        coords_r = np.array([_coords_from_geom(g) for g in sub_r.geometry], dtype=float)  # lon, lat
                        if len(coords_r) == 0:
                            continue
                        lon_r = float(np.average(coords_r[:, 0], weights=w_r))
                        lat_r = float(np.average(coords_r[:, 1], weights=w_r))

                        # главный кластер = max weight, при равенстве — ближе к центру района
                        best_cid = None
                        best_w = -1.0
                        best_d = float("inf")
                        for cid, sub_c in sub_r.groupby(cluster_col):
                            if len(sub_c) == 0:
                                continue
                            w_c = float(sub_c["weight"].values[0]) if "weight" in sub_c.columns else float(len(sub_c))
                            # centroid точки кластера:
                            lon_c, lat_c = _coords_from_geom(sub_c.iloc[0].geometry)
                            d_c = haversine_km(lat_r, lon_r, lat_c, lon_c)
                            if w_c > best_w or (w_c == best_w and d_c < best_d):
                                best_w = w_c
                                best_d = d_c
                                best_cid = cid
                        if best_cid is not None:
                            main_cluster_by_region[rid] = best_cid

                # 1) charge_a: по району ставим ровно 1 станцию в главный кластер.
                if region_col is not None and force_type_a_per_region and cluster_col is not None:
                    max_a_dist = (
                        mandatory_a_max_distance_km if mandatory_a_max_distance_km is not None else float("inf")
                    )
                    allowed_a = list(range(len(cand_pts)))
                    if "source" in candidates.columns:
                        allowed_a = [i for i in allowed_a if candidates.iloc[i].get("source") == "building"]
                        if len(allowed_a) == 0:
                            allowed_a = list(range(len(cand_pts)))

                    for rid, sub_r in centroid_df.groupby(region_col):
                        if len(sub_r) == 0:
                            continue
                        if rid not in main_cluster_by_region:
                            continue
                        main_cid = main_cluster_by_region[rid]

                        # Строго: кандидат должен лежать внутри hull своего main_cid кластера.
                        # Если кандидатов внутри hull не нашлось — станцию не ставим, чтобы не нарушить правило
                        # "нельзя допускать, чтобы станция оказывалась ближе/внутри другого кластера".
                        target_allowed_a = [i for i in candidates_by_cluster.get(main_cid, []) if i in allowed_a]
                        if not target_allowed_a:
                            # Если точка-кандидат кластера нигде не попала в hull (часто из-за усечения hull'ов),
                            # то хотя бы не допускаем попадания внутрь hull других кластеров:
                            # разрешаем кандидаты только с принадлежностью "снаружи всех hull".
                            target_allowed_a = [i for i in outside_no_hull_indices if i in allowed_a]
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
                    for cid, sub_c in centroid_df.groupby(cluster_col):
                        if len(sub_c) == 0:
                            continue
                        # определяем район для этого кластера (берём region_id из centroid_df)
                        if region_col is None or region_col not in sub_c.columns:
                            rid = None
                        else:
                            rid = sub_c[region_col].values[0] if len(sub_c[region_col].values) > 0 else None
                        if rid is not None and rid in main_cluster_by_region and main_cluster_by_region[rid] == cid:
                            continue  # в главный кластер charge_b не ставим
                        lon_c, lat_c = _coords_from_geom(sub_c.iloc[0].geometry)

                        # Строго: кандидат должен лежать внутри hull своего cid кластера.
                        target_allowed_b = candidates_by_cluster.get(cid, [])
                        if not target_allowed_b:
                            # Если кандидаты внутрь hull отсутствуют — разрешаем только кандидатов снаружи всех hull,
                            # иначе есть риск "попасть в другой кластер".
                            target_allowed_b = outside_no_hull_indices
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

                if skip_greedy_after_forced and (force_type_a_per_region or force_type_b_per_cluster):
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
                        f"[forced] Зарядки: тип А {n_a}, тип Б {n_b}, покрытие {metrics['coverage_ratio']:.2%}"
                    )
                    return gdf, metrics

        # Фаза 1: зарядки типа А — покрывают основную часть спроса.
        # В оценке кандидата берём весь спрос в радиусе (ignore_covered_in_score=True),
        # чтобы станции А «обнимали» основную зону спроса, а не разлетались по мелким островкам.
        run_phase("charge_a", type_a_coverage_ratio, ignore_covered_in_score=True)

        # Фаза 2: станции типа Б (дополняют покрытие оставшегося спроса).
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

        nfz_union = None
        if obstacles is not None and getattr(obstacles, "is_empty", True) is False and getattr(obstacles, "is_valid", True):
            nfz_union = obstacles

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

        pts = np.array([[lon, lat] for _, lon, lat in trunk_nodes])
        ids = [nid for nid, _, _ in trunk_nodes]
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
            for d_km, j in cand[:k]:
                nid_j = ids[j]
                if G.has_edge(nid_i, nid_j):
                    continue
                lon_j, lat_j = pts[j, 0], pts[j, 1]
                # Проверяем: проходит ли прямая между станциями через беспилотную зону
                if not _edge_intersects_nfz(lon_i, lat_i, lon_j, lat_j):
                    G.add_edge(nid_i, nid_j, weight=d_km, length=d_km)
                    continue
                # Отрезок пересекает беспилотную зону — строим обход по воздушному графу (A*)
                self.logger.debug(
                    "Магистраль пересекает беспилотную зону %s–%s, строим обход A*", nid_i, nid_j
                )
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
                        self.logger.info(
                            "Обход беспилотной зоны: %s–%s, точек маршрута %s",
                            nid_i,
                            nid_j,
                            len(detour_coords),
                        )
                        continue
                self.logger.warning("Обход не построен для %s–%s, оставлена прямая", nid_i, nid_j)
                G.add_edge(nid_i, nid_j, weight=d_km, length=d_km)

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
        self.logger.info(
            f"Trunk (А–А, макс. {max_neighbors_a} соседей): {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер"
        )
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
        if not b_indices:
            return []
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
        kk = min(k_nearest, len(targets_pts))
        for bi, idx_b in enumerate(b_indices):
            lon_b, lat_b = b_pts[bi, 0], b_pts[bi, 1]
            cand = []
            for lon_t, lat_t, tid in targets_pts:
                d_km = haversine_km(lat_b, lon_b, lat_t, lon_t)
                if d_km <= max_branch_km:
                    cand.append((d_km, lon_t, lat_t, tid))
            cand.sort(key=lambda x: x[0])
            for d_km, lon_t, lat_t, tid in cand[:kk]:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": [[lon_b, lat_b], [lon_t, lat_t]]},
                        "properties": {
                            "edge_type": "branch",
                            "source_id": str(idx_b),
                            "target_id": tid,
                            "weight_km": round(d_km, 4),
                        },
                    }
                )
        self.logger.info(f"Ветки Б→А/гараж/ТО (макс. {kk} на станцию): {len(features)} рёбер")
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
            cand = []
            for j in range(len(b_indices)):
                if j == i:
                    continue
                d_km = haversine_km(b_pts[i, 1], b_pts[i, 0], b_pts[j, 1], b_pts[j, 0])
                if d_km <= max_edge_km:
                    cand.append((d_km, j))
            cand.sort(key=lambda x: x[0])
            for d_km, j in cand[:k]:
                ui, vi = b_indices[i], b_indices[j]
                edge_key = (min(ui, vi), max(ui, vi))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                features.append(
                    {
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

    if demand_method == "dbscan" and charge_candidates_full is not None and len(charge_candidates_full) > 0:
        charge_candidates = placement.build_charge_candidates_from_clusters(
            demand, charge_candidates_full
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
                        lon_s, lat_s = _coords_from_geom(srow.geometry)
                        pt = Point(lon_s, lat_s)

                        idxs = None
                        if sindex is not None:
                            try:
                                idxs = list(sindex.query(pt, predicate="covers"))
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
                        elif len(unique_cids) == 0:
                            # Точка не попала ни в один hull: назначаем ближайший hull по расстоянию,
                            # но только если он заметно ближе второго.
                            try:
                                best_cid: Optional[int] = None
                                best_d: Optional[float] = None
                                second_d: Optional[float] = None
                                for hi in range(len(hulls_gdf)):
                                    geom = hulls_gdf.geometry.iloc[int(hi)]
                                    if geom is None or not geom.is_valid:
                                        continue
                                    d = float(geom.distance(pt))
                                    cid = int(hulls_gdf.iloc[int(hi)]["cluster_id"])
                                    if best_d is None or d < best_d:
                                        second_d = best_d
                                        best_d = d
                                        best_cid = cid
                                    elif second_d is None or d < second_d:
                                        second_d = d
                                if best_cid is not None and best_d is not None and second_d is not None:
                                    # Условие "заметно ближе": минимум на 5% меньше расстояния до второго.
                                    if second_d <= 0:
                                        assigned[pos] = None
                                    elif best_d < second_d * 0.95:
                                        assigned[pos] = best_cid
                                    else:
                                        assigned[pos] = None
                                else:
                                    assigned[pos] = None
                            except Exception:
                                assigned[pos] = None
                        else:
                            assigned[pos] = None
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

    # --- Отключаем гаражи/ТО и магистраль (trunk) по запросу пользователя ---
    garages = gpd.GeoDataFrame()
    to_stations = gpd.GeoDataFrame()
    trunk = None
    branch_edges: List[Dict[str, Any]] = []
    local_edges: List[Dict[str, Any]] = []

    metrics = placement.compute_metrics(demand, charge_stations, trunk)
    metrics["charge"] = charge_metrics
    metrics["cluster_count"] = cluster_count
    metrics["charging_buildings_in_clusters"] = charging_buildings_in_clusters

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
            # Радиусы отсутствуют: покрытие и привязка делаются по `cluster_id`.
        },
    }


def _empty_fc() -> Dict[str, Any]:
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
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": line_coords},
                "properties": {
                    "source": str(u),
                    "target": str(v),
                    "weight_km": float(w) if w is not None else None,
                },
            }
        )
    return feats


def pipeline_result_to_geojson(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ответ API `/api/stations/placement`: слои для карты + метрики и параметры."""
    err = raw.get("error")
    charge = raw.get("charge_stations")
    if charge is not None and len(charge) > 0 and "station_type" in charge.columns:
        charging_a = charge[charge["station_type"] == "charge_a"]
        charging_b = charge[charge["station_type"] == "charge_b"]
    elif charge is not None and len(charge) > 0:
        charging_a = charge
        charging_b = gpd.GeoDataFrame()
    else:
        charging_a = gpd.GeoDataFrame()
        charging_b = gpd.GeoDataFrame()

    trunk_feats = trunk_graph_to_geojson_features(raw.get("trunk_graph"))
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

    out: Dict[str, Any] = {
        "charging_type_a": _gdf_to_fc(charging_a),
        "charging_type_b": _gdf_to_fc(charging_b),
        "garages": _gdf_to_fc(raw.get("garages")),
        "to_stations": _gdf_to_fc(raw.get("to_stations")),
        "trunk": {"type": "FeatureCollection", "features": trunk_feats},
        "branch_edges": _list_features_to_fc(raw.get("branch_edges")),
        "local_edges": _list_features_to_fc(raw.get("local_edges")),
        "metrics": raw.get("metrics") or {},
        "params": raw.get("params") or {},
        "cluster_centroids": _gdf_to_fc(cluster_centroids),
    }
    if err:
        out["error"] = err
    return out
