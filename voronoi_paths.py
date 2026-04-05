"""
Локальные рёбра диаграммы Вороного по кластерам спроса.
Сайты — центроиды зданий (возможно агрегированные) и точки зарядных станций в кластере.
"""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np

import pandas as pd
from shapely.geometry import LineString, Point

from data_service import DataService


def _normalize_hulls_gdf(hulls_gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    if hulls_gdf is None or len(hulls_gdf) == 0:
        return None
    h = hulls_gdf.to_crs("EPSG:4326").copy()
    if "cluster_id" not in h.columns:
        if "subcluster_id" in h.columns:
            h["cluster_id"] = h["subcluster_id"]
        else:
            return None
    h = h[h["cluster_id"].notna()].copy()
    return h if len(h) > 0 else None


def _hull_polygon_for_cluster(hulls_norm: gpd.GeoDataFrame | None, cluster_id: Any):
    if hulls_norm is None or len(hulls_norm) == 0:
        return None
    try:
        sub = hulls_norm[hulls_norm["cluster_id"] == cluster_id]
        if len(sub) == 0:
            sub = hulls_norm[hulls_norm["cluster_id"].astype(str) == str(cluster_id)]
        if len(sub) == 0:
            return None
        g = sub.geometry.iloc[0]
        if g is None or getattr(g, "is_empty", True):
            return None
        return g
    except Exception:
        return None


def _point_in_hull(lon: float, lat: float, hull_poly) -> bool:
    if hull_poly is None or getattr(hull_poly, "is_empty", True):
        return True
    try:
        p = Point(float(lon), float(lat))
        return bool(hull_poly.covers(p) or hull_poly.contains(p))
    except Exception:
        return False


def _clip_segment_to_hull_coords(
    lon1: float, lat1: float, lon2: float, lat2: float, hull_poly
) -> list[list[float]] | None:
    """Обрезает отрезок между сайтами по полигону кластера (EPSG:4326)."""
    if hull_poly is None or getattr(hull_poly, "is_empty", True):
        return [[lon1, lat1], [lon2, lat2]]
    try:
        line = LineString([(lon1, lat1), (lon2, lat2)])
        inter = line.intersection(hull_poly.buffer(0))
        if inter.is_empty:
            return None
        if inter.geom_type == "LineString":
            return [[float(x), float(y)] for x, y in inter.coords]
        if inter.geom_type == "MultiLineString":
            best = max(inter.geoms, key=lambda gg: gg.length)
            return [[float(x), float(y)] for x, y in best.coords]
        return None
    except Exception:
        return [[lon1, lat1], [lon2, lat2]]


def _filter_station_features_to_hulls(
    features: list | None,
    hulls_norm: gpd.GeoDataFrame | None,
) -> list[dict]:
    """Оставляет только точки станций, геометрически попадающие в hull своего cluster_id."""
    if not features:
        return []
    if hulls_norm is None or len(hulls_norm) == 0:
        return [f for f in features if isinstance(f, dict)]
    out: list[dict] = []
    for f in features:
        if not isinstance(f, dict):
            continue
        geom = f.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = f.get("properties") or {}
        cid = props.get("cluster_id")
        if cid is None:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        hp = _hull_polygon_for_cluster(hulls_norm, cid)
        if hp is not None and not _point_in_hull(lon, lat, hp):
            continue
        out.append(f)
    return out


def charging_station_gdfs_to_features(*layers) -> list[dict]:
    """
    Точки зарядок / гаражей / ТО с привязкой к кластеру для построения Вороного вместе со спросом.
    Берётся cluster_id; если нет — subcluster_id.
    """
    out: list[dict] = []
    for gdf in layers:
        if gdf is None or len(gdf) == 0:
            continue
        for _, row in gdf.iterrows():
            g = row.geometry
            if g is None or getattr(g, "is_empty", True):
                continue
            cid = row.get("cluster_id")
            if cid is None or (isinstance(cid, float) and pd.isna(cid)):
                cid = row.get("subcluster_id")
            if cid is None or (isinstance(cid, float) and pd.isna(cid)):
                continue
            try:
                lon, lat = float(g.x), float(g.y)
            except Exception:
                continue
            try:
                cid_val = int(cid)
            except (TypeError, ValueError):
                cid_val = str(cid)
            out.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"cluster_id": cid_val},
                }
            )
    return out


def dedupe_close_parallel_voronoi_edges(
    fc: dict,
    *,
    max_midpoint_distance_m: float = 90.0,
    max_angle_diff_deg: float = 18.0,
) -> dict:
    """Убирает почти дублирующиеся близкие и параллельные рёбра Вороного."""
    if not fc or not isinstance(fc, dict):
        return {"type": "FeatureCollection", "features": []}
    feats = list(fc.get("features", []) or [])
    if len(feats) < 2:
        return {"type": "FeatureCollection", "features": feats}

    angle_thr = np.radians(float(max_angle_diff_deg))

    def _xy_m(lon: float, lat: float, ref_lat: float) -> tuple[float, float]:
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.radians(ref_lat))
        return lon * m_per_deg_lon, lat * m_per_deg_lat

    def _enrich(feature: dict):
        geom = (feature or {}).get("geometry") or {}
        if geom.get("type") != "LineString":
            return None
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            return None
        p1 = coords[0]
        p2 = coords[-1]
        if len(p1) < 2 or len(p2) < 2:
            return None
        lon1, lat1 = float(p1[0]), float(p1[1])
        lon2, lat2 = float(p2[0]), float(p2[1])
        ref_lat = 0.5 * (lat1 + lat2)
        x1, y1 = _xy_m(lon1, lat1, ref_lat)
        x2, y2 = _xy_m(lon2, lat2, ref_lat)
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 0.1:
            return None
        ang = float(np.arctan2(dy, dx))
        if ang < 0:
            ang += np.pi
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        return {
            "feature": feature,
            "cluster": str(((feature.get("properties") or {}).get("cluster_id"))),
            "p1": (x1, y1),
            "p2": (x2, y2),
            "mx": mx,
            "my": my,
            "angle": ang,
            "length": length,
            "ux": dx / length,
            "uy": dy / length,
        }

    enriched = [e for e in (_enrich(f) for f in feats) if e is not None]
    if len(enriched) < 2:
        return {"type": "FeatureCollection", "features": feats}

    by_cluster: dict[str, list[dict]] = {}
    for e in enriched:
        by_cluster.setdefault(e["cluster"], []).append(e)

    kept_features: list[dict] = []
    for _, items in by_cluster.items():
        items.sort(key=lambda it: it["length"], reverse=True)
        kept: list[dict] = []
        for cur in items:
            is_dup = False
            for k in kept:
                d_ang = abs(cur["angle"] - k["angle"])
                d_ang = min(d_ang, np.pi - d_ang)
                if d_ang > angle_thr:
                    continue
                d_mid = float(np.hypot(cur["mx"] - k["mx"], cur["my"] - k["my"]))
                if d_mid > max_midpoint_distance_m:
                    continue
                nx, ny = -k["uy"], k["ux"]
                perp_cur = abs((cur["mx"] - k["mx"]) * nx + (cur["my"] - k["my"]) * ny)
                if perp_cur > (max_midpoint_distance_m * 0.7):
                    continue

                tx, ty = k["ux"], k["uy"]
                cur_min_t = min(cur["p1"][0] * tx + cur["p1"][1] * ty, cur["p2"][0] * tx + cur["p2"][1] * ty)
                cur_max_t = max(cur["p1"][0] * tx + cur["p1"][1] * ty, cur["p2"][0] * tx + cur["p2"][1] * ty)
                k_min_t = min(k["p1"][0] * tx + k["p1"][1] * ty, k["p2"][0] * tx + k["p2"][1] * ty)
                k_max_t = max(k["p1"][0] * tx + k["p1"][1] * ty, k["p2"][0] * tx + k["p2"][1] * ty)
                overlap = max(0.0, min(cur_max_t, k_max_t) - max(cur_min_t, k_min_t))
                min_len = max(1.0, min(cur["length"], k["length"]))
                if overlap >= (0.35 * min_len):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cur)
        kept_features.extend([k["feature"] for k in kept])

    return {"type": "FeatureCollection", "features": kept_features}


def _require_voronoi():
    try:
        from scipy.spatial import Voronoi

        return Voronoi
    except Exception as e:
        raise RuntimeError(f"Scipy Voronoi недоступен: {e}") from e


def merge_xy_with_station_flags(
    building_xy: np.ndarray,
    station_xy: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Объединяет точки центроидов и станций; при совпадении координат сайт считается станцией (логическое ИЛИ).
    """
    rows: list[tuple[float, float, bool]] = []
    if building_xy is not None and len(building_xy) > 0:
        b = np.asarray(building_xy, dtype=float)
        for i in range(len(b)):
            rows.append((float(b[i, 0]), float(b[i, 1]), False))
    if station_xy is not None and len(station_xy) > 0:
        s = np.asarray(station_xy, dtype=float)
        for i in range(len(s)):
            rows.append((float(s[i, 0]), float(s[i, 1]), True))
    if not rows:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=bool)
    merged: dict[tuple[float, float], bool] = {}
    order_keys: list[tuple[float, float]] = []
    for x, y, is_st in rows:
        key = (round(x, 6), round(y, 6))
        if key not in merged:
            merged[key] = is_st
            order_keys.append(key)
        else:
            merged[key] = merged[key] or is_st
    pts = np.array([[k[0], k[1]] for k in order_keys], dtype=float)
    flags = np.array([merged[k] for k in order_keys], dtype=bool)
    return pts, flags


def _station_edge_props(
    cluster_prop: Any,
    edge_type: str,
    a: int,
    b: int,
    is_station: np.ndarray,
    *,
    group_mode: str | None = None,
    source_i: int | None = None,
    target_i: int | None = None,
) -> dict:
    sa = bool(is_station[a]) if a < len(is_station) else False
    sb = bool(is_station[b]) if b < len(is_station) else False
    props: dict[str, Any] = {
        "cluster_id": cluster_prop,
        "edge_type": edge_type,
        "source_i": int(source_i if source_i is not None else a),
        "target_i": int(target_i if target_i is not None else b),
        "source_is_station": sa,
        "target_is_station": sb,
        "connects_station": sa or sb,
    }
    if group_mode is not None:
        props["group_mode"] = group_mode
    return props


def _append_voronoi_edges_for_cluster(
    features: list[dict],
    pts_arr: np.ndarray,
    is_station: np.ndarray,
    group_id: Any,
    edge_type: str,
    *,
    group_mode: str | None = None,
    hull_poly=None,
) -> bool:
    """Добавляет рёбра Вороного для одного кластера (только внутри hull_poly, если задан)."""
    Voronoi = _require_voronoi()
    n0 = len(features)
    cid = int(group_id) if isinstance(group_id, (int, np.integer)) else str(group_id)

    if len(pts_arr) < 2:
        return False

    if len(pts_arr) == 2:
        p0, p1 = pts_arr[0], pts_arr[1]
        clipped = _clip_segment_to_hull_coords(
            float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), hull_poly
        )
        if clipped is None or len(clipped) < 2:
            return False
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": clipped,
                },
                "properties": _station_edge_props(cid, edge_type, 0, 1, is_station, group_mode=group_mode),
            }
        )
        return True

    coords = np.asarray(pts_arr, dtype=float)
    flags = np.asarray(is_station, dtype=bool)
    if len(flags) != len(coords):
        flags = np.zeros(len(coords), dtype=bool)

    map_idx: dict[int, int] = {i: i for i in range(len(coords))}
    try:
        vor = Voronoi(coords)
    except Exception:
        uniq, inv = np.unique(coords, axis=0, return_inverse=True)
        if len(uniq) < 3:
            return False
        fu = np.zeros(len(uniq), dtype=bool)
        for k in range(len(flags)):
            fu[inv[k]] |= flags[k]
        coords = uniq
        flags = fu
        map_idx = {i: i for i in range(len(coords))}
        try:
            vor = Voronoi(coords)
        except Exception:
            return False

    seen_pairs = set()
    for ridge in vor.ridge_points:
        i_raw, j_raw = int(ridge[0]), int(ridge[1])
        if i_raw not in map_idx or j_raw not in map_idx:
            continue
        i = map_idx[i_raw]
        j = map_idx[j_raw]
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        pi = coords[a]
        pj = coords[b]
        clipped = _clip_segment_to_hull_coords(
            float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1]), hull_poly
        )
        if clipped is None or len(clipped) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": clipped,
                },
                "properties": _station_edge_props(cid, edge_type, a, b, flags, group_mode=group_mode),
            }
        )
    return len(features) > n0


def _cluster_label_for_props(cid: Any) -> Any:
    if isinstance(cid, (int, np.integer)):
        return int(cid)
    return str(cid)


def _inter_cluster_pair_label(cid_a: Any, cid_b: Any) -> str:
    sa, sb = sorted((str(cid_a), str(cid_b)))
    return f"{sa}|{sb}"


def _canonical_cluster_pair_key(cid_a: Any, cid_b: Any) -> tuple[str, str]:
    return tuple(sorted((str(cid_a), str(cid_b))))


def _segment_length_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Длина отрезка в метрах (локально equirectangular по средней широте)."""
    lat0 = 0.5 * (lat1 + lat2)
    m_lat = 111_320.0
    m_lon = 111_320.0 * float(np.cos(np.radians(lat0)))
    return float(np.hypot((lon2 - lon1) * m_lon, (lat2 - lat1) * m_lat))


def _inter_cluster_pairs_present(features: list[dict]) -> set[tuple[str, str]]:
    """Уже есть хотя бы одно межкластерное ребро между этой парой cluster_id."""
    out: set[tuple[str, str]] = set()
    for f in features:
        p = f.get("properties") or {}
        if not p.get("inter_cluster"):
            continue
        cid = p.get("cluster_id")
        if isinstance(cid, str) and "|" in cid:
            parts = cid.split("|")
            if len(parts) == 2:
                out.add(tuple(sorted((parts[0], parts[1]))))
    return out


def _add_shortest_bridges_for_allowed_pairs(
    features: list[dict],
    coords: np.ndarray,
    flags: np.ndarray,
    cluster_per_point: list[Any],
    allowed_pair_keys: set[tuple[str, str]],
    max_edge_m: float,
) -> None:
    """
    Если для пары кластеров из allowed_pair_keys Delaunay не дал ни одного ребра,
    добавляет отрезок между ближайшими сайтами (длина <= max_edge_m).
    """
    if max_edge_m <= 0 or not allowed_pair_keys:
        return
    by_c: dict[str, list[int]] = {}
    for i, c in enumerate(cluster_per_point):
        key = str(c)
        by_c.setdefault(key, []).append(i)
    have = _inter_cluster_pairs_present(features)
    for a, b in allowed_pair_keys:
        pk = tuple(sorted((a, b)))
        if pk in have:
            continue
        ia = by_c.get(a, [])
        ib = by_c.get(b, [])
        if not ia or not ib:
            continue
        best_i, best_j = -1, -1
        best_d = float("inf")
        for i in ia:
            pi = coords[i]
            for j in ib:
                pj = coords[j]
                d = _segment_length_m(float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1]))
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i < 0 or best_d > max_edge_m:
            continue
        pi = coords[best_i]
        pj = coords[best_j]
        ca, cb = cluster_per_point[best_i], cluster_per_point[best_j]
        cid_prop = _inter_cluster_pair_label(ca, cb)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(pi[0]), float(pi[1])],
                        [float(pj[0]), float(pj[1])],
                    ],
                },
                "properties": {
                    **_station_edge_props(
                        cid_prop,
                        "voronoi_local",
                        best_i,
                        best_j,
                        flags,
                        group_mode=None,
                    ),
                    "inter_cluster": True,
                    "bridge_gap_fill": True,
                },
            }
        )
        have.add(pk)


def _bordering_cluster_pair_keys(
    hulls_norm: gpd.GeoDataFrame | None,
    *,
    max_boundary_gap_m: float = 45.0,
) -> set[tuple[str, str]]:
    """
    Пары cluster_id, у которых полигоны hull в EPSG:3857 касаются или разделены не больше max_boundary_gap_m.
    Так отсекаются длинные рёбра Делоне между далёкими кластерами (через реку и т.п.).
    """
    if hulls_norm is None or len(hulls_norm) < 2:
        return set()
    try:
        h = hulls_norm.to_crs(epsg=3857)
    except Exception:
        return set()
    out: set[tuple[str, str]] = set()
    ids: list[str] = [str(x) for x in h["cluster_id"].tolist()]
    geoms: list = []
    for g in h.geometry:
        if g is None or getattr(g, "is_empty", True):
            geoms.append(None)
            continue
        try:
            geoms.append(g.buffer(0))
        except Exception:
            geoms.append(g)
    tol = float(max_boundary_gap_m)
    n = len(ids)
    for i in range(n):
        gi = geoms[i]
        if gi is None:
            continue
        for j in range(i + 1, n):
            gj = geoms[j]
            if gj is None:
                continue
            if ids[i] == ids[j]:
                continue
            try:
                d = float(gi.distance(gj))
            except Exception:
                continue
            if d <= tol:
                out.add(tuple(sorted((ids[i], ids[j]))))
    return out


def _separate_coincident_sites(coords: np.ndarray) -> np.ndarray:
    """Сдвигает дубликаты координат на ~0.5 m, чтобы сайты разных кластеров не схлопывались в одном Voronoi."""
    c = np.asarray(coords, dtype=float).copy()
    if len(c) <= 1:
        return c
    ref_lat = float(np.mean(c[:, 1]))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = max(111_320.0 * np.cos(np.radians(ref_lat)), 1e-3)
    step_lat = 0.5 / m_per_deg_lat
    step_lon = 0.5 / m_per_deg_lon
    seen: dict[tuple[float, float], int] = {}
    for i in range(len(c)):
        key = (round(float(c[i, 0]), 9), round(float(c[i, 1]), 9))
        n = seen.get(key, 0)
        if n > 0:
            k = n
            c[i, 0] += (k // 4) * step_lon
            c[i, 1] += (k % 4) * step_lat
        seen[key] = n + 1
    return c


def _append_voronoi_edges_global(
    features: list[dict],
    pts_arr: np.ndarray,
    is_station: np.ndarray,
    cluster_per_point: list[Any],
    hulls_norm: gpd.GeoDataFrame | None,
    edge_type: str = "voronoi_local",
    *,
    bordering_pair_keys: set[tuple[str, str]] | None = None,
    inter_cluster_max_edge_length_m: float | None = None,
) -> bool:
    """
    Один Voronoi по всем сайтам: рёбра — пары ridge_points (как у Делоне).
    Отрезок всегда между двумя сайтами целиком (без обрезки по hull), в т.ч. внутри одного кластера —
    чтобы линии не обрывались на границе полигона.
    Межкластерные рёбра — только если пара в bordering_pair_keys (смежные hull / зазор до порога),
    и длина отрезка не больше inter_cluster_max_edge_length_m (если задано).
    """
    Voronoi = _require_voronoi()
    n0 = len(features)

    if len(pts_arr) < 2:
        return False

    c_labels = list(cluster_per_point)
    if len(c_labels) != len(pts_arr):
        return False

    if len(pts_arr) == 2:
        coords2 = _separate_coincident_sites(np.asarray(pts_arr, dtype=float))
        p0, p1 = coords2[0], coords2[1]
        c0, c1 = c_labels[0], c_labels[1]
        flags = np.asarray(is_station, dtype=bool)
        if len(flags) != 2:
            flags = np.array([False, False], dtype=bool)
        if str(c0) == str(c1):
            clipped = [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]
            cid_prop = _cluster_label_for_props(c0)
        else:
            pk = _canonical_cluster_pair_key(c0, c1)
            if bordering_pair_keys is not None and pk not in bordering_pair_keys:
                return len(features) > n0
            if inter_cluster_max_edge_length_m is not None and inter_cluster_max_edge_length_m > 0:
                sl = _segment_length_m(
                    float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])
                )
                if sl > inter_cluster_max_edge_length_m:
                    return len(features) > n0
            clipped = [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]
            cid_prop = _inter_cluster_pair_label(c0, c1)
        if clipped is None or len(clipped) < 2:
            return False
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": clipped},
                "properties": {
                    **_station_edge_props(cid_prop, edge_type, 0, 1, flags, group_mode=None),
                    "inter_cluster": str(c0) != str(c1),
                },
            }
        )
        return True

    coords = _separate_coincident_sites(np.asarray(pts_arr, dtype=float))
    flags = np.asarray(is_station, dtype=bool)
    labels = list(c_labels)
    if len(flags) != len(coords):
        flags = np.zeros(len(coords), dtype=bool)

    map_idx: dict[int, int] = {i: i for i in range(len(coords))}
    try:
        vor = Voronoi(coords)
    except Exception:
        rng = np.random.default_rng(42)
        coords = coords + (rng.random(coords.shape) - 0.5) * 1e-6
        try:
            vor = Voronoi(coords)
        except Exception:
            uniq, inv = np.unique(coords, axis=0, return_inverse=True)
            if len(uniq) < 2:
                return False
            fu = np.zeros(len(uniq), dtype=bool)
            lab_u: list[Any] = [None] * len(uniq)
            for k in range(len(flags)):
                u = int(inv[k])
                fu[u] |= bool(flags[k])
                if lab_u[u] is None:
                    lab_u[u] = labels[k]
            if any(x is None for x in lab_u):
                return False
            coords = uniq
            flags = fu
            labels = lab_u
            map_idx = {i: i for i in range(len(coords))}
            if len(coords) == 2:
                return _append_voronoi_edges_global(
                    features,
                    coords,
                    flags,
                    labels,
                    hulls_norm,
                    edge_type=edge_type,
                    bordering_pair_keys=bordering_pair_keys,
                    inter_cluster_max_edge_length_m=inter_cluster_max_edge_length_m,
                )
            try:
                vor = Voronoi(coords)
            except Exception:
                return False

    seen_pairs: set[tuple[int, int]] = set()
    for ridge in vor.ridge_points:
        i_raw, j_raw = int(ridge[0]), int(ridge[1])
        if i_raw not in map_idx or j_raw not in map_idx:
            continue
        i = map_idx[i_raw]
        j = map_idx[j_raw]
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        pi = coords[a]
        pj = coords[b]
        ca, cb = labels[a], labels[b]
        same = str(ca) == str(cb)
        if same:
            clipped = [
                [float(pi[0]), float(pi[1])],
                [float(pj[0]), float(pj[1])],
            ]
            cid_prop = _cluster_label_for_props(ca)
        else:
            pk = _canonical_cluster_pair_key(ca, cb)
            if bordering_pair_keys is not None and pk not in bordering_pair_keys:
                continue
            if inter_cluster_max_edge_length_m is not None and inter_cluster_max_edge_length_m > 0:
                sl = _segment_length_m(
                    float(pi[0]), float(pi[1]), float(pj[0]), float(pj[1])
                )
                if sl > inter_cluster_max_edge_length_m:
                    continue
            clipped = [
                [float(pi[0]), float(pi[1])],
                [float(pj[0]), float(pj[1])],
            ]
            cid_prop = _inter_cluster_pair_label(ca, cb)
        if clipped is None or len(clipped) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": clipped},
                "properties": {
                    **_station_edge_props(cid_prop, edge_type, a, b, flags, group_mode=None),
                    "inter_cluster": not same,
                },
            }
        )
    return len(features) > n0


def build_voronoi_local_paths_fc(
    data_service: DataService,
    city: str,
    network_type: str,
    simplify: bool,
    dbscan_eps_m: float,
    dbscan_min_samples: int,
    use_all_buildings: bool,
    buildings_per_centroid: int = 60,
    charging_station_features: list | None = None,
    inter_cluster_max_hull_gap_m: float = 2000.0,
    inter_cluster_max_edge_length_m: float = 2000.0,
) -> dict:
    _require_voronoi()

    data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
    buildings = data.get("buildings")
    road_graph = data.get("road_graph")
    city_boundary = data.get("city_boundary")
    no_fly_zones = data.get("no_fly_zones")

    if buildings is None or len(buildings) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    buildings_wgs = buildings.to_crs("EPSG:4326").copy()
    if not use_all_buildings:
        buildings_wgs = data_service._filter_buildings_for_demand(buildings_wgs)
    if buildings_wgs is None or len(buildings_wgs) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    hulls_gdf = data.get("demand_hulls")
    if hulls_gdf is None or len(hulls_gdf) == 0:
        result = data_service.get_demand_points_weighted(
            buildings_wgs,
            road_graph,
            city_boundary,
            method="dbscan",
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
            return_hulls=True,
            use_all_buildings=use_all_buildings,
            no_fly_zones=no_fly_zones,
        )
        _, hulls_gdf = result if isinstance(result, tuple) else (None, None)

    if hulls_gdf is None or len(hulls_gdf) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    hulls = hulls_gdf.to_crs("EPSG:4326").copy()
    if "cluster_id" not in hulls.columns:
        if "subcluster_id" in hulls.columns:
            hulls["cluster_id"] = hulls["subcluster_id"]
        elif "region_id" in hulls.columns:
            hulls["cluster_id"] = hulls["region_id"]
        else:
            return {
                "type": "FeatureCollection",
                "features": [],
                "clusters_total": 0,
                "clusters_with_paths": 0,
                "edges_total": 0,
            }
    hulls = hulls[hulls["cluster_id"].notna()].copy()
    if len(hulls) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    hulls_norm = _normalize_hulls_gdf(hulls)

    b_proj = buildings_wgs.to_crs(epsg=3857)
    b_centroids_wgs = gpd.GeoSeries(b_proj.geometry.centroid, crs=b_proj.crs).to_crs("EPSG:4326")
    b_points = gpd.GeoDataFrame({"geometry": b_centroids_wgs}, index=buildings_wgs.index, crs="EPSG:4326")
    try:
        assigned = gpd.sjoin(
            b_points,
            hulls[["cluster_id", "geometry"]],
            how="inner",
            predicate="within",
        )
    except Exception:
        assigned = gpd.sjoin(
            b_points,
            hulls[["cluster_id", "geometry"]],
            how="inner",
            op="within",
        )

    if assigned is None or len(assigned) < 2:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }
    assigned = assigned.copy()
    if "index_left" in assigned.columns:
        assigned["_left_id"] = assigned["index_left"]
    else:
        assigned["_left_id"] = assigned.index
    assigned = assigned.drop_duplicates(subset=["_left_id"]).set_index("_left_id")
    assigned = assigned[assigned["cluster_id"].notna()].copy()
    if len(assigned) < 2:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": 0,
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    charging_by_cluster: dict[str, list[list[float]]] = {}
    if charging_station_features:
        for f in charging_station_features:
            if not isinstance(f, dict):
                continue
            geom = f.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            props = f.get("properties") or {}
            cid = props.get("cluster_id")
            if cid is None:
                continue
            key = str(cid)
            charging_by_cluster.setdefault(key, []).append([float(coords[0]), float(coords[1])])

    target_group_size = max(1, int(buildings_per_centroid or 20))
    features: list[dict] = []
    clusters_total = 0
    all_coords: list[list[float]] = []
    all_flags: list[bool] = []
    all_cluster: list[Any] = []

    for group_id, sub in assigned.groupby("cluster_id"):
        if sub is None or len(sub) < 2:
            continue

        pts = []
        for _, row in sub.iterrows():
            geom = row.get("geometry")
            if geom is None or geom.is_empty:
                continue
            try:
                pts.append([float(geom.x), float(geom.y)])
            except Exception:
                continue
        if len(pts) < 2:
            continue

        pts_arr = np.array(pts, dtype=float)
        if len(pts_arr) > target_group_size:
            n_groups = int(np.ceil(len(pts_arr) / float(target_group_size)))
            n_groups = max(2, min(n_groups, len(pts_arr)))
            try:
                from sklearn.cluster import MiniBatchKMeans

                km = MiniBatchKMeans(
                    n_clusters=n_groups,
                    random_state=42,
                    batch_size=min(4096, max(1024, len(pts_arr) // 3)),
                    n_init=3,
                    max_iter=200,
                )
                labels = km.fit_predict(pts_arr)
                centers = []
                for k in range(n_groups):
                    members = pts_arr[labels == k]
                    if len(members) == 0:
                        continue
                    centers.append(members.mean(axis=0))
                if len(centers) >= 2:
                    pts_arr = np.asarray(centers, dtype=float)
            except Exception:
                order = np.lexsort((pts_arr[:, 1], pts_arr[:, 0]))
                ordered = pts_arr[order]
                centers = []
                for i in range(0, len(ordered), target_group_size):
                    chunk = ordered[i : i + target_group_size]
                    if len(chunk) == 0:
                        continue
                    centers.append(chunk.mean(axis=0))
                if len(centers) >= 2:
                    pts_arr = np.asarray(centers, dtype=float)

        st_points = charging_by_cluster.get(str(group_id), [])
        hull_poly = _hull_polygon_for_cluster(hulls_norm, group_id)
        if st_points and hull_poly is not None:
            st_points = [
                [float(x), float(y)]
                for x, y in st_points
                if _point_in_hull(float(x), float(y), hull_poly)
            ]
        st_arr = np.asarray(st_points, dtype=float) if st_points else None
        pts_arr, is_station = merge_xy_with_station_flags(pts_arr, st_arr)

        if len(pts_arr) < 2:
            continue
        clusters_total += 1
        for k in range(len(pts_arr)):
            all_coords.append([float(pts_arr[k, 0]), float(pts_arr[k, 1])])
            all_flags.append(bool(is_station[k]))
            all_cluster.append(group_id)

    if len(all_coords) < 2:
        return {
            "type": "FeatureCollection",
            "features": [],
            "clusters_total": int(clusters_total),
            "clusters_with_paths": 0,
            "edges_total": 0,
        }

    coords_g = np.asarray(all_coords, dtype=float)
    flags_g = np.asarray(all_flags, dtype=bool)
    bordering_keys = _bordering_cluster_pair_keys(
        hulls_norm, max_boundary_gap_m=float(inter_cluster_max_hull_gap_m)
    )
    cap_edge = float(inter_cluster_max_edge_length_m)
    edge_cap: float | None = cap_edge if cap_edge > 0 else None
    _append_voronoi_edges_global(
        features,
        coords_g,
        flags_g,
        all_cluster,
        hulls_norm,
        "voronoi_local",
        bordering_pair_keys=bordering_keys,
        inter_cluster_max_edge_length_m=edge_cap,
    )
    _add_shortest_bridges_for_allowed_pairs(
        features,
        coords_g,
        flags_g,
        all_cluster,
        bordering_keys,
        max_edge_m=cap_edge,
    )

    deduped = dedupe_close_parallel_voronoi_edges(
        {
            "type": "FeatureCollection",
            "features": features,
        }
    )

    touched: set[str] = set()
    for f in deduped.get("features", []) or []:
        p = f.get("properties") or {}
        cid = p.get("cluster_id")
        if p.get("inter_cluster"):
            if isinstance(cid, str) and "|" in cid:
                for part in cid.split("|"):
                    touched.add(part)
        elif cid is not None:
            touched.add(str(cid))

    deduped["clusters_total"] = int(clusters_total)
    deduped["clusters_with_paths"] = int(len(touched))
    deduped["edges_total"] = len(deduped.get("features", []) or [])
    return deduped


def build_voronoi_edges_from_station_geojson(geo: dict) -> dict:
    """Строит локальные рёбра Вороного по точкам станций из ответа pipeline_result_to_geojson (по cluster_id)."""
    _require_voronoi()

    station_layers = [
        geo.get("charging_type_a") or {"type": "FeatureCollection", "features": []},
        geo.get("charging_type_b") or {"type": "FeatureCollection", "features": []},
        geo.get("garages") or {"type": "FeatureCollection", "features": []},
        geo.get("to_stations") or {"type": "FeatureCollection", "features": []},
    ]

    by_cluster: dict[str, list[list[float]]] = {}
    for layer in station_layers:
        for f in layer.get("features", []) or []:
            geom = f.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            props = f.get("properties") or {}
            cid = props.get("cluster_id")
            pt = [float(coords[0]), float(coords[1])]
            if cid is not None:
                key = str(cid)
                by_cluster.setdefault(key, []).append(pt)

    out_features: list[dict] = []
    for cluster_key, pts in by_cluster.items():
        if len(pts) < 2:
            continue
        pts_arr = np.asarray(pts, dtype=float)
        is_station = np.ones(len(pts_arr), dtype=bool)
        _append_voronoi_edges_for_cluster(
            out_features,
            pts_arr,
            is_station,
            cluster_key,
            "voronoi_station_local",
            group_mode="cluster_or_fallback",
            hull_poly=None,
        )

    return {"type": "FeatureCollection", "features": out_features}


def build_voronoi_edges_from_pipeline_raw(
    data_service: DataService,
    raw: dict,
    charging_station_features: list | None = None,
    buildings_per_centroid: int = 60,
) -> dict:
    """
    Быстрый Вороной без повторного data_service.get_city_data:
    используем demand из уже выполненного run_full_pipeline.
    """
    _require_voronoi()

    demand = raw.get("demand")
    if demand is None or len(demand) == 0:
        return {"type": "FeatureCollection", "features": []}
    d = demand.copy()
    if "is_cluster_fill" in d.columns:
        d = d[~d["is_cluster_fill"].fillna(False).astype(bool)].copy()
    if len(d) == 0:
        return {"type": "FeatureCollection", "features": []}

    if "cluster_id" not in d.columns:
        if "subcluster_id" in d.columns:
            d["cluster_id"] = d["subcluster_id"]
        else:
            return {"type": "FeatureCollection", "features": []}
    d = d[d["cluster_id"].notna()].copy()
    if len(d) == 0:
        return {"type": "FeatureCollection", "features": []}

    hulls_norm = _normalize_hulls_gdf(raw.get("demand_hulls"))

    grouped_points: dict[str, list[list[float]]] = {}
    for cluster_id, sub in d.groupby("cluster_id"):
        pts = []
        for _, row in sub.iterrows():
            g = row.get("geometry")
            if g is None or g.is_empty:
                continue
            try:
                pts.append([float(g.x), float(g.y)])
            except Exception:
                continue
        if pts:
            grouped_points[str(cluster_id)] = pts

    # Добавляем только здания, чей центроид попадает в hull соответствующего кластера (не по ближайшему центроиду).
    if not any(len(pts) >= 2 for pts in grouped_points.values()):
        buildings = raw.get("buildings")
        if (
            buildings is not None
            and len(buildings) > 0
            and hulls_norm is not None
            and len(hulls_norm) > 0
        ):
            try:
                b = buildings.to_crs("EPSG:4326").copy()
                try:
                    b = data_service._filter_buildings_for_demand(b)
                except Exception:
                    pass
                if b is not None and len(b) > 0:
                    b_proj = b.to_crs(epsg=3857)
                    b_centroids_wgs = gpd.GeoSeries(b_proj.geometry.centroid, crs=b_proj.crs).to_crs("EPSG:4326")
                    b_points = gpd.GeoDataFrame(
                        {"geometry": b_centroids_wgs},
                        index=b.index,
                        crs="EPSG:4326",
                    )
                    h_join = hulls_norm[["cluster_id", "geometry"]].copy()
                    try:
                        assigned = gpd.sjoin(
                            b_points,
                            h_join,
                            how="inner",
                            predicate="within",
                        )
                    except Exception:
                        assigned = gpd.sjoin(b_points, h_join, how="inner", op="within")
                    if assigned is not None and len(assigned) > 0:
                        assigned = assigned.copy()
                        if "index_left" in assigned.columns:
                            assigned["_bid"] = assigned["index_left"]
                        else:
                            assigned["_bid"] = assigned.index
                        assigned = assigned.drop_duplicates(subset=["_bid"])
                        gp: dict[str, list[list[float]]] = {}
                        for _, row in assigned.iterrows():
                            g = row.geometry
                            cid = row.get("cluster_id")
                            if g is None or g.is_empty or cid is None:
                                continue
                            try:
                                gp.setdefault(str(cid), []).append([float(g.x), float(g.y)])
                            except Exception:
                                continue
                        if gp:
                            grouped_points = gp
            except Exception:
                pass

    charging_station_features = _filter_station_features_to_hulls(
        charging_station_features,
        hulls_norm,
    )

    charging_by_cluster: dict[str, list[list[float]]] = {}
    if charging_station_features:
        for f in charging_station_features:
            geom = (f or {}).get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            props = (f or {}).get("properties") or {}
            cid = props.get("cluster_id")
            if cid is None:
                continue
            charging_by_cluster.setdefault(str(cid), []).append([float(coords[0]), float(coords[1])])

    target_group_size = max(1, int(buildings_per_centroid or 60))
    out_features: list[dict] = []
    for cluster_id, pts in grouped_points.items():
        if len(pts) < 2:
            continue

        pts_arr = np.asarray(pts, dtype=float)
        if len(pts_arr) > target_group_size:
            n_groups = int(np.ceil(len(pts_arr) / float(target_group_size)))
            n_groups = max(2, min(n_groups, len(pts_arr)))
            try:
                from sklearn.cluster import MiniBatchKMeans

                km = MiniBatchKMeans(
                    n_clusters=n_groups,
                    random_state=42,
                    batch_size=min(4096, max(1024, len(pts_arr) // 3)),
                    n_init=3,
                    max_iter=200,
                )
                labels = km.fit_predict(pts_arr)
                centers = []
                for k in range(n_groups):
                    members = pts_arr[labels == k]
                    if len(members) > 0:
                        centers.append(members.mean(axis=0))
                if len(centers) >= 2:
                    pts_arr = np.asarray(centers, dtype=float)
            except Exception:
                pass

        st_pts = charging_by_cluster.get(str(cluster_id), [])
        hull_poly = _hull_polygon_for_cluster(hulls_norm, cluster_id)
        if st_pts and hull_poly is not None:
            st_pts = [
                [float(x), float(y)]
                for x, y in st_pts
                if _point_in_hull(float(x), float(y), hull_poly)
            ]
        st_arr = np.asarray(st_pts, dtype=float) if st_pts else None
        pts_arr, is_station = merge_xy_with_station_flags(pts_arr, st_arr)

        if len(pts_arr) < 2:
            continue
        _append_voronoi_edges_for_cluster(
            out_features,
            pts_arr,
            is_station,
            cluster_id,
            "voronoi_local",
            hull_poly=hull_poly,
        )
    return {"type": "FeatureCollection", "features": out_features}
