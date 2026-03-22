import json
import logging
import os

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from data_service import DataService
from station_placement import pipeline_result_to_geojson, run_full_pipeline

app = FastAPI(title="Clustering", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_service = DataService()
logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Нет templates/index.html</h1>", status_code=404)


@app.get("/api/ping")
def ping():
    return {"status": "ok"}


@app.get("/api/city/map")
def get_city_map(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение"),
):
    """Границы города, no_fly_zones, bbox/center для карты."""
    try:
        data = data_service.get_city_data(
            city, network_type=network_type, simplify=simplify, load_no_fly_zones=True
        )
        city_boundary = data.get("city_boundary")
        road_graph = data.get("road_graph")
        no_fly_zones = data.get("no_fly_zones")
        no_fly_zones_geojson = None
        if no_fly_zones is not None:
            try:
                if isinstance(no_fly_zones, gpd.GeoDataFrame) and len(no_fly_zones) > 0:
                    no_fly_zones_geojson = json.loads(no_fly_zones.to_json())
                elif isinstance(no_fly_zones, list) and len(no_fly_zones) > 0:
                    zones_gdf = gpd.GeoDataFrame([{"geometry": z} for z in no_fly_zones], crs="EPSG:4326")
                    no_fly_zones_geojson = json.loads(zones_gdf.to_json())
            except Exception as e:
                logger.warning("Ошибка сериализации no_fly_zones: %s", e)
        if city_boundary is not None:
            boundary_gdf = gpd.GeoDataFrame([{"geometry": city_boundary}], crs="EPSG:4326")
            city_boundary_geojson = json.loads(boundary_gdf.to_json())
            bbox = list(city_boundary.bounds)
            center_lon = (bbox[0] + bbox[2]) / 2.0
            center_lat = (bbox[1] + bbox[3]) / 2.0
        elif road_graph is not None and len(road_graph.nodes) > 0:
            import osmnx as ox

            gdf_nodes, _ = ox.graph_to_gdfs(road_graph)
            xmin, ymin, xmax, ymax = gdf_nodes.total_bounds
            bbox = [xmin, ymin, xmax, ymax]
            center_lat = (ymin + ymax) / 2.0
            center_lon = (xmin + xmax) / 2.0
            city_boundary_geojson = None
        else:
            raise HTTPException(status_code=404, detail="Нет данных по городу")
        return JSONResponse(
            {
                "city_boundary": city_boundary_geojson,
                "no_fly_zones": no_fly_zones_geojson,
                "bbox": bbox,
                "center": {"lat": center_lat, "lon": center_lon},
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/buildings/export")
def export_buildings_with_heights(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение"),
):
    """Здания с высотами для подложки карты."""
    try:
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
        buildings = data.get("buildings")
        flight_levels = data.get("flight_levels", [])
        if buildings is None or len(buildings) == 0:
            return JSONResponse(
                {
                    "type": "FeatureCollection",
                    "features": [],
                    "flight_levels": flight_levels,
                    "building_height_stats": data.get("building_height_stats", {}),
                }
            )

        if "height_m" not in buildings.columns:
            buildings = data_service._compute_building_heights(buildings)

        if "area_m2" not in buildings.columns:
            try:
                b_proj = buildings
                if b_proj.crs is None:
                    b_proj = b_proj.set_crs("EPSG:4326", allow_override=True)
                if b_proj.crs and b_proj.crs.is_geographic:
                    b_proj = b_proj.to_crs(epsg=3857)
                buildings = buildings.copy()
                buildings["area_m2"] = b_proj.geometry.area.astype(float)
            except Exception:
                try:
                    buildings = buildings.copy()
                    buildings["area_m2"] = buildings.geometry.area.astype(float)
                except Exception:
                    buildings = buildings.copy()
                    buildings["area_m2"] = None

        levels_m = [f["altitude_m"] for f in flight_levels] if flight_levels else [40, 65, 90, 115]
        features = []
        for idx, row in buildings.iterrows():
            geom = row.geometry
            if geom is None or not getattr(geom, "is_valid", True):
                continue
            h = float(row.get("height_m", 10))
            a_raw = row.get("area_m2", None)
            try:
                a = float(a_raw) if a_raw is not None else None
            except (TypeError, ValueError):
                a = None
            rec_level = 1
            for i, alt in enumerate(levels_m):
                if alt >= h + 10:
                    rec_level = i + 1
                    break
            feat = {
                "type": "Feature",
                "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
                "properties": {
                    "height_m": round(h, 1),
                    "area_m2": round(a, 1) if a is not None else None,
                    "flight_level": rec_level,
                },
            }
            features.append(feat)
        return JSONResponse(
            {
                "type": "FeatureCollection",
                "features": features,
                "flight_levels": flight_levels,
                "building_height_stats": data.get("building_height_stats", {}),
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/buildings/clusters")
def get_building_clusters(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
    method: str = Query("dbscan", description="Только dbscan (кластеризация по зданиям)"),
    dbscan_eps_m: float = Query(180.0, description="Радиус локального уровня (м)"),
    dbscan_min_samples: int = Query(15, description="min_samples DBSCAN локального уровня"),
    use_all_buildings: bool = Query(False, description="Все здания вне no-fly"),
):
    """Кластеры спроса (двухуровневая кластеризация в data_service) + полигоны hull."""

    def round_coords(obj, precision=6):
        if isinstance(obj, dict):
            return {k: round_coords(v, precision) for k, v in obj.items()}
        if isinstance(obj, list):
            return [round_coords(item, precision) for item in obj]
        if isinstance(obj, float):
            return round(obj, precision)
        return obj

    if method != "dbscan":
        raise HTTPException(status_code=400, detail="Поддерживается только method=dbscan")

    try:
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
        buildings = data.get("buildings")
        road_graph = data.get("road_graph")
        city_boundary = data.get("city_boundary")
        if buildings is None or len(buildings) == 0:
            return JSONResponse(
                {
                    "buildings": {"type": "FeatureCollection", "features": []},
                    "clusters": {"type": "FeatureCollection", "features": []},
                    "cluster_hulls": {"type": "FeatureCollection", "features": []},
                    "total": 0,
                    "message": "Нет зданий",
                }
            )

        demand_gdf = data.get("demand_points")
        hulls_gdf = data.get("demand_hulls")
        if demand_gdf is None or len(demand_gdf) == 0:
            no_fly_zones = data.get("no_fly_zones")
            result = data_service.get_demand_points_weighted(
                buildings,
                road_graph,
                city_boundary,
                method="dbscan",
                dbscan_eps_m=dbscan_eps_m,
                dbscan_min_samples=dbscan_min_samples,
                return_hulls=True,
                use_all_buildings=use_all_buildings,
                no_fly_zones=no_fly_zones,
            )
            demand_gdf, hulls_gdf = result if isinstance(result, tuple) else (result, None)

        if demand_gdf is None or len(demand_gdf) == 0:
            clusters_fc = {"type": "FeatureCollection", "features": []}
            cluster_hulls_fc = {"type": "FeatureCollection", "features": []}
        else:
            features = []
            source_gdf = hulls_gdf if (hulls_gdf is not None and len(hulls_gdf) > 0) else demand_gdf
            for idx, row in source_gdf.iterrows():
                geom = row.get("geometry")
                if geom is None or geom.is_empty:
                    continue
                try:
                    c = getattr(geom, "centroid", None) or geom
                    lon, lat = float(c.x), float(c.y)
                except Exception:
                    continue
                weight = int(row.get("weight", 1))
                cid = row.get("cluster_id")
                if cid is None or (isinstance(cid, float) and pd.isna(cid)):
                    cid = idx
                try:
                    cid = int(cid)
                except (TypeError, ValueError):
                    cid = hash(str(idx)) % 10_000_000
                props = {"weight": weight, "cluster_id": cid}
                nb = row.get("n_buildings")
                if nb is None or (isinstance(nb, float) and pd.isna(nb)):
                    nb = 1
                props["n_buildings"] = int(nb)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                        "properties": props,
                    }
                )
            clusters_fc = round_coords({"type": "FeatureCollection", "features": features}, precision=6)
            cluster_hulls_fc = {"type": "FeatureCollection", "features": []}
            if hulls_gdf is not None and len(hulls_gdf) > 0:
                hull_features = []
                for idx, row in hulls_gdf.iterrows():
                    g = row["geometry"]
                    if g is None or g.is_empty:
                        continue
                    try:
                        geom_json = json.loads(gpd.GeoSeries([g], crs="EPSG:4326").to_json())
                        geom_j = geom_json.get("features", [{}])[0].get("geometry")
                        if geom_j:
                            w = int(row.get("weight", 1))
                            _cid = row.get("cluster_id")
                            if _cid is None or (isinstance(_cid, float) and pd.isna(_cid)):
                                _cid = idx
                            try:
                                _cid = int(_cid)
                            except (TypeError, ValueError):
                                _cid = hash(str(idx)) % 10_000_000
                            props = {"weight": w, "cluster_id": _cid}
                            nb = row.get("n_buildings")
                            if nb is None or (isinstance(nb, float) and pd.isna(nb)):
                                nb = 1
                            props["n_buildings"] = int(nb)
                            hull_features.append(
                                {
                                    "type": "Feature",
                                    "geometry": round_coords(geom_j, precision=6),
                                    "properties": props,
                                }
                            )
                    except Exception:
                        pass
                cluster_hulls_fc = {"type": "FeatureCollection", "features": hull_features}

        buildings_fc = clusters_fc
        total = len(clusters_fc.get("features", []))
        logger.info("Кластеры: %s, hulls: %s", total, len(cluster_hulls_fc.get("features", [])))

        return JSONResponse(
            {
                "buildings": buildings_fc,
                "clusters": clusters_fc,
                "total": total,
                "method": method,
                "cluster_hulls": cluster_hulls_fc,
                "dbscan_eps_m": dbscan_eps_m,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stations/placement")
def get_stations_placement(
    city: str = Query(..., description="Название города"),
    network_type: str = Query("drive", description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
    demand_method: str = Query("dbscan", description="dbscan или grid"),
    dbscan_eps_m: float = Query(180.0, description="eps DBSCAN (м)"),
    dbscan_min_samples: int = Query(15, description="min_samples DBSCAN"),
):
    """
    Размещение зарядок (тип А/Б), гаражей и ТО по правилам R_charge / R_garage_TO;
    магистраль, ветви B→A и локальные B↔B. Слои разделены в ответе.
    """
    try:
        raw = run_full_pipeline(
            data_service,
            city,
            network_type=network_type,
            simplify=simplify,
            demand_method=demand_method,
            dbscan_eps_m=dbscan_eps_m,
            dbscan_min_samples=dbscan_min_samples,
        )
        if raw.get("error"):
            return JSONResponse({**pipeline_result_to_geojson(raw), "error": raw["error"]})
        geo = pipeline_result_to_geojson(raw)
        return JSONResponse(geo)
    except Exception as e:
        logger.exception("stations placement")
        raise HTTPException(status_code=500, detail=str(e))


def _ensure_static_mount():
    static_dir = os.path.join(os.getcwd(), "static")
    if os.path.isdir(static_dir):
        from fastapi.staticfiles import StaticFiles

        app.mount("/static", StaticFiles(directory=static_dir), name="static")


_ensure_static_mount()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
