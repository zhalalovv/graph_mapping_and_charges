import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from data_service import DataService
import json
import osmnx as ox


app = FastAPI(title="Graph Service UI", version="0.1.0")

# CORS (на случай локального фронтенда)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_service = DataService()


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>UI не найден. Создайте templates/index.html</h1>", status_code=200)


@app.get("/api/ping")
def ping():
    return {"status": "ok"}


@app.get("/api/city")
def get_city(
    city: str = Query(..., description="Название города, например 'Volgograd, Russia'"),
    network_type: str = Query('drive', description="Тип сети OSM: drive, walk, bike, all, all_private"),
    simplify: bool = Query(True, description="Упрощать ли граф"),
):
    try:
        progress_events = []

        def _progress(stage: str, percentage: int, message: str = ""):
            progress_events.append({
                "stage": stage,
                "percentage": percentage,
                "message": message,
            })

        data_service.add_progress_callback(_progress)
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)

        # Для передачи в JSON: убираем тяжёлые объекты и возвращаем краткую сводку
        stats = data.get("stats", {})
        result = {
            "city_name": data.get("city_name"),
            "stats": stats,
            "params": data.get("params", {}),
            "progress": progress_events,
        }
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph")
def graph_geojson(
    city: str = Query(..., description="Название города, как в /api/city"),
    network_type: str = Query('drive', description="Тип сети OSM"),
    simplify: bool = Query(True, description="Упрощение графа"),
):
    try:
        # Загружаем (или берём из кэша) данные города
        data = data_service.get_city_data(city, network_type=network_type, simplify=simplify)
        graph = data.get("road_graph")
        if graph is None:
            raise HTTPException(status_code=404, detail="Граф не найден")

        # Преобразуем граф в GeoDataFrame и далее в GeoJSON
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)
        xmin, ymin, xmax, ymax = gdf_nodes.total_bounds  # lon/lat
        center_lat = (ymin + ymax) / 2.0
        center_lon = (xmin + xmax) / 2.0

        edges_geojson = json.loads(gdf_edges.to_json())

        return JSONResponse({
            "bbox": [xmin, ymin, xmax, ymax],
            "center": {"lat": center_lat, "lon": center_lon},
            "edges": edges_geojson,
            "stats": data.get("stats", {}),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _ensure_static_mount():
    static_dir = os.path.join(os.getcwd(), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")


_ensure_static_mount()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)


