Graph Service (FastAPI + UI)

Установка

1) Создайте и активируйте виртуальное окружение (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Установите зависимости:
```powershell
pip install -r requirements.txt
```

Запуск

```powershell
python app.py
```
или (через uvicorn):
```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Откройте `http://localhost:8000` в браузере.

Эндпоинты

- GET /api/ping — проверка доступности
- GET /api/city?city=Volgograd, Russia — загрузка данных (кэш/OSM)
- GET /api/graph?city=Volgograd, Russia — GeoJSON рёбер для отрисовки

Примечания

- Для ускорения повторных запросов можно запустить локальный Redis (`REDIS_URL`), иначе используется файловый кэш в `cache_data/`.
- Пакеты `osmnx` и `geopandas` могут требовать системные зависимости (GEOS/GDAL/Proj). На Windows проще установить через conda, если возникнут трудности с pip.

Docker

Собрать и запустить локально (только приложение):
```powershell
docker build -t graph_service .
docker run --rm -p 8000:8000 -e REDIS_URL=redis://host.docker.internal:6379/0 graph_service
```

Docker Compose (с Redis)
```powershell
docker compose up --build
```
Затем откройте: `http://localhost:8000`.


