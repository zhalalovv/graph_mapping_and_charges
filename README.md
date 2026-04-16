# Drone Planner (FastAPI + Leaflet UI)

Сервис для загрузки городских данных из OSM, кластеризации спроса, расчета расстановки инфраструктуры БПЛА и визуализации маршрутов по эшелонам.

## Что умеет

- Загружает городские данные (границы, дороги, здания, no-fly) из OpenStreetMap.
- Строит кластеры спроса (DBSCAN + hull-полигоны).
- Считает расстановку станций (A/B), гаражей и ТО.
- Строит маршрутные слои (Вороной, магистраль, ветки, локальные связи).
- Кэширует тяжелые данные в Redis.

## Кнопки в UI

В интерфейсе на главной странице две основные кнопки:

- `Загрузить`  
  Загружает **свежие данные** (принудительное обновление из OSM), затем отображает город/здания/кластеры.
- `Загрузить из БД`  
  Загружает данные из Redis/БД (без принудительного обновления OSM), а расстановку — из сохраненного Redis-кэша.

Примечание: в режиме `Загрузить из БД` кластеры на карте скрываются.

## Требования

- Python 3.10+
- Redis (рекомендуется; используется как основной кэш)

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Если есть проблемы с `numpy`:

```powershell
pip install "numpy<2.0.0" --force-reinstall
```

## Запуск

```powershell
python app.py
```

или:

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Откройте: `http://localhost:8000`

## Переменные окружения

- `REDIS_URL` — URL Redis (по умолчанию `redis://localhost:6379/0`)
- `PORT` — порт FastAPI при запуске через `python app.py` (по умолчанию `8000`)

## Основные API

### GET `/api/ping`
Проверка доступности сервиса.

### GET `/api/city/map`
Возвращает границы города, no-fly, bbox, центр карты.

Ключевые параметры:
- `city` (обязательно)
- `network_type` (по умолчанию `drive`)
- `simplify` (по умолчанию `true`)
- `force_refresh` (по умолчанию `false`) — принудительно обновляет данные из OSM

### GET `/api/buildings/export`
Здания с расчетными высотами и рекомендацией эшелона.

Ключевые параметры:
- `city`
- `network_type`
- `simplify`
- `force_refresh`

### GET `/api/buildings/clusters`
Кластеры спроса (DBSCAN) + cluster hulls.

Ключевые параметры:
- `city`
- `network_type`
- `simplify`
- `dbscan_eps_m`
- `dbscan_min_samples`
- `use_all_buildings`
- `force_refresh`

### GET `/api/stations/placement`
Полный расчет расстановки и маршрутных слоев, с поддержкой Redis-кэша.

Ключевые параметры:
- `city`
- `use_saved` — использовать сохраненный результат при наличии
- `only_saved` — только кэш Redis, без пересчета
- `save_result` — сохранять новый результат в Redis

### GET `/api/stations/export_saved`
Возвращает сохраненный JSON расстановки из Redis.

## Docker

Локальный запуск:

```powershell
docker build -t graph_service .
docker run --rm -p 8000:8000 -e REDIS_URL=redis://host.docker.internal:6379/0 graph_service
```

Через compose:

```powershell
docker compose up --build
```

## Структура проекта

```text
.
├── app.py                 # FastAPI API + HTML entrypoint
├── data_service.py        # OSM загрузка, нормализация, Redis-кэш
├── station_placement.py   # Расстановка станций/гаражей/ТО, графовые расчеты
├── voronoi_paths.py       # Локальные маршруты Вороного и фильтры
├── templates/
│   └── index.html         # UI (Leaflet)
├── static/
│   └── styles.css         # Стили
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
