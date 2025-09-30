# Базовый образ с micromamba для удобной установки geo-стека
FROM mambaorg/micromamba:1.5.8

USER root
WORKDIR /app

# Системные зависимости для osmnx/geopandas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      gcc \
      g++ \
      libgeos-dev \
      libproj-dev \
      libspatialindex-dev \
      libspatialite-dev \
      gdal-bin \
      libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

# Создаём окружение с нужными пакетами через micromamba (conda-forge)
COPY --chown=micromamba:micromamba requirements.txt /app/requirements.txt

RUN micromamba install -y -n base -c conda-forge \
      python=3.11 \
      pip \
      geopandas \
      osmnx \
      geopy \
      gdal \
      proj \
      rtree \
      shapely && \
    micromamba clean --all -y

# Затем доустанавливаем pip-зависимости (fastapi/uvicorn/redis и т.п.)
RUN micromamba run -n base pip install --no-cache-dir -r /app/requirements.txt

# Копируем код
COPY . /app

ENV PORT=8000
EXPOSE 8000

# Запуск uvicorn через micromamba (base)
CMD ["micromamba", "run", "-n", "base", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


