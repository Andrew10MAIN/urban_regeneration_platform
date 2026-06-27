FROM python:3.12-slim

LABEL org.opencontainers.image.title="urban-platform-python"
LABEL org.opencontainers.image.description="Urban Regeneration Platform - Python ETL/ML base image"
LABEL python.version="3.12"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python", "--version"]
