FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY mats ./mats
COPY scripts ./scripts
COPY config ./config
COPY perchance ./perchance
COPY vault ./vault

RUN pip install -U pip && pip install .

EXPOSE 8787 8788 9010 9011 9012 9013 9014 9015 9016 9017 9018 9019
CMD ["python", "-m", "mats.relay.fastapi_relay"]
