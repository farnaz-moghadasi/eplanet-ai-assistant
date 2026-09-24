FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY gunicorn.conf.py ./

EXPOSE 8000

# Config (workers, timeouts, LLM keys, REDIS_URL, QDRANT_URL, ...) is
# supplied at runtime via environment variables - never baked into the
# image. See gunicorn.conf.py and app/core/config.py.
CMD ["gunicorn", "app.main:app", "-c", "gunicorn.conf.py"]
