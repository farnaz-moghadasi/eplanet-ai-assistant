import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Async (uvicorn) workers handle many concurrent requests per process via
# the event loop, unlike sync workers - so this app doesn't need the
# classic (2 x cores) + 1 sync formula. One worker per core is a
# reasonable default; override with WEB_CONCURRENCY per environment.
workers = int(
    os.environ.get(
        "WEB_CONCURRENCY",
        multiprocessing.cpu_count(),
    )
)

worker_class = "uvicorn.workers.UvicornWorker"

# Generous timeout: a single chat turn can involve several sequential
# LLM calls (understanding, reranking, generation, grounding) and has
# been observed to take 10-30+ seconds end to end.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

accesslog = "-"
errorlog = "-"
