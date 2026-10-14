from __future__ import annotations

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.chat import router as chat_router
from app.api.routes.eval import router as eval_router
from app.api.routes.health import router as health_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.observability import log_event
from app.runtime.bootstrap import build_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=settings.LOG_LEVEL)

    log_event("application.starting", environment=(settings.APP_ENV))

    runtime = await build_runtime()

    app.state.runtime = runtime

    log_event("application.started", environment=(settings.APP_ENV))

    try:
        yield

    finally:
        log_event("application.stopping")

        await runtime.close()

        app.state.runtime = None

        log_event("application.stopped")


app = FastAPI(
    title=("eplanet Brokers " "AI Assistant"), version="0.1.0", lifespan=lifespan
)


app.include_router(health_router)

app.include_router(chat_router)

app.include_router(eval_router)

_DEMO_DIR = Path(__file__).resolve().parent / "web" / "demo"

app.mount("/demo", StaticFiles(directory=_DEMO_DIR, html=True), name="demo")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_event(
        "api.unhandled_exception",
        level=40,
        error_type=(exc.__class__.__name__),
        path=(request.url.path),
        method=(request.method),
    )

    return JSONResponse(status_code=500, content={"detail": ("Internal server error.")})
