"""FastAPI app factory for Pravaha v3 with all routes and middleware."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Engine lifecycle — load on startup, shutdown gracefully."""
    logger.info("Pravaha v3 starting up...")
    try:
        from pravaha.config.engine_config import EngineConfig
        from pravaha.engine.async_engine import AsyncPravahaEngine

        config = EngineConfig.default()
        engine = AsyncPravahaEngine(config=config)
        app.state.engine = engine
        logger.info("Engine ready.")
    except Exception as e:
        logger.warning(f"Engine init deferred: {e}")
        app.state.engine = None
    yield
    if hasattr(app.state, "engine") and app.state.engine:
        app.state.engine.stop()
        logger.info("Engine stopped.")


def create_app() -> FastAPI:
    """Build the FastAPI application with all routes and middleware."""
    app = FastAPI(
        title="Pravaha v3",
        description="Self-healing, swarm-ready LLM inference engine",
        version="3.0.0",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    from pravaha.serving.middleware import (
        ErrorHandlerMiddleware,
        RequestIDMiddleware,
        TimingMiddleware,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    # Routes
    from pravaha.serving.routes.admin import router as admin_router
    from pravaha.serving.routes.branches import router as branches_router
    from pravaha.serving.routes.chat import router as chat_router
    from pravaha.serving.routes.completions import router as completions_router
    from pravaha.serving.routes.debug import router as debug_router
    from pravaha.serving.routes.health import router as health_router
    from pravaha.serving.routes.metrics import router as metrics_router
    from pravaha.serving.routes.models import router as models_router
    from pravaha.serving.routes.rag import router as rag_router
    from pravaha.serving.routes.swarm import router as swarm_router
    from pravaha.serving.routes.vision import router as vision_router
    from pravaha.serving.routes.world import router as world_router
    from pravaha.serving.websocket import router as ws_router

    app.include_router(completions_router, prefix="/v1")
    app.include_router(chat_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(swarm_router, prefix="/v1")
    app.include_router(rag_router, prefix="/v1")
    app.include_router(vision_router, prefix="/v1")
    app.include_router(branches_router, prefix="/v1")
    app.include_router(debug_router, prefix="/v1")
    app.include_router(admin_router)
    app.include_router(world_router)
    app.include_router(ws_router)

    return app
