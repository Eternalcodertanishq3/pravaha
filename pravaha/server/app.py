"""Pravāha API Server — FastAPI Application Factory.

Production-grade inference server with:
- OpenAI-compatible API endpoints
- SSE token streaming
- CORS, Request ID, and Timing middleware
- Lifespan-managed engine (auto-start, graceful shutdown)

Usage:
    uvicorn pravaha.server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure pravaha is importable when running as module
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pravaha.engine_async import AsyncPravahaEngine
from pravaha.server.middleware import (
    ErrorHandlerMiddleware,
    RequestIDMiddleware,
    TimingMiddleware,
)
from pravaha.server.routes_chat import router as chat_router
from pravaha.server.routes_completions import router as completions_router
from pravaha.server.routes_health import router as health_router
from pravaha.server.routes_models import router as models_router

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONFIG_PATH = "configs/default.yaml"


# ─────────────────────────────────────────────
# Lifespan (startup/shutdown)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine on startup, gracefully stop on shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("  Pravāha Inference Server — Starting Up")
    logger.info("=" * 60)

    # Initialize the engine
    engine = AsyncPravahaEngine(config_path=CONFIG_PATH)
    app.state.engine = engine

    model_name = engine.config.model.model_path
    quantization = engine.config.model.quantization or "none"
    logger.info(f"  Model:        {model_name}")
    logger.info(f"  Quantization: {quantization}")
    logger.info(f"  Device:       {engine._device}")
    logger.info("=" * 60)
    logger.info("  Server ready. Accepting requests.")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down Pravāha engine...")
    engine.stop()
    logger.info("Engine stopped. Goodbye.")


# ─────────────────────────────────────────────
# Application Factory
# ─────────────────────────────────────────────

app = FastAPI(
    title="Pravāha Inference Server",
    description=(
        "A vLLM-inspired LLM inference engine with continuous batching, "
        "PagedAttention, and INT4/INT8 quantization. "
        "Provides an OpenAI-compatible API for text and chat completions."
    ),
    version="0.6.0",
    lifespan=lifespan,
)

# ── Middleware Stack (applied in reverse order) ──
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──
app.include_router(completions_router, prefix="/v1")
app.include_router(chat_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(health_router)


# ── Root endpoint ──
@app.get("/")
async def root():
    """Server information."""
    return {
        "name": "Pravāha Inference Server",
        "version": "0.6.0",
        "docs": "/docs",
        "endpoints": [
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/models",
            "/health",
            "/health/ready",
            "/metrics",
        ],
    }
