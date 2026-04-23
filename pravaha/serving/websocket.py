"""WebSocket streaming endpoint for real-time token delivery."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pravaha.decoder.sampling import SamplingParams

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket) -> None:
    """Stream tokens over WebSocket connection."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            engine = websocket.app.state.engine
            params = SamplingParams(
                temperature=data.get("temperature", 0.7),
                max_new_tokens=data.get("max_tokens", 256),
            )
            async for token in engine.generate(prompt, params):
                await websocket.send_json({"token": token, "done": False})
            await websocket.send_json({"token": "", "done": True})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))
