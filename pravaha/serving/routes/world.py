from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
from pravaha.engine.event_bus import get_event_bus
import os
import json

router = APIRouter()

WORLD_HTML_PATH = os.path.join(os.path.dirname(__file__), "..", "world.html")

@router.get("/world", response_class=HTMLResponse)
async def get_world():
    with open(WORLD_HTML_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

async def event_generator(request: Request):
    queue = asyncio.Queue()
    bus = get_event_bus()

    def _event_listener(event):
        try:
            queue.put_nowait(event)
        except Exception:
            pass

    bus.subscribe_all(_event_listener)

    try:
        while True:
            if await request.is_disconnected():
                break
            
            # Use asyncio.wait_for to periodically check for disconnects
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                data = json.dumps({
                    "event_type": event.event_type,
                    "request_id": event.request_id,
                    "data": event.data
                }, default=str)
                yield f"data: {data}\n\n"
            except asyncio.TimeoutError:
                continue
    finally:
        bus.unsubscribe_all(_event_listener)

@router.get("/world/events")
async def world_events(request: Request):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")
