"""Chat API endpoints — conversational Q&A with generative widgets."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.api.chat_schemas import ChatRequest, ChatResponse, PinnedWidget, WidgetSpec

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory store for pinned widgets.
# In production, replace with Postgres (see PinnedWidget schema).
_pinned_widgets: dict[str, dict[str, Any]] = {}


# ── Chat (SSE) ──────────────────────────────────────────────────


@router.post("/")
async def chat(request: Request, body: ChatRequest):
    """Chat endpoint — returns an SSE stream with text and optional widget.

    The stream emits JSON lines:
      {"type": "text", "content": "..."}
      {"type": "widget", "content": { ... WidgetSpec ... }}
      {"type": "done"}
    """
    user = getattr(request.state, "user", "unknown")
    logger.info(f"Chat request from {user}: {body.message[:100]}")

    history = [{"role": m.role.value, "content": m.content} for m in body.history]

    async def event_stream():
        try:
            from backend.api.chat_agent import run_agent_stream

            async for event in run_agent_stream(body.message, history=history):
                if event["type"] == "text_delta":
                    yield _sse_event("text", event["content"])
                elif event["type"] == "widget":
                    yield _sse_event("widget", event["content"])
                elif event["type"] == "done":
                    yield _sse_event("done", None)
        except Exception as e:
            logger.error(f"Chat agent error: {e}")
            yield _sse_event("text", f"Sorry, something went wrong: {e}")
            yield _sse_event("done", None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ask")
async def chat_ask(request: Request, body: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint — returns a single JSON response.

    Useful for simpler clients that don't support SSE.
    """
    user = getattr(request.state, "user", "unknown")
    logger.info(f"Chat ask from {user}: {body.message[:100]}")

    history = [{"role": m.role.value, "content": m.content} for m in body.history]

    try:
        from backend.api.chat_agent import run_agent

        result = await run_agent(body.message, history=history)
        widget = None
        if result.get("widget"):
            widget = WidgetSpec(**result["widget"])

        return ChatResponse(text=result.get("text"), widget=widget)
    except Exception as e:
        logger.error(f"Chat ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Widget Live Updates (SSE) ────────────────────────────────────


@router.get("/live")
async def live_metrics(query: str = Query(..., description="PromQL query for live updates")):
    """SSE endpoint that pushes Prometheus metric updates every 5 seconds.

    Used by the frontend to power real-time chart refreshes.
    """
    import asyncio

    from backend.api.chat_tools import query_prometheus_instant

    async def metric_stream():
        try:
            while True:
                result = await query_prometheus_instant(query)
                yield _sse_event("metric", result)
                await asyncio.sleep(5)
        except Exception as e:
            logger.warning(f"Live metric stream error: {e}")
            yield _sse_event("error", str(e))

    return StreamingResponse(
        metric_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Pinned Widgets ───────────────────────────────────────────────


@router.post("/pins")
async def pin_widget(request: Request, body: dict[str, Any]):
    """Pin a widget to the user's workspace.

    Body should contain: title, query (original question), widget (WidgetSpec).
    """
    user = getattr(request.state, "user", "unknown")
    pin_id = str(uuid.uuid4())[:8]

    pinned = PinnedWidget(
        id=pin_id,
        user=user,
        title=body.get("title", "Untitled"),
        query=body.get("query", ""),
        widget=WidgetSpec(**body["widget"]),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _pinned_widgets[pin_id] = pinned.model_dump()

    logger.info(f"User {user} pinned widget {pin_id}: {pinned.title}")
    return {"id": pin_id, "status": "pinned"}


@router.get("/pins")
async def list_pins(request: Request) -> list[dict[str, Any]]:
    """List all pinned widgets for the current user."""
    user = getattr(request.state, "user", "unknown")
    return [w for w in _pinned_widgets.values() if w["user"] == user]


@router.delete("/pins/{pin_id}")
async def unpin_widget(pin_id: str, request: Request):
    """Remove a pinned widget."""
    user = getattr(request.state, "user", "unknown")
    pin = _pinned_widgets.get(pin_id)
    if not pin or pin["user"] != user:
        raise HTTPException(status_code=404, detail="Pin not found")

    del _pinned_widgets[pin_id]
    return {"status": "unpinned"}


# ── Helpers ──────────────────────────────────────────────────────


def _sse_event(event_type: str, content: Any) -> str:
    """Format a Server-Sent Event line."""
    payload = {"type": event_type, "content": content}
    return f"data: {json.dumps(payload, default=str)}\n\n"
