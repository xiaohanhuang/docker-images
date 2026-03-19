"""Pydantic models for the Generative UI chat system."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field  # noqa: I001

# ── Widget Spec ──────────────────────────────────────────────────


class WidgetType(str, Enum):
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    PIE = "pie"
    STAT = "stat"
    TABLE = "table"


class WidgetSeries(BaseModel):
    """A single data series within a chart widget."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="Legend label for this series")
    data_key: str = Field(description="Key in the data array for Y-axis values", alias="dataKey")
    color: str | None = Field(None, description="Hex color, e.g. '#8884d8'")


class WidgetSpec(BaseModel):
    """Config-driven widget specification returned by the LLM agent."""

    model_config = ConfigDict(populate_by_name=True)

    type: WidgetType
    title: str
    description: str | None = None

    # Chart data (for line/bar/area/pie)
    data: list[dict[str, Any]] = Field(default_factory=list)
    x_axis_key: str | None = Field(
        None, description="Key in data for X-axis (time, label)", alias="xAxisKey"
    )
    series: list[WidgetSeries] = Field(default_factory=list)

    # Stat widget
    value: str | None = Field(None, description="Main value for stat widget")
    unit: str | None = None
    trend: float | None = Field(None, description="Percentage change for stat widget")

    # Table widget
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)

    # Live updates
    live: bool = Field(False, description="Enable SSE live updates for this widget")
    refresh_query: str | None = Field(
        None, description="PromQL query for live refresh", alias="refreshQuery"
    )


# ── Chat Messages ────────────────────────────────────────────────


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    message: str = Field(description="The user's natural language question")
    history: list[ChatMessage] = Field(
        default_factory=list, description="Previous conversation turns"
    )


class ChatResponse(BaseModel):
    """A single response chunk — may contain text, a widget, or both."""

    text: str | None = None
    widget: WidgetSpec | None = None


# ── Pinned Widget ────────────────────────────────────────────────


class PinnedWidget(BaseModel):
    id: str
    user: str
    title: str
    query: str = Field(description="The original natural-language question")
    widget: WidgetSpec
    created_at: str
