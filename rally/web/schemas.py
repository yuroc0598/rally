"""Validated request contracts for the web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

MAX_LABEL_ITEMS = 50
MAX_EDIT_SEGMENTS = 100


class SegmentEdit(BaseModel):
    segments: list[list[float]] = Field(max_length=MAX_EDIT_SEGMENTS)


class LabelTaskRequest(BaseModel):
    kinds: list[str] = Field(
        default=["player_identity", "serve_motion"], min_length=1, max_length=2)
    max_items: int = Field(default=10, ge=1, le=MAX_LABEL_ITEMS)
    match_type: str = "auto"
    regenerate: bool = False


class LabelPayload(BaseModel):
    revision: str = Field(min_length=1, max_length=100)
    task_id: str = Field(min_length=1, max_length=100)
    kind: str = Field(min_length=1, max_length=40)
    values: dict[str, Any] = Field(default_factory=dict, max_length=20)


class RosterUpdate(BaseModel):
    revision: str = Field(min_length=1, max_length=100)
    roster: list[dict[str, Any]] = Field(max_length=20)
