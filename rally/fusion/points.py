"""Final point-list utilities shared by analysis and publishing."""

from __future__ import annotations

Segment = tuple[float, float]


def total_kept_seconds(segments: list[Segment]) -> float:
    """Return covered source time after coalescing accidental overlaps."""
    merged: list[list[float]] = []
    for start, end in sorted((float(start), float(end)) for start, end in segments):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return float(sum(end - start for start, end in merged))
