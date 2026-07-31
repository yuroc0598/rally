"""Explicit interval policies shared by proposal ownership and output refinement."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

Segment = tuple[float, float]


def overlaps(left: Segment, right: Segment) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def events_in(segment: Segment, events: np.ndarray) -> np.ndarray:
    values = np.asarray(events, dtype=float)
    return values[(values >= segment[0] - 1e-9) & (values <= segment[1] + 1e-9)]


def contains_all_events(point: Segment, region: Segment, events: np.ndarray) -> bool:
    """Whether ``region`` owns every event assigned to ``point``."""
    values = events_in(point, events)
    if values.size:
        return bool(np.all((values >= region[0] - 1e-9)
                           & (values <= region[1] + 1e-9)))
    return point[0] >= region[0] - 1e-9 and point[1] <= region[1] + 1e-9


def covered_by_regions(point: Segment, regions: Sequence[Segment],
                       events: np.ndarray) -> bool:
    """Whether region cores collectively own every event assigned to ``point``."""
    values = events_in(point, events)
    if not values.size:
        return any(point[0] >= start - 1e-9 and point[1] <= end + 1e-9
                   for start, end in regions)
    return all(any(start - 1e-9 <= event <= end + 1e-9 for start, end in regions)
               for event in values)


def trim_previous_on_overlap(segments: Iterable[Segment]) -> list[Segment]:
    """Keep later starts authoritative by trimming the previous overlapping end."""
    output: list[Segment] = []
    for start, end in sorted((float(start), float(end)) for start, end in segments):
        if end <= start:
            continue
        if output:
            prior_start, prior_end = output[-1]
            if start < prior_end:
                if start > prior_start:
                    output[-1] = (prior_start, start)
                else:
                    output.pop()
        output.append((start, end))
    return output
