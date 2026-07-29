"""Evaluate rally segments against an independently annotated JSON file.

This deliberately lives outside the detector: the evaluator consumes only intervals and
does not import scoring/decoding code, so implementation assumptions cannot leak into the
oracle. Gold files may be a sidecar-shaped object with ``segments`` or a plain list of
``{"start": ..., "end": ...}`` / ``[start, end]`` records.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def _segments(value) -> list[tuple[float, float]]:
    if isinstance(value, dict):
        value = value.get("segments", [])
    if not isinstance(value, list):
        raise ValueError("segments must be a list or an object containing a segments list")
    out: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict):
            start, end = item.get("start"), item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            raise ValueError(f"invalid segment record: {item!r}")
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"invalid segment bounds: {item!r}")
        out.append((start, end))
    out.sort()
    return out


def load_segments(path: str | Path) -> list[tuple[float, float]]:
    with Path(path).open() as fh:
        return _segments(json.load(fh))


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / union if union > 0 else 0.0


def _union_length(intervals: Iterable[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return float(sum(end - start for start, end in merged))


def evaluate_segments(predicted, gold, *, min_iou: float = 0.3) -> dict:
    """One-to-one interval matching plus boundary and retained-dead-time metrics."""
    pred = _segments(predicted)
    truth = _segments(gold)
    if not 0 <= min_iou <= 1:
        raise ValueError("min_iou must be in [0, 1]")

    matrix = np.zeros((len(pred), len(truth)), dtype=float)
    for i, p in enumerate(pred):
        for j, g in enumerate(truth):
            matrix[i, j] = _iou(p, g)
    matched: list[tuple[int, int, float]] = []
    if matrix.size:
        # Maximize the number of threshold-valid matches first, then their total IoU.
        # Maximizing raw IoU and filtering afterward can sacrifice a valid edge for two
        # subthreshold edges and undercount true positives.
        valid = matrix >= min_iou
        cardinality_bonus = max(matrix.shape) + 1.0
        objective = np.where(valid, cardinality_bonus + matrix, 0.0)
        rows, cols = linear_sum_assignment(-objective)
        matched = [(int(i), int(j), float(matrix[i, j])) for i, j in zip(rows, cols)
                   if matrix[i, j] >= min_iou]

    pred_used = {i for i, _, _ in matched}
    gold_used = {j for _, j, _ in matched}
    tp, fp, fn = len(matched), len(pred) - len(matched), len(truth) - len(matched)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not truth else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    start_err = [abs(pred[i][0] - truth[j][0]) for i, j, _ in matched]
    end_err = [abs(pred[i][1] - truth[j][1]) for i, j, _ in matched]
    overlap_intervals = [
        (max(p[0], g[0]), min(p[1], g[1]))
        for p in pred for g in truth if max(p[0], g[0]) < min(p[1], g[1])
    ]
    retained_dead = max(0.0, _union_length(pred) - _union_length(overlap_intervals))

    def stats(values: list[float]) -> dict:
        return {
            "mean_s": float(np.mean(values)) if values else None,
            "median_s": float(np.median(values)) if values else None,
            "max_s": float(np.max(values)) if values else None,
        }

    return {
        "min_iou": min_iou,
        "counts": {"predicted": len(pred), "gold": len(truth),
                   "true_positive": tp, "false_positive": fp, "false_negative": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "start_error": stats(start_err),
        "end_error": stats(end_err),
        "retained_dead_seconds": retained_dead,
        "matches": [
            {"predicted_index": i, "gold_index": j, "iou": iou,
             "start_error_s": abs(pred[i][0] - truth[j][0]),
             "end_error_s": abs(pred[i][1] - truth[j][1])}
            for i, j, iou in matched
        ],
        "false_positive_indices": [i for i in range(len(pred)) if i not in pred_used],
        "missed_gold_indices": [j for j in range(len(truth)) if j not in gold_used],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare predicted rally segments with independently annotated gold JSON")
    parser.add_argument("predicted")
    parser.add_argument("gold")
    parser.add_argument("--min-iou", type=float, default=0.3)
    parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)
    result = evaluate_segments(load_segments(args.predicted), load_segments(args.gold),
                               min_iou=args.min_iou)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
