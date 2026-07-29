"""Golden accuracy checks for the independently annotated sample match.

The real-video check is intentionally opt-in because the default pipeline runs YOLO,
TrackNet, and serve validation over a 425 MB video. Run it end to end with::

    RALLY_RUN_GOLDEN=1 pytest -q tests/test_golden_rallies.py

To evaluate an already-produced pipeline sidecar without repeating inference, use::

    RALLY_GOLDEN_SIDECAR_1=/path/to/rallies-1.json \
    RALLY_GOLDEN_SIDECAR_2=/path/to/rallies-2.json \
        pytest -q tests/test_golden_rallies.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_TOLERANCE_S = 2.0
TRACKNET_SHA256 = "c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076"

_POINT_LINE = re.compile(r"^\s*Point\s+(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_TIMESTAMP = re.compile(r"(?<![\d:])(\d+(?::[0-5]?\d)?)(?![\d:])")


@dataclass(frozen=True)
class GoldenPoint:
    acceptable_starts: tuple[float, ...]
    end: float


@dataclass(frozen=True)
class GoldenDataset:
    number: int
    video: Path
    annotation: Path
    video_sha256: str
    expected: tuple[GoldenPoint, ...]

    @property
    def sidecar_environment_variable(self) -> str:
        return f"RALLY_GOLDEN_SIDECAR_{self.number}"


GOLDEN_DATASETS = (
    GoldenDataset(
        number=1,
        video=ROOT / "samples" / "test_input_1.mp4",
        annotation=ROOT / "samples" / "test_output_rally_1.txt",
        video_sha256=(
            "4e443ad9d52d7ed54e241534176216634edf4b0f9e51d2fa1faf1dd18fd4221a"
        ),
        expected=(
            GoldenPoint((8.0,), 15.0),
            GoldenPoint((29.0,), 35.0),
            GoldenPoint((45.0,), 71.0),
            GoldenPoint((91.0, 105.0), 112.0),
            GoldenPoint((163.0,), 174.0),
            GoldenPoint((186.0,), 195.0),
            GoldenPoint((202.0,), 211.0),
            GoldenPoint((224.0, 232.0), 238.0),
            GoldenPoint((259.0, 275.0), 282.0),
        ),
    ),
    GoldenDataset(
        number=2,
        video=ROOT / "samples" / "test_input_2.mp4",
        annotation=ROOT / "samples" / "test_output_rally_2.txt",
        video_sha256=(
            "a1b1b1e7046edfed2f624d37c320f2f88a3d71b98fbc3761709611654f2182d7"
        ),
        expected=(
            GoldenPoint((14.0,), 22.0),
            GoldenPoint((35.0, 42.0), 48.0),
            GoldenPoint((64.0,), 70.0),
            GoldenPoint((85.0,), 90.0),
            GoldenPoint((155.0,), 172.0),
            GoldenPoint((184.0,), 196.0),
            GoldenPoint((203.0,), 217.0),
            GoldenPoint((229.0,), 244.0),
            GoldenPoint((271.0,), 282.0),
        ),
    ),
)


def _seconds(timestamp: str) -> float:
    if ":" not in timestamp:
        return float(timestamp)
    minutes, seconds = timestamp.split(":", 1)
    return 60.0 * float(minutes) + float(seconds)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotated_points(path: Path) -> list[GoldenPoint]:
    """Parse each point and any explicitly annotated retry-serve start.

    When a point includes a fault/let followed by another serve, one prediction may start
    at either the original serve or the retry, but it must remain one point and end at the
    annotated point end.
    """
    points: list[GoldenPoint] = []
    point_numbers: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _POINT_LINE.match(line)
        if match is None:
            continue
        # Descriptions such as "point 9 ends" contain a point number that is not a
        # timestamp. Remove those references before extracting time values.
        description = re.sub(r"\bpoint\s+\d+\b", "point", match.group(2), flags=re.I)
        timestamps = _TIMESTAMP.findall(description)
        if len(timestamps) < 2:
            raise ValueError(f"{path}:{line_number}: point needs start and end timestamps")
        values = [_seconds(timestamp) for timestamp in timestamps]
        end = values[-1]
        starts = (values[0],) if len(values) == 2 else (values[0], values[-2])
        starts = tuple(dict.fromkeys(starts))
        if not starts or any(not 0.0 <= start < end for start in starts):
            raise ValueError(f"{path}:{line_number}: invalid point bounds {starts}-{end}")
        point_numbers.append(int(match.group(1)))
        points.append(GoldenPoint(starts, end))

    expected_numbers = list(range(1, len(points) + 1))
    if point_numbers != expected_numbers:
        raise ValueError(
            f"{path}: expected consecutive point numbers {expected_numbers}, got {point_numbers}"
        )
    if any(
        current.acceptable_starts[0] < previous.end
        for previous, current in zip(points, points[1:])
    ):
        raise ValueError(f"{path}: point intervals overlap or are out of order")
    return points


def assert_rallies_within_boundary_tolerance(
    predicted: list[tuple[float, float]],
    expected: list[GoldenPoint | tuple[float, float]],
    tolerance_s: float = BOUNDARY_TOLERANCE_S,
) -> None:
    """Require one prediction per point and independent start/end errors <= tolerance.

    A gold point may allow multiple starts when a fault/let was followed by another serve.
    """
    predicted = sorted((float(start), float(end)) for start, end in predicted)
    expected = [
        item
        if isinstance(item, GoldenPoint)
        else GoldenPoint((float(item[0]),), float(item[1]))
        for item in expected
    ]
    expected.sort(key=lambda point: point.acceptable_starts[0])
    assert len(predicted) == len(expected), (
        f"expected {len(expected)} rallies but detected {len(predicted)}; "
        f"expected={expected!r}, predicted={predicted!r}"
    )

    failures = []
    for index, ((actual_start, actual_end), gold) in enumerate(
        zip(predicted, expected), 1
    ):
        start_error = min(abs(actual_start - start) for start in gold.acceptable_starts)
        end_error = abs(actual_end - gold.end)
        if start_error > tolerance_s or end_error > tolerance_s:
            failures.append(
                f"point {index}: expected start in {gold.acceptable_starts!r}, "
                f"end {gold.end:.3f}s; "
                f"detected {actual_start:.3f}-{actual_end:.3f}s "
                f"(start error {start_error:.3f}s, end error {end_error:.3f}s)"
            )
    assert not failures, (
        f"rally boundaries exceeded the {tolerance_s:.1f}s tolerance:\n"
        + "\n".join(failures)
    )


def _segments_from_sidecar(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a sidecar object or segment list")
    return [(float(item["start"]), float(item["end"])) for item in records]


@pytest.mark.parametrize(
    "dataset", GOLDEN_DATASETS, ids=lambda dataset: f"dataset-{dataset.number}"
)
def test_ground_truth_annotation_allows_original_or_retry_serve_start(dataset):
    assert load_annotated_points(dataset.annotation) == list(dataset.expected)


def test_boundary_gate_accepts_errors_up_to_two_seconds_independently():
    gold = [(10.0, 20.0), (30.0, 40.0)]
    assert_rallies_within_boundary_tolerance([(8.0, 22.0), (32.0, 38.0)], gold)


def test_boundary_gate_accepts_retry_serve_as_the_point_start():
    gold = [GoldenPoint((91.0, 105.0), 112.0)]
    assert_rallies_within_boundary_tolerance([(103.0, 114.0)], gold)
    assert_rallies_within_boundary_tolerance([(89.0, 110.0)], gold)
    with pytest.raises(AssertionError):
        assert_rallies_within_boundary_tolerance([(98.0, 112.0)], gold)


@pytest.mark.parametrize(
    "predicted",
    [
        [(7.999, 20.0)],
        [(10.0, 22.001)],
        [],
        [(10.0, 20.0), (30.0, 40.0)],
    ],
)
def test_boundary_gate_rejects_large_errors_and_wrong_point_counts(predicted):
    with pytest.raises(AssertionError):
        assert_rallies_within_boundary_tolerance(predicted, [(10.0, 20.0)])


@pytest.mark.parametrize(
    "dataset", GOLDEN_DATASETS, ids=lambda dataset: f"dataset-{dataset.number}"
)
def test_sample_video_matches_all_golden_rallies(dataset, tmp_path):
    cached_sidecar = os.environ.get(dataset.sidecar_environment_variable)
    run_end_to_end = os.environ.get("RALLY_RUN_GOLDEN", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if cached_sidecar:
        predicted = _segments_from_sidecar(Path(cached_sidecar))
    elif run_end_to_end:
        if not dataset.video.is_file():
            pytest.fail(f"golden input video is missing: {dataset.video}")
        assert _sha256(dataset.video) == dataset.video_sha256, (
            "golden input changed without updating its independent annotation"
        )
        from rally.config import RallyConfig
        from rally.pipeline import trim

        sidecar = tmp_path / f"golden-predicted-{dataset.number}.json"
        result = trim(
            str(dataset.video),
            cfg=RallyConfig(),
            json_path=str(sidecar),
            detect_players=True,
        )
        predicted = result.segments
        assert result.stages["audio"]["status"] == "used"
        assert result.stages["visual"]["status"] == "used"
        assert result.stages["visual"]["players"] is True
        assert result.stages["ball_arbiter"]["status"] == "used"
        assert result.stages["ball_arbiter"]["weights_sha256"] == TRACKNET_SHA256
        assert result.stages["match_state"]["status"] != "failed"
    else:
        pytest.skip(
            "slow real-video gate; set RALLY_RUN_GOLDEN=1 or "
            f"{dataset.sidecar_environment_variable}"
        )

    assert_rallies_within_boundary_tolerance(
        predicted, load_annotated_points(dataset.annotation)
    )
