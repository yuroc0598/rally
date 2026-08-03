"""Independent ball-tracking module: video -> per-frame ball position.

Two backends behind one interface:

* ``tracknet`` — the purpose-built heatmap CNN (the reliable choice). It needs a
  pretrained weights file (``weights_path``); the architecture here follows the
  TrackNet design (stacked consecutive frames -> Gaussian heatmap -> peak = ball).
* ``motion`` — a training-free classical detector (3-frame differencing + small
  fast-blob + speed-limited linking). Runs anywhere, but low recall on small/blurry
  balls (verified on this night footage) — use it as a fallback / for well-lit,
  higher-resolution clips.

The output is a :class:`BallTrack` (time, x, y image px, visibility). Feed it to
``rally.signals.ballrules`` for bounce / in-out / point-end / speed.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


# Keep uploads/CPU preprocessing concurrent while limiting simultaneous TrackNet streams.
# Two streams overlap video decoding and GPU execution well; four independent models tend
# to reduce per-job throughput and make CUDA memory peaks unpredictable.
_GPU_TRACK_SLOTS = _positive_env_int("RALLY_GPU_TRACK_SLOTS", 2)
_GPU_TRACK_SEMAPHORE = threading.BoundedSemaphore(_GPU_TRACK_SLOTS)
_HEATMAP_DECODE_WORKERS = _positive_env_int(
    "RALLY_HEATMAP_DECODE_WORKERS", min(4, max(1, os.cpu_count() or 1)))
_HEATMAP_EXECUTOR = ThreadPoolExecutor(
    max_workers=_HEATMAP_DECODE_WORKERS, thread_name_prefix="tracknet-heatmap")
_BALL_MODEL_LOCK = threading.Lock()
_BALL_MODEL_CACHE: dict[tuple, object] = {}


def resolve_ball_batch_size(device, requested: Optional[int] = None, *, torch_module=None) -> int:
    """Resolve an explicit or memory-aware TrackNet inference batch size."""
    if requested is not None and int(requested) > 0:
        return int(requested)
    env_batch = os.environ.get("RALLY_BALL_BATCH_SIZE")
    if env_batch is not None:
        return _positive_env_int("RALLY_BALL_BATCH_SIZE", 1)
    if getattr(device, "type", str(device).split(":", 1)[0]) != "cuda":
        return 1
    if torch_module is None:
        import torch as torch_module
    try:
        free_bytes = int(torch_module.cuda.mem_get_info(device)[0])
    except Exception:
        return 4
    gib = free_bytes / float(1024 ** 3)
    # TrackNet's 256-class heatmaps are the dominant allocation. Large accelerator cards
    # can amortize video decode, host/device copies, and kernel launch overhead with a
    # materially larger batch; retain conservative sizes on ordinary consumer GPUs.
    if gib >= 60.0:
        return 32
    if gib >= 36.0:
        return 16
    if gib >= 20.0:
        return 8
    if gib >= 10.0:
        return 4
    return 2


@dataclass
class BallTrack:
    t: np.ndarray          # timestamps (s)
    x: np.ndarray          # image x (px), NaN where not detected
    y: np.ndarray          # image y (px), NaN where not detected

    @property
    def visible(self) -> np.ndarray:
        return np.isfinite(self.x) & np.isfinite(self.y)

    def detection_rate(self) -> float:
        return float(self.visible.mean()) if self.t.size else 0.0


# ---------------------------------------------------------------------------
# classical motion backend (runs without weights)
# ---------------------------------------------------------------------------
def track_motion(video: str, fps_a: Optional[float] = None,
                 min_area: int = 4, max_area: int = 400,
                 diff_thresh: int = 22, speed_limit_px: float = 250.0) -> BallTrack:
    """Track the ball by 3-frame differencing + small fast-blob + trajectory linking.

    Processes at native fps (fast motion needs it). ``speed_limit_px`` rejects blobs
    that jump implausibly far from the previous ball position (players, noise).
    """
    import cv2

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ts, xs, ys = [], [], []
    prev_ball = None
    misses = 0
    pending_reacq = None
    pending_count = 0

    ok, f0 = cap.read()
    ok2, f1 = cap.read()
    if not (ok and ok2):
        cap.release()
        return BallTrack(np.array([]), np.array([]), np.array([]))
    g_prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g_cur = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    idx = 1
    while True:
        ok, f2 = cap.read()
        if not ok:
            break
        g_next = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        d1 = cv2.threshold(cv2.absdiff(g_cur, g_prev), diff_thresh, 255, cv2.THRESH_BINARY)[1]
        d2 = cv2.threshold(cv2.absdiff(g_next, g_cur), diff_thresh, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(cv2.bitwise_and(d1, d2), np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            a = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            if min_area < a < max_area and 0.4 < bw / max(bh, 1) < 2.6:
                cands.append((x + bw / 2.0, y + bh / 2.0, a))
        best = None
        if cands:
            reacquire_after = max(3, int(round(0.25 * fps)))
            if prev_ball is not None and misses < reacquire_after:
                # Never widen immediate acceptance after a miss: a candidate outside the
                # normal gate must go through confirmed reacquisition below.
                gate = speed_limit_px
                near = [(cx, cy, a) for cx, cy, a in cands
                        if np.hypot(cx - prev_ball[0], cy - prev_ball[1]) <= gate]
                # A speed gate is a rejection gate, not merely a preference.  Falling
                # back to the closest far-away blob makes the track teleport onto a
                # player or line whenever the real ball is missed for one frame.
                if near:
                    best = min(near, key=lambda z: np.hypot(
                        z[0] - prev_ball[0], z[1] - prev_ball[1]))
            else:
                if pending_reacq is None:
                    candidate = max(cands, key=lambda z: z[2])
                    pending_count = 1
                else:
                    near = [c for c in cands if np.hypot(
                        c[0] - pending_reacq[0], c[1] - pending_reacq[1]) <= speed_limit_px]
                    candidate = (min(near, key=lambda z: np.hypot(
                        z[0] - pending_reacq[0], z[1] - pending_reacq[1])) if near else None)
                    pending_count = pending_count + 1 if candidate is not None else 0
                pending_reacq = candidate
                if candidate is not None and pending_count >= 3:
                    best = candidate
                    pending_reacq = None
                    pending_count = 0
        elif pending_reacq is not None:
            pending_reacq = None
            pending_count = 0
        ts.append(idx / fps)
        if best is not None:
            xs.append(best[0]); ys.append(best[1])
            prev_ball = best
            misses = 0
            pending_reacq = None
            pending_count = 0
        else:
            xs.append(np.nan); ys.append(np.nan)
            misses += 1
        g_prev, g_cur = g_cur, g_next
        idx += 1
    cap.release()
    return BallTrack(np.array(ts), np.array(xs), np.array(ys))


# ---------------------------------------------------------------------------
# TrackNet backend (3-frame heatmap CNN, PyTorch; needs pretrained weights)
# ---------------------------------------------------------------------------
def _heatmap_candidates(feature_map_360x640: np.ndarray, scale: int = 1):
    """Extract TrackNet circles independently of chronological association."""
    import cv2
    hm = feature_map_360x640.astype(np.uint8)
    # Most between-point frames are pure background. HoughCircles still scans the entire
    # 640x360 map before returning None, so preserve the exact result with a cheap guard.
    if not np.any(hm > 127):
        return []
    _, binm = cv2.threshold(hm, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(binm, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                               param1=50, param2=2, minRadius=2, maxRadius=7)
    if circles is None:
        return []
    return [(c[0] * scale, c[1] * scale) for c in circles[0]]


def _select_heatmap_candidate(feature_map_360x640: np.ndarray, cands, prev=None,
                              speed_limit_px: float = 200.0, scale: int = 1):
    if not cands:
        return None
    if prev is not None:
        near = [c for c in cands if np.hypot(c[0] - prev[0], c[1] - prev[1]) <= speed_limit_px]
        if not near:
            return None
        return min(near, key=lambda c: np.hypot(c[0] - prev[0], c[1] - prev[1]))
    # With no prior, prefer the strongest heatmap peak instead of OpenCV's arbitrary
    # Hough ordering. Once a track exists, continuity above remains the primary signal.
    def heat(c):
        hm = feature_map_360x640
        x = int(np.clip(round(c[0] / scale), 0, hm.shape[1] - 1))
        y = int(np.clip(round(c[1] / scale), 0, hm.shape[0] - 1))
        return int(hm[y, x])
    return max(cands, key=heat)


def _target_court_heatmap_candidates(
    candidates, court, scale_x: float, scale_y: float, *,
    sideline_margin_m: float, baseline_margin_m: float,
):
    """Filter candidate centres before association can lock onto a neighboring ball."""
    if court is None or not candidates:
        return list(candidates)
    from .court import COURT_L, DOUBLES_W

    image_candidates = np.asarray([
        (candidate[0] * scale_x, candidate[1] * scale_y)
        for candidate in candidates
    ], dtype=float)
    court_candidates = court.to_court(image_candidates)
    return [
        candidate for candidate, mapped in zip(candidates, court_candidates)
        if (np.isfinite(mapped).all()
            and -sideline_margin_m <= mapped[0] <= DOUBLES_W + sideline_margin_m
            and -baseline_margin_m <= mapped[1] <= COURT_L + baseline_margin_m)
    ]


def _decode_heatmap(feature_map_360x640: np.ndarray, prev=None,
                    speed_limit_px: float = 200.0, scale: int = 1):
    """Compatibility wrapper for candidate extraction plus chronological association."""
    candidates = _heatmap_candidates(feature_map_360x640, scale=scale)
    return _select_heatmap_candidate(
        feature_map_360x640, candidates, prev=prev,
        speed_limit_px=speed_limit_px, scale=scale)


def discover_ball_weights(models_dir: Optional[str] = None) -> Optional[str]:
    """Find a locally installed TrackNet checkpoint so ball mode needs no explicit path.

    Only TrackNet-named files are eligible. Falling back to an arbitrary ``.pt`` can load
    a pose, court, or learned serve checkpoint as the ball network and fail after an
    expensive job has already started. Discovery proves neither provenance nor licence.
    """
    import os
    from pathlib import Path

    if models_dir is None:
        # Do not make model discovery depend on the process's working directory.
        # An explicit argument remains relative to the caller, while the environment
        # override is useful for packaged/deployed installations.
        models_dir = os.environ.get("RALLY_MODELS_DIR")
        if not models_dir:
            models_dir = str(Path(__file__).resolve().parents[2] / "models")

    for pat in ("tracknet*.pt", "*tracknet*.pt"):
        for path in sorted(Path(models_dir).glob(pat)):
            return str(path)
    return None


def resolve_device(torch=None):
    """Pick the torch device for ball tracking: GPU when available, else CPU.

    Honours ``RALLY_DEVICE`` (e.g. ``cuda``, ``cuda:1``, ``cpu``) as an override; if it
    asks for CUDA but none is present, we warn and fall back to CPU rather than crash.
    """
    import os

    if torch is None:
        import torch

    want = os.environ.get("RALLY_DEVICE", "").strip().lower()
    if want:
        if want.startswith("cuda") and not torch.cuda.is_available():
            print(f"[ball] RALLY_DEVICE={want} requested but CUDA is unavailable — using CPU")
            return torch.device("cpu")
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ball_model(weights_path: str, device=None):
    """Load the 3-frame PyTorch TrackNet once (reuse across segments).

    The model is moved to ``device`` (default: GPU when available — see
    :func:`resolve_device`). ``track_tracknet`` reads the device back off the model, so a
    preloaded model keeps running wherever it was placed.
    """
    import torch

    from ..vendor.tracknet_torch import BallTrackerNet

    model = BallTrackerNet()
    # weights_only=True: the checkpoint is a plain tensor state-dict, so refuse to
    # unpickle arbitrary objects (avoids code execution from a tampered .pt file).
    # Load onto CPU first, then move — robust whether or not a GPU is present.
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
    model.load_state_dict(state)
    model.eval()
    dev = resolve_device(torch) if device is None else torch.device(device)
    model.to(dev)
    if dev.type == "cuda":
        # Fixed-resolution inference benefits from cuDNN autotuning. Track-time precision
        # selection happens later so the CPU fallback remains full precision.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[ball] TrackNet ball tracker running on {dev}")
    return model


def get_cached_ball_model(
    weights_path: str, device=None, *, half_precision: bool = True,
):
    """Reuse one immutable eval-mode TrackNet per checkpoint/device/precision."""
    import torch
    from pathlib import Path

    path = Path(weights_path).expanduser().resolve()
    stat = path.stat()
    dev = resolve_device(torch) if device is None else torch.device(device)
    use_half = bool(half_precision and dev.type == "cuda")
    key = (str(path), int(stat.st_mtime_ns), int(stat.st_size), str(dev), use_half)
    with _BALL_MODEL_LOCK:
        model = _BALL_MODEL_CACHE.get(key)
        if model is None:
            model = load_ball_model(str(path), device=dev)
            if use_half:
                model.half()
            _BALL_MODEL_CACHE[key] = model
        return model


def track_tracknet(video: str, weights_path: Optional[str] = None, *, model=None,
                   start_s: float = 0.0, end_s: Optional[float] = None,
                   width: int = 640, height: int = 360, speed_limit_px: float = 200.0,
                   batch_size: Optional[int] = None,
                   tracking_fps: Optional[float] = 30.0,
                   half_precision: bool = True,
                   court=None,
                   court_sideline_margin_m: float = 1.0,
                   court_baseline_margin_m: float = 3.0,
                   progress_callback: Callable[[int, int, int], None] | None = None,
                   metrics: Optional[dict] = None,
                   cancel_check: Callable[[], None] = lambda: None) -> BallTrack:
    """Ball positions per frame via the 3-frame PyTorch TrackNet (BallTrackerNet).

    Three sampled frames are stacked (9 channels) so the net sees the ball's motion —
    far more reliable for a small/blurry ball than any single-frame detector. Optionally
    restrict to ``[start_s, end_s]`` (to process one rally) and pass a preloaded ``model``.
    Pure PyTorch — no TensorFlow.
    """
    import cv2
    import torch

    if model is None:
        if not weights_path:
            raise ValueError("TrackNet weights_path or model is required")
        model = get_cached_ball_model(
            weights_path, half_precision=half_precision)
    # Run inputs on whatever device the model lives on (GPU when available).  CNN outputs
    # are batched, while heatmap/data-association decoding remains strictly chronological.
    device = next(model.parameters()).device
    if device.type == "cuda" and half_precision:
        model.half()
    model_dtype = next(model.parameters()).dtype
    batch_size = resolve_ball_batch_size(device, batch_size, torch_module=torch)

    wall_started = time.perf_counter()
    perf = Counter()
    batch_histogram: Counter[int] = Counter()
    pipeline_enabled = os.environ.get(
        "RALLY_TRACKNET_PIPELINE", "1").strip().lower() not in {
            "0", "false", "no", "off",
        }

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_stride = max(
        1, int(round(fps / max(float(tracking_fps or fps), 1e-6))))
    effective_fps = fps / sample_stride
    # Two sampled frames of lead are required for the first 3-frame stack.
    start_f = max(0, int(round((start_s - 2 / effective_fps) * fps)))
    start_f -= start_f % sample_stride
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    end_f = None if end_s is None else int(round(end_s * fps))
    source_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    last_f = max(
        start_f - 1,
        (source_frames - 1 if end_f is None else min(source_frames - 1, end_f)),
    )
    total_frames = max(0, (last_f - start_f) // sample_stride + 1)
    scale_x = scale_y = None
    ts, xs, ys = [], [], []
    pending_inputs = []
    pending_indices: list[int] = []
    pending_heatmaps = deque()
    pinned_input = None
    prev = None
    misses = 0
    pending_reacq = None
    pending_count = 0
    fi = start_f
    reacquire_after = max(3, int(round(0.25 * effective_fps)))
    next_progress_fraction = 0.0

    def report_progress(*, force: bool = False) -> None:
        nonlocal next_progress_fraction
        if progress_callback is None:
            return
        done = len(ts)
        fraction = done / max(1, total_frames)
        if force or fraction + 1e-9 >= next_progress_fraction:
            progress_callback(done, total_frames, batch_size)
            next_progress_fraction = min(1.0, fraction + 0.05)

    def decode_map(fm, output_index: int, candidates) -> None:
        nonlocal prev, misses, pending_reacq, pending_count
        pos = None
        base_gate = speed_limit_px / max(scale_x, 1)
        candidates = _target_court_heatmap_candidates(
            candidates, court, scale_x, scale_y,
            sideline_margin_m=court_sideline_margin_m,
            baseline_margin_m=court_baseline_margin_m,
        )
        if prev is not None and misses < reacquire_after:
            pv = (prev[0] / scale_x, prev[1] / scale_y)
            decoded = _select_heatmap_candidate(
                fm, candidates, prev=pv, speed_limit_px=base_gate)
            if decoded is not None:
                pos = (decoded[0] * scale_x, decoded[1] * scale_y)
        else:
            pv = (None if pending_reacq is None else
                  (pending_reacq[0] / scale_x, pending_reacq[1] / scale_y))
            decoded = _select_heatmap_candidate(
                fm, candidates, prev=pv, speed_limit_px=base_gate)
            if decoded is None:
                pending_reacq = None
                pending_count = 0
            else:
                candidate = (decoded[0] * scale_x, decoded[1] * scale_y)
                pending_count = pending_count + 1 if pending_reacq is not None else 1
                pending_reacq = candidate
                if pending_count >= 3:
                    pos = candidate
                    pending_reacq = None
                    pending_count = 0
        if pos is not None:
            xs[output_index], ys[output_index] = pos
            prev = pos
            misses = 0
            pending_reacq = None
            pending_count = 0
        else:
            misses += 1

    def decode_heatmap_batch() -> None:
        if not pending_heatmaps:
            return
        frame_maps, output_indices, futures = pending_heatmaps.popleft()
        wait_started = time.perf_counter()
        timed_candidates = [future.result() for future in futures]
        perf["heatmap_wait_seconds"] += time.perf_counter() - wait_started
        perf["heatmap_seconds"] += sum(seconds for _candidates, seconds in timed_candidates)
        association_started = time.perf_counter()
        for output_index, fm, (candidates, _seconds) in zip(
                output_indices, frame_maps, timed_candidates):
            decode_map(fm, output_index, candidates)
        perf["association_seconds"] += time.perf_counter() - association_started

    def timed_heatmap_candidates(frame_map):
        started = time.perf_counter()
        result = _heatmap_candidates(frame_map)
        return result, time.perf_counter() - started

    def flush_batch() -> None:
        nonlocal pinned_input
        if not pending_inputs:
            return
        cancel_check()
        stack_started = time.perf_counter()
        if device.type == "cuda":
            shape = (batch_size, *pending_inputs[0].shape)
            if pinned_input is None or tuple(pinned_input.shape) != shape:
                pinned_input = torch.empty(
                    shape, dtype=pending_inputs[0].dtype, pin_memory=True)
            inp = pinned_input[:len(pending_inputs)]
            torch.stack(pending_inputs, dim=0, out=inp)
        else:
            inp = torch.stack(pending_inputs, dim=0)
        perf["stack_pin_seconds"] += time.perf_counter() - stack_started
        h2d_started = time.perf_counter()
        # Keep decoded frames as uint8 through the CPU queue and PCIe transfer.  The old
        # path expanded every sample to float32 before pinning, which quadrupled queue and
        # transfer traffic.  Normalize in float32 on the inference device so values round
        # the same way before the optional float16 cast used by TrackNet.
        inp = inp.to(device=device, non_blocking=True)
        inp = inp.to(dtype=torch.float32).div_(255.0)
        if model_dtype != torch.float32:
            inp = inp.to(dtype=model_dtype)
        perf["h2d_submit_seconds"] += time.perf_counter() - h2d_started
        inference_started = time.perf_counter()
        maps = model(inp).argmax(dim=1).cpu().numpy()
        perf["inference_d2h_seconds"] += time.perf_counter() - inference_started
        perf["inferred_frames"] += len(pending_inputs)
        batch_histogram[len(pending_inputs)] += 1
        cancel_check()
        frame_maps = [fm.reshape((height, width)) for fm in maps]
        futures = [
            _HEATMAP_EXECUTOR.submit(timed_heatmap_candidates, frame_map)
            for frame_map in frame_maps
        ]
        pending_heatmaps.append((frame_maps, list(pending_indices), futures))
        pending_inputs.clear()
        pending_indices.clear()
        # Keep at most one completed/in-flight heatmap batch behind GPU inference. This
        # overlaps CPU Hough extraction for N with inference for N+1 while preserving
        # strictly chronological association when the oldest batch is drained.
        if not pipeline_enabled or len(pending_heatmaps) >= 2:
            decode_heatmap_batch()
        report_progress()

    def accept_sample(sample_time: float, tensor, frame_width: int, frame_height: int) -> None:
        nonlocal scale_x, scale_y, misses
        if scale_x is None:
            scale_x, scale_y = frame_width / width, frame_height / height
        ts.append(sample_time)
        xs.append(np.nan)
        ys.append(np.nan)
        perf["sampled_frames"] += 1
        if tensor is None:
            misses += 1
            return
        pending_inputs.append(tensor)
        pending_indices.append(len(xs) - 1)
        if len(pending_inputs) >= batch_size:
            flush_batch()

    def serial_decode() -> None:
        nonlocal fi
        buf = []
        while True:
            cancel_check()
            if end_f is not None and fi > end_f:
                break
            decode_started = time.perf_counter()
            ok = cap.grab()
            perf["decode_seconds"] += time.perf_counter() - decode_started
            if not ok:
                break
            frame_index = fi
            fi += 1
            perf["decoded_frames"] += 1
            if (frame_index - start_f) % sample_stride:
                continue
            retrieve_started = time.perf_counter()
            ok, fr = cap.retrieve()
            perf["decode_seconds"] += time.perf_counter() - retrieve_started
            if not ok:
                break
            preprocess_started = time.perf_counter()
            buf.append(cv2.resize(fr, (width, height)))
            if len(buf) > 3:
                buf.pop(0)
            tensor = None
            if len(buf) == 3:
                imgs = np.concatenate((buf[2], buf[1], buf[0]), axis=2)
                tensor = torch.from_numpy(np.rollaxis(imgs, 2, 0))
            perf["preprocess_seconds"] += time.perf_counter() - preprocess_started
            accept_sample(frame_index / fps, tensor, fr.shape[1], fr.shape[0])

    def pipelined_decode() -> None:
        nonlocal fi
        work_queue: queue.Queue = queue.Queue(maxsize=max(4, batch_size * 2))
        sentinel = object()
        stop = threading.Event()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    work_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def producer() -> None:
            local_fi = fi
            buf = []
            error = None
            try:
                while True:
                    cancel_check()
                    if end_f is not None and local_fi > end_f:
                        break
                    decode_started = time.perf_counter()
                    ok = cap.grab()
                    perf["decode_seconds"] += time.perf_counter() - decode_started
                    if not ok:
                        break
                    frame_index = local_fi
                    local_fi += 1
                    perf["decoded_frames"] += 1
                    if (frame_index - start_f) % sample_stride:
                        continue
                    retrieve_started = time.perf_counter()
                    ok, fr = cap.retrieve()
                    perf["decode_seconds"] += time.perf_counter() - retrieve_started
                    if not ok:
                        break
                    preprocess_started = time.perf_counter()
                    buf.append(cv2.resize(fr, (width, height)))
                    if len(buf) > 3:
                        buf.pop(0)
                    tensor = None
                    if len(buf) == 3:
                        imgs = np.concatenate((buf[2], buf[1], buf[0]), axis=2)
                        tensor = torch.from_numpy(np.rollaxis(imgs, 2, 0))
                    perf["preprocess_seconds"] += time.perf_counter() - preprocess_started
                    if not put((frame_index / fps, tensor, fr.shape[1], fr.shape[0])):
                        return
            except BaseException as exc:  # propagate to the inference owner
                error = exc
            finally:
                put((sentinel, error, None, None))

        producer_thread = threading.Thread(
            target=producer, name="tracknet-decode", daemon=True)
        producer_thread.start()
        try:
            while True:
                queue_started = time.perf_counter()
                item = work_queue.get()
                perf["input_queue_wait_seconds"] += time.perf_counter() - queue_started
                if item[0] is sentinel:
                    if item[1] is not None:
                        raise item[1]
                    break
                accept_sample(*item)
        finally:
            stop.set()
            producer_thread.join(timeout=5.0)
            if producer_thread.is_alive():
                raise RuntimeError("TrackNet decode producer did not stop")

    slot = _GPU_TRACK_SEMAPHORE if device.type == "cuda" else nullcontext()
    try:
        with slot, torch.inference_mode():
            report_progress(force=True)
            if pipeline_enabled:
                pipelined_decode()
            else:
                serial_decode()
            flush_batch()
            while pending_heatmaps:
                decode_heatmap_batch()
            report_progress(force=True)
    finally:
        cap.release()
        perf["wall_seconds"] = time.perf_counter() - wall_started
        perf["pipeline_enabled"] = int(pipeline_enabled)
        if metrics is not None:
            metrics.update({
                key: (int(value) if key.endswith("_frames") else float(value))
                for key, value in perf.items()
            })
            metrics["batch_histogram"] = {
                str(size): int(count) for size, count in sorted(batch_histogram.items())
            }
    return BallTrack(np.array(ts), np.array(xs), np.array(ys))


def ball_in_play_channel(track: BallTrack, timeline: np.ndarray,
                         window_s: float = 1.0, min_speed_px: float = 3.0,
                         min_displacement_px: float = 0.75, court=None,
                         court_margin_m: float = 0.75) -> np.ndarray:
    """Per-analysis-frame ball-in-play evidence (0..1) for the rally fusion.

    For each timeline time, the fraction of nearby tracked frames where the ball is both
    detected and moving (a live ball) — high during a rally, ~0 between points.
    """
    timeline = np.asarray(timeline, float)
    if window_s <= 0 or min_speed_px < 0 or min_displacement_px < 0:
        raise ValueError("window_s must be positive and speed/displacement non-negative")
    out = np.zeros(timeline.size, float)
    if track.t.size < 2:
        return out
    tt = np.asarray(track.t, float)
    if np.any(np.diff(tt) < 0):
        raise ValueError("BallTrack timestamps must be monotonic")
    vis = track.visible
    sp = np.zeros(track.t.size)
    dt = np.diff(tt)
    dx, dy = np.diff(track.x), np.diff(track.y)
    displacement = np.hypot(dx, dy)
    # A speed threshold alone turns alternating detector jitter into high speed at high
    # frame rates. Require meaningful, directionally coherent motion across an adjacent
    # pair of steps. A genuine turn may lose one sample; it does not fabricate a run.
    coherent = np.ones(displacement.size, dtype=bool) if displacement.size == 1 else np.zeros(
        displacement.size, dtype=bool)
    if displacement.size >= 2:
        same_direction = (dx[:-1] * dx[1:] + dy[:-1] * dy[1:]) > 0
        coherent[:-1] |= same_direction
        coherent[1:] |= same_direction
    adjacent = (vis[1:] & vis[:-1] & (dt > 0)
                & (displacement >= min_displacement_px) & coherent)
    step_speed = np.zeros(dt.size, float)
    np.divide(displacement, dt, out=step_speed, where=adjacent)
    sp[1:] = step_speed
    active = vis & (sp > min_speed_px)
    if court is not None and np.any(active):
        from .court import COURT_L, DOUBLES_W

        indices = np.flatnonzero(active)
        coords = court.to_court(np.stack([track.x[indices], track.y[indices]], axis=1))
        inside = (
            (coords[:, 0] >= -court_margin_m)
            & (coords[:, 0] <= DOUBLES_W + court_margin_m)
            & (coords[:, 1] >= -court_margin_m)
            & (coords[:, 1] <= COURT_L + court_margin_m)
        )
        active[indices[~inside]] = False
    # Prefix counts plus vectorised binary searches replace one full-track mask per
    # timeline sample.  Memory stays O(N+T), and no T-by-N temporary is created.
    lo = np.searchsorted(tt, timeline - window_s / 2, side="left")
    hi = np.searchsorted(tt, timeline + window_s / 2, side="right")
    prefix = np.r_[0, np.cumsum(active, dtype=np.int64)]
    count = hi - lo
    nonempty = count > 0
    out[nonempty] = ((prefix[hi] - prefix[lo])[nonempty]
                     / count[nonempty])
    return out


class BallTracker:
    """Unified ball tracker. ``backend='tracknet'`` (3-frame PyTorch net + weights) is
    the reliable choice; ``backend='motion'`` runs anywhere but with low recall."""

    def __init__(self, backend: str = "motion", weights_path: Optional[str] = None):
        self.backend = backend
        self.weights_path = weights_path

    def track(self, video: str, **kw) -> BallTrack:
        if self.backend == "tracknet":
            if not self.weights_path:
                raise ValueError("tracknet backend needs weights_path (a TrackNet .pt)")
            return track_tracknet(video, self.weights_path, **kw)
        return track_motion(video, **kw)
