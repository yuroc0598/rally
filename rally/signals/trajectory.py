"""Ball-trajectory reconstruction: turn a raw, gappy detection track into a clean,
smoothed trajectory with per-sample confidence — the SwingVision-style step that sits
between the ball detector and the tennis rules.

A single-camera ball track (from :mod:`rally.signals.ball`) is noisy and full of holes:
the detector misses the ball on serves, at high speed, and through occlusions (net,
player), and it occasionally locks onto a distractor for one frame. Feeding that raw
track straight into bounce / in-out logic is the main reason ball-based decisions are
inaccurate — a single dropped frame fabricates or hides a bounce.

This module fixes the track *before* the rules see it:

* :func:`smooth_track` runs a constant-velocity **Kalman filter + RTS smoother** over the
  image-space track. Missing frames become model predictions (gaps filled), one-frame
  jumps are gated out as outliers, and every output sample carries a **confidence** in
  ``[0, 1]`` derived from the posterior covariance — high where the ball was tracked
  cleanly, low across long dropouts. Downstream rules weight by this confidence.
* :func:`bounces_from_velocity` detects ground contacts from the sign flip of the ball's
  **vertical image velocity** (moving down the screen -> moving up) on the *smoothed*
  track — the "velocity angle change" cue (cf. the Hawk-Eye stack in DESIGN.md), which is
  far steadier than raw image-y local maxima.

Pure NumPy, no weights, independent of how the track was produced — unit-testable on
synthetic trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ball import BallTrack


@dataclass
class SmoothTrack:
    """A gap-filled, denoised ball track with per-sample confidence.

    ``x``/``y`` are always finite (predictions fill detection gaps). ``confidence`` is
    ~1 where the ball was tracked cleanly and decays toward 0 across long dropouts, so
    downstream rules can ignore fabricated positions. ``measured`` marks samples that had
    a real (non-gated) detection behind them.
    """
    t: np.ndarray            # timestamps (s)
    x: np.ndarray            # smoothed image x (px), always finite
    y: np.ndarray            # smoothed image y (px), always finite
    vx: np.ndarray           # image x-velocity (px/s)
    vy: np.ndarray           # image y-velocity (px/s)
    confidence: np.ndarray   # per-sample confidence in [0, 1]
    measured: np.ndarray     # bool: a real detection backed this sample


def smooth_track(
    track: BallTrack,
    *,
    accel_std_px_s2: float = 1500.0,
    meas_std_px: float = 3.0,
    gate_sigma: float = 5.0,
    max_gap_s: float = 0.5,
) -> SmoothTrack:
    """Kalman-filter + RTS-smooth a raw ball track (constant-velocity model).

    State is ``[x, y, vx, vy]`` in image pixels; measurements are ``[x, y]``. Frames with
    no detection are handled as prediction-only steps (the gap is bridged by the motion
    model). A measurement whose innovation exceeds ``gate_sigma`` standard deviations is
    rejected as an outlier (a one-frame lock onto a player/noise) and also treated as a
    gap. Samples reached only through a dropout longer than ``max_gap_s`` get confidence 0
    (the model can't be trusted to invent a fast ball's path over a long hole).

    The two knobs are physical and interpretable:

    * ``accel_std_px_s2`` — how hard the ball can accelerate between frames (the CV model's
      process noise). A tennis ball reverses direction sharply at a bounce, so this must be
      large enough to follow it; too small and the filter lags into a straight line.
    * ``meas_std_px`` — the detector's positional jitter. Larger => smooth harder.
    """
    t = np.asarray(track.t, float)
    zx = np.asarray(track.x, float)
    zy = np.asarray(track.y, float)
    n = t.size
    process_var = float(accel_std_px_s2) ** 2   # accel PSD for the white-accel Q
    meas_var = float(meas_std_px) ** 2
    if n == 0:
        z = np.zeros(0)
        b = np.zeros(0, bool)
        return SmoothTrack(z, z, z.copy(), z.copy(), z.copy(), z.copy(), b)

    dt_all = np.diff(t)
    dt_med = float(np.median(dt_all)) if dt_all.size else 1.0 / 30.0
    if not np.isfinite(dt_med) or dt_med <= 0:
        dt_med = 1.0 / 30.0

    # storage for filter (forward) results, later refined by the RTS backward pass
    xs_f = np.zeros((n, 4))          # filtered state means
    Ps_f = np.zeros((n, 4, 4))       # filtered covariances
    xs_p = np.zeros((n, 4))          # predicted (a-priori) state means
    Ps_p = np.zeros((n, 4, 4))       # predicted covariances
    Fs = np.zeros((n, 4, 4))         # transition used into each step
    measured = np.zeros(n, bool)

    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    R = np.eye(2) * meas_var

    def _F(dt: float) -> np.ndarray:
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)

    def _Q(dt: float) -> np.ndarray:
        # continuous white-acceleration process noise discretised over dt
        q = process_var
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        return q * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ])

    # --- initialise on the first real detection -----------------------------
    first = int(np.argmax(np.isfinite(zx) & np.isfinite(zy))) if (
        np.isfinite(zx) & np.isfinite(zy)).any() else 0
    x0 = np.array([zx[first] if np.isfinite(zx[first]) else 0.0,
                   zy[first] if np.isfinite(zy[first]) else 0.0, 0.0, 0.0])
    P0 = np.diag([meas_var, meas_var, 1e4, 1e4])
    x_prev, P_prev = x0, P0
    gap_since_meas = 0.0

    for i in range(n):
        dt = dt_med if i == 0 else max(t[i] - t[i - 1], 1e-6)
        F = _F(dt)
        Fs[i] = F
        if i == 0:
            x_pred, P_pred = x0, P0
        else:
            x_pred = F @ x_prev
            P_pred = F @ P_prev @ F.T + _Q(dt)
        xs_p[i], Ps_p[i] = x_pred, P_pred

        z_ok = np.isfinite(zx[i]) and np.isfinite(zy[i])
        if z_ok:
            z = np.array([zx[i], zy[i]])
            S = H @ P_pred @ H.T + R
            innov = z - H @ x_pred
            # Mahalanobis gate: reject one-frame jumps onto players/noise
            try:
                md2 = float(innov @ np.linalg.solve(S, innov))
            except np.linalg.LinAlgError:  # pragma: no cover
                md2 = np.inf
            if md2 <= gate_sigma ** 2:
                K = P_pred @ H.T @ np.linalg.inv(S)
                x_upd = x_pred + K @ innov
                P_upd = (np.eye(4) - K @ H) @ P_pred
                measured[i] = True
                gap_since_meas = 0.0
            else:
                x_upd, P_upd = x_pred, P_pred      # gated out -> predict only
                gap_since_meas += dt
        else:
            x_upd, P_upd = x_pred, P_pred          # missing -> predict only
            gap_since_meas += dt

        xs_f[i], Ps_f[i] = x_upd, P_upd
        x_prev, P_prev = x_upd, P_upd

    # --- RTS backward smoother ---------------------------------------------
    xs_s = xs_f.copy()
    Ps_s = Ps_f.copy()
    for i in range(n - 2, -1, -1):
        F = Fs[i + 1]
        Pp = Ps_p[i + 1]
        try:
            C = Ps_f[i] @ F.T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        xs_s[i] = xs_f[i] + C @ (xs_s[i + 1] - xs_p[i + 1])
        Ps_s[i] = Ps_f[i] + C @ (Ps_s[i + 1] - Pp) @ C.T

    # --- confidence from positional posterior variance ----------------------
    # trace of the position covariance ~ how uncertain the (x, y) estimate is; map to
    # (0, 1] with the measurement noise as the "fully confident" scale.
    pos_var = np.clip(Ps_s[:, 0, 0] + Ps_s[:, 1, 1], 1e-6, None)
    conf = (2.0 * meas_var) / (2.0 * meas_var + pos_var)
    # kill confidence inside dropouts longer than max_gap_s (run length of un-measured)
    conf = _zero_long_gaps(conf, measured, t, max_gap_s)

    return SmoothTrack(
        t=t, x=xs_s[:, 0], y=xs_s[:, 1], vx=xs_s[:, 2], vy=xs_s[:, 3],
        confidence=conf, measured=measured,
    )


def _zero_long_gaps(conf: np.ndarray, measured: np.ndarray, t: np.ndarray,
                    max_gap_s: float) -> np.ndarray:
    """Set confidence to 0 across runs of unmeasured samples spanning > max_gap_s."""
    conf = conf.copy()
    n = conf.size
    i = 0
    while i < n:
        if measured[i]:
            i += 1
            continue
        j = i
        while j < n and not measured[j]:
            j += 1
        span = t[min(j, n - 1)] - t[i]
        if span > max_gap_s:
            conf[i:j] = 0.0
        i = j
    return conf


def bounces_from_velocity(
    track: SmoothTrack,
    *,
    min_descent_px_s: float = 40.0,
    min_sep_s: float = 0.3,
    min_conf: float = 0.3,
) -> list[int]:
    """Ground-contact sample indices from the vertical-velocity sign flip.

    On the *smoothed* track a bounce is where the ball stops moving down the screen and
    starts moving up: ``vy`` crosses from positive (descending, image-y increasing) to
    negative (ascending). Requiring a minimum descent speed just before the flip
    (``min_descent_px_s``) rejects the trajectory apex and slow wobble; ``min_sep_s`` is
    the refractory spacing; low-confidence samples (long dropouts) are skipped.

    This is steadier than image-y ``find_peaks`` because it keys on the *dynamics*
    (velocity reversal) rather than the exact peak height, which perspective distorts.
    """
    vy = np.asarray(track.vy, float)
    t = np.asarray(track.t, float)
    conf = np.asarray(track.confidence, float)
    n = vy.size
    if n < 3:
        return []
    dt_med = float(np.median(np.diff(t))) if n > 1 else 1.0 / 30.0
    refractory = max(1, int(round(min_sep_s / max(dt_med, 1e-6))))

    out: list[int] = []
    for i in range(1, n):
        # vertical velocity crosses from descending (vy > 0, moving down the screen) to
        # ascending (vy <= 0) — the ground contact. (>0 / <=0 keeps the exact-zero sample
        # on one side so a crossing that lands on vy==0 is still caught.)
        if not (vy[i - 1] > 0.0 and vy[i] <= 0.0):
            continue
        # require a genuinely fast descent just before (rejects apex / slow wobble)
        w0 = max(0, i - refractory)
        if vy[w0:i].max(initial=0.0) < min_descent_px_s:
            continue
        if conf[i] < min_conf and conf[i - 1] < min_conf:
            continue                                  # inside a long dropout
        if out and i - out[-1] < refractory:
            continue                                  # refractory spacing
        out.append(i)
    return out
