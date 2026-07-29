"""Train and cross-validate a serve classifier on labelled candidates.

Reads candidates/labels.csv (from rally.tools.serve_dataset, with is_serve filled in),
extracts audio + visual features for each candidate, and runs leave-one-out
cross-validation with logistic regression. Because the dataset is tiny, LOO is the
honest estimate of accuracy — and it's compared against the majority-class baseline so
you can see whether the model actually learns anything beyond "guess the common class".

Usage:
    python -m rally.tools.serve_train samples/tennis_short.mp4 candidates/labels.csv
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from ..signals.audio import detect_strikes_stream
from ..config import RallyConfig
from ..io.ffmpeg import iter_audio_mono


def _clusters(onsets, gap_s=2.5):
    cl = [[float(onsets[0])]]
    for t in onsets[1:]:
        (cl[-1].append(float(t)) if t - cl[-1][-1] <= gap_s else cl.append([float(t)]))
    return cl


def extract_features(video: str, labels_csv: str):
    import cv2
    from ultralytics import YOLO
    from ..signals.player import discover_yolo_weights

    cfg = RallyConfig()
    rows = list(csv.DictReader(open(labels_csv)))
    rows = [r for r in rows if r["is_serve"] in ("0", "1")]
    if not rows:
        raise ValueError("labels CSV contains no rows labeled is_serve=0/1")

    onsets = detect_strikes_stream(
        iter_audio_mono(video, cfg.audio_sr, chunk_s=60.0), cfg.audio_sr, cfg)
    if onsets.size == 0:
        raise ValueError("no audio strike candidates were detected")
    cl = _clusters(onsets)
    firsts = np.array([c[0] for c in cl])

    det = YOLO(discover_yolo_weights("yolov8n.pt"))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"could not decode video or determine frame rate: {video}")

    def feet(t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if not ok:
            return []
        h, w = fr.shape[:2]
        r = det.predict(fr, conf=0.3, classes=[0], verbose=False)[0]
        return [((b[0] + b[2]) / 2 / w, b[3] / h) for b in r.boxes.xyxy.cpu().numpy()]

    # global baseline estimate from feet at all candidate times
    X, y, names = [], [], ["gap_before", "n_strikes", "near_disp", "backmost_y", "frontmost_y"]
    per = []
    used_clusters: set[int] = set()
    for r in rows:
        ps = float(r["point_start_s"])
        fs = ps + cfg.toss_preroll_s
        ci = int(np.argmin(np.abs(firsts - fs)))
        if abs(float(firsts[ci]) - fs) > 1.0:
            raise ValueError(
                f"label {r['index']} has no audio cluster within 1s of its expected strike")
        if ci in used_clusters:
            raise ValueError(f"multiple labels map to audio cluster {ci}; fix candidate alignment")
        used_clusters.add(ci)
        c = cl[ci]
        gap_before = c[0] - cl[ci - 1][-1] if ci > 0 else 30.0
        n_strikes = len(c)
        # visual around the serve
        near0 = None
        disp = 0.0
        backs, fronts = [], []
        t = c[0] - 2.0
        while t <= c[0] + 0.3:
            fs_pts = feet(t)
            if fs_pts:
                backs.append(max(p[1] for p in fs_pts))
                fronts.append(min(p[1] for p in fs_pts))
                near = [p for p in fs_pts if p[1] > 0.5]
                if near:
                    p = max(near, key=lambda z: z[1])
                    if near0 is None:
                        near0 = p
                    disp = max(disp, np.hypot(p[0] - near0[0], p[1] - near0[1]))
            t += 0.3
        if len(backs) < 2:
            raise ValueError(
                f"insufficient player detections around label {r['index']} ({len(backs)} frames)")
        backmost = max(backs) if backs else 0.0
        frontmost = min(fronts) if fronts else 1.0
        X.append([gap_before, n_strikes, disp, backmost, frontmost])
        y.append(int(r["is_serve"]))
        per.append((r["index"], ps, int(r["is_serve"])))
    cap.release()
    return np.array(X, float), np.array(y, int), names, per


def evaluate(X, y, names, per, model_out=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    if n < 4 or classes.size != 2 or counts.min() < 2:
        raise ValueError("serve evaluation needs both classes with at least two samples each")
    maj = 1 if y.sum() >= n / 2 else 0
    base_acc = max(y.sum(), n - y.sum()) / n

    def make_clf():
        # random forest handles small tabular data and non-linear splits well;
        # falls back gracefully with class balancing.
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      min_samples_leaf=2, random_state=0)

    # cross-validation: stratified k-fold when there's enough data, else leave-one-out
    preds = np.zeros(n, int)
    if n >= 20 and y.sum() >= 5 and (n - y.sum()) >= 5:
        k = min(5, int(y.sum()), int(n - y.sum()))
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        scheme = f"{k}-fold stratified CV"
        for tr, te in cv.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = make_clf().fit(sc.transform(X[tr]), y[tr])
            preds[te] = clf.predict(sc.transform(X[te]))
    else:
        scheme = "leave-one-out"
        for i in range(n):
            tr = [j for j in range(n) if j != i]
            sc = StandardScaler().fit(X[tr])
            clf = make_clf()
            clf.fit(sc.transform(X[tr]), y[tr])
            preds[i] = clf.predict(sc.transform(X[i:i + 1]))[0]
    print(f"\nvalidation scheme: {scheme}")

    acc = (preds == y).mean()
    tp = int(((preds == 1) & (y == 1)).sum()); fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum()); tn = int(((preds == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    print(f"\nsamples: {n}  ({int(y.sum())} serve, {int(n - y.sum())} non-serve)")
    print(f"majority-class baseline accuracy: {base_acc:.0%} (predict '{['non-serve','serve'][maj]}')")
    print(f"\ncross-validated RandomForest:")
    print(f"  accuracy = {acc:.0%}   precision = {prec:.0%}   recall = {rec:.0%}")
    print(f"  confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"\nper-candidate (idx, start, true, pred):")
    for (idx, ps, ytrue), p in zip(per, preds):
        flag = "" if p == ytrue else "   <-- wrong"
        print(f"  {idx:>2}  {ps:6.1f}  true={ytrue}  pred={int(p)}{flag}")

    # feature importances from a model fit on all data
    sc = StandardScaler().fit(X)
    full = make_clf().fit(sc.transform(X), y)
    print("\nfeature importances (model fit on all data):")
    for nm, imp in sorted(zip(names, full.feature_importances_), key=lambda z: -z[1]):
        print(f"  {nm:>11}: {imp:.2f}")

    verdict = "USEFUL — beats baseline" if acc > base_acc + 0.05 else \
        ("marginal" if acc >= base_acc else "NOT useful — at/below baseline")
    print(f"\nVERDICT: {verdict}  (CV {acc:.0%} vs baseline {base_acc:.0%})")

    if model_out:
        import joblib
        joblib.dump({"scaler": sc, "model": full, "features": names}, model_out)
        print(f"saved model -> {model_out}")
    return acc, base_acc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rally.tools.serve_train")
    p.add_argument("video")
    p.add_argument("labels_csv")
    p.add_argument("--model-out", default=None, help="save the fitted model to this path (joblib)")
    args = p.parse_args(argv)
    X, y, names, per = extract_features(args.video, args.labels_csv)
    evaluate(X, y, names, per, model_out=args.model_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
