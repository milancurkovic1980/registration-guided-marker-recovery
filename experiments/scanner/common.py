"""Paths and helpers of the experiments on the two scanning sessions.

data/scans/session1/        first session: flat side and side with spheres of the calibration object, desk pad, head
data/scans/session2_glare/  second session: glazed porcelain plate, bowl and cup under projector glare
results/scanner/            outputs of the experiments"""
import json, os, re, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from recovery import Session  # noqa: E402

SESSIONS = {"session1": os.path.join(ROOT, "data", "scans", "session1"),
            "session2_glare": os.path.join(ROOT, "data", "scans", "session2_glare")}
RESULTS = os.path.join(ROOT, "results", "scanner")
os.makedirs(RESULTS, exist_ok=True)
THRESHOLDS = {"yolo12n": 0.15, "yolo12s": 0.03, "rfdetr_s": 0.3}   # chosen on the validation scene of the photographs


def number(view):
    return int(re.findall(r"\d+", view.split("/")[1])[0])


def group(session, view):
    """Surface (session 1) or object (session 2) of a view."""
    p, v = view.split("/")[0], number(view)
    if session == "session1":
        if p == "25_10_2026_13_27_54":
            return "spheres" if v <= 8 else "flat" if v <= 13 else "head" if v <= 16 else "pad"
        return "flat" if v <= 8 or v >= 19 else "spheres" if v <= 11 else "pad"
    return "bowl" if v in (1, 2, 3, 12, 13, 14, 15) else "plate" if 4 <= v <= 11 else "cup"


def load_json(name):
    return json.load(open(os.path.join(RESULTS, name)))


def save_json(obj, name):
    json.dump(obj, open(os.path.join(RESULTS, name), "w"), indent=1)


def bootstrap_gain(n0, n1, n, reps=10000, seed=0):
    """Gain in recall with a 95% interval from a bootstrap over views."""
    n0, n1, n = map(np.asarray, (n0, n1, n)); rng = np.random.default_rng(seed); bs = []
    for _ in range(reps):
        s = rng.integers(0, len(n), len(n)); bs.append((n1[s].sum() - n0[s].sum()) / n[s].sum())
    return float((n1.sum() - n0.sum()) / n.sum()), [float(x) for x in np.percentile(bs, [2.5, 97.5])]
