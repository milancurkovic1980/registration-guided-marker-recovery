"""Shared paths and helpers of the experiments.

photos/<photo>.jpg              photographs
data/photos.csv                 scene, lighting, split (train / test / glare), annotation (tuning / evaluation)
data/annotations/<photo>.json   hand-marked marker centres (certain, uncertain)
data/labels/<setting>.csv       candidate labels: 1 marker, 0 non-marker, -2 too small, -1 excluded
work/                           results of the experiments"""
import csv, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
PHOTOS, DATA, WORK = (os.path.join(ROOT, d) for d in ("photos", "data", "work"))
SETTINGS = {"fixed": ("fixed", 0.0), "adaptive": ("adaptive", 0.0), "adaptive_smoothed": ("adaptive", 0.6)}
METHODS = ["OTSU", "SAUVOLA", "BRADLEY", "WAN"]          # other binarizations (Doxa library)


def photo_table():
    with open(os.path.join(DATA, "photos.csv")) as f:
        return {r["photo"]: r for r in csv.DictReader(f)}


def photo_path(photo):
    for ext in (".jpg", ".png", ".bmp"):
        p = os.path.join(PHOTOS, photo + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{photo} not found in {PHOTOS}")


def annotations(photo):
    a = json.load(open(os.path.join(DATA, "annotations", photo + ".json")))
    return np.array(a["certain"], float).reshape(-1, 2), np.array(a["uncertain"], float).reshape(-1, 2)


def match(quads, pts, g=1.3):
    """found[i]: point i lies in a candidate enlarged g times; false[k]: candidate k contains no point."""
    from markerdet.extract import inside_mask
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(quads) == 0:
        return np.zeros(len(pts), bool), np.zeros(0, bool)
    ins = np.zeros((len(pts), len(quads)), bool)
    for k, q in enumerate(np.asarray(quads, float)):
        c = q.mean(0); qq = c + g * (q - c)
        ins[:, k] = inside_mask([tuple(t) for t in qq], pts[:, 0], pts[:, 1])
    return ins.any(1), ~ins.any(0)


def grey_stretched(photo):
    """Grey image stretched to 0..255, input of the other binarizations."""
    from markerdet import load_rgb
    rgb = load_rgb(photo_path(photo)).astype(np.float32)
    g = 0.30 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    g = (g - g.min()) * (255.0 / max(g.max() - g.min(), 1e-6))
    return np.clip(np.round(g), 0, 255).astype(np.uint8)


def doxa_binarize(g, method, window=None):
    """Binarization with the Doxa library (pip install doxapy); local methods use the given or the fixed window."""
    import doxapy
    out = np.empty_like(g)
    alg = doxapy.Binarization(getattr(doxapy.Binarization.Algorithms, method))
    alg.initialize(g)
    alg.to_binary(out, {} if method == "OTSU" else {"window": int(window or 2 * (min(g.shape) // 20) + 1)})
    return out
