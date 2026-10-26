"""Labelled data sets of the verifiers.

    python experiments/datasets.py                   candidates (extract.py) + labels -> work/data/<setting>_{scenes,glare}.npz
    python experiments/datasets.py --binarizations   the same candidates of the fixed window with crops cut from other
                                                     binarizations -> work/data/fixed-<method>_{scenes,glare}.npz

Each file holds photo, scene, setup (scene/lighting, the group of the cross-validation), split, y (label), quad,
side (mean side in px) and the crops G, B (64 x 64) and B50 (50 x 50)."""
import csv, os, sys
import numpy as np
from joblib import Parallel, delayed
from common import DATA, WORK, SETTINGS, METHODS, photo_table, grey_stretched, doxa_binarize
from markerdet import rectify


def load_labels(setting):
    lab = {}
    with open(os.path.join(DATA, "labels", setting + ".csv")) as f:
        for r in csv.DictReader(f):
            q = [float(r[k]) for k in ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3")]
            lab.setdefault(r["photo"], []).append((q, int(r["label"])))
    return {p: (np.array([q for q, _ in v]), np.array([y for _, y in v])) for p, v in lab.items()}


def label_of(lab, photo, q):
    Q, Y = lab[photo]
    d = np.abs(Q - np.asarray(q).ravel()).max(1); k = int(np.argmin(d))
    if d[k] > 1e-3:
        raise KeyError(f"candidate of {photo} has no label")
    return int(Y[k])


def build(setting, T):
    lab = load_labels(setting); parts = {"scenes": [], "glare": []}
    for photo in sorted(T):
        z = np.load(os.path.join(WORK, "crops", setting, photo + ".npz")); t = T[photo]
        for k, q in enumerate(z["quads"]):
            side = float(np.mean([np.hypot(*(q[i] - q[(i + 1) % 4])) for i in range(4)]))
            parts["glare" if t["split"] == "glare" else "scenes"].append(
                (photo, t["scene"], f"{t['scene']}/{t['lighting']}", t["split"], label_of(lab, photo, q), q, side,
                 z["G"][k], z["B"][k], z["B50"][k]))
    for part, rows in parts.items():
        c = list(zip(*rows))
        np.savez_compressed(os.path.join(WORK, "data", f"{setting}_{part}.npz"), photo=np.array(c[0]), scene=np.array(c[1]),
                            setup=np.array(c[2]), split=np.array(c[3]), y=np.array(c[4]), quad=np.array(c[5]),
                            side=np.array(c[6]), G=np.stack(c[7]), B=np.stack(c[8]), B50=np.stack(c[9]))
        print(f"{setting}_{part}: {len(rows)} candidates", flush=True)


def binarized_crops(photo, quads):
    g = grey_stretched(photo); out = {}
    for m in METHODS:
        b = doxa_binarize(g, m)
        out[m] = (np.array([rectify(b, q) for q in quads], np.uint8).reshape(-1, 64, 64),
                  np.array([rectify(b, q, 50) for q in quads], np.uint8).reshape(-1, 50, 50))
    return photo, out


if __name__ == "__main__":
    os.makedirs(os.path.join(WORK, "data"), exist_ok=True)
    if "--binarizations" not in sys.argv:
        T = photo_table()
        for setting in SETTINGS:
            build(setting, T)
    else:
        for part in ("scenes", "glare"):
            z = np.load(os.path.join(WORK, "data", f"fixed_{part}.npz"))
            res = Parallel(n_jobs=-1)(delayed(binarized_crops)(p, z["quad"][z["photo"] == p]) for p in sorted(set(z["photo"])))
            for m in METHODS:
                d = {k: z[k] for k in z.files}; d["B"], d["B50"] = d["B"].copy(), d["B50"].copy()
                for photo, out in res:
                    sel = np.where(z["photo"] == photo)[0]; d["B"][sel], d["B50"][sel] = out[m]
                np.savez_compressed(os.path.join(WORK, "data", f"fixed-{m.lower()}_{part}.npz"), **d)
                print(f"fixed-{m.lower()}_{part}", flush=True)
