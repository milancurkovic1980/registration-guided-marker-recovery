"""SimpleBlobDetector of OpenCV as a reference detector of the bright dot.

    python experiments/blobs.py      -> work/results/blobs_result.json (filters chosen on the tuning photographs by F1)
                                         and blobs_grid.json (all 18 filter combinations)

A blob of diameter d stands for a marker square of side d / 0.45; a marker is found if it lies inside this
square enlarged 1.3 times."""
import os, json, itertools, time, numpy as np, cv2
from common import WORK, photo_table, annotations, grey_stretched

FEAT = os.path.join(WORK, "blobs")
_T = photo_table()
TUNE = sorted(p for p, r in _T.items() if r["annotation"] == "tuning")
HOLD = sorted(p for p, r in _T.items() if r["annotation"] == "evaluation")
# circularity, inertia ratio, convexity
GRID = list(itertools.product((0.5, 0.7, 0.85), (0.2, 0.4, 0.6), (0.8, 0.9)))


def detector(circ, inert, conv, H, W):
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 10; p.maxThreshold = 250; p.thresholdStep = 10; p.minRepeatability = 2
    p.minDistBetweenBlobs = 3
    p.filterByColor = True; p.blobColor = 255
    p.filterByArea = True; p.minArea = 12; p.maxArea = 0.01 * H * W
    p.filterByCircularity = True; p.minCircularity = circ
    p.filterByInertia = True; p.minInertiaRatio = inert
    p.filterByConvexity = True; p.minConvexity = conv
    return cv2.SimpleBlobDetector_create(p)


def one(p):
    out = os.path.join(FEAT, p + ".npz")
    if os.path.exists(out):
        return
    g = grey_stretched(p); H, W = g.shape
    R = {}
    for prm in GRID:
        kp = detector(*prm, H, W).detect(g)
        R["_".join(map(str, prm))] = np.array([[k.pt[0], k.pt[1], k.size] for k in kp], np.float32).reshape(-1, 3)
    os.makedirs(FEAT, exist_ok=True); np.savez_compressed(out + ".tmp.npz", **R); os.replace(out + ".tmp.npz", out)
    print(p, "done", flush=True)


def evaluate(P, key, F):
    nf = na = ns = nfs = fp = nc = r3 = r6 = 0
    for p in P:
        x = F[p][key]
        S, U = annotations(p); allp = np.vstack([S, U])
        na += len(allp); ns += len(S); nc += len(x)
        if len(x) == 0 or len(allp) == 0:
            fp += len(x); continue
        h = 1.3 * 0.5 * x[:, 2] / 0.45
        inside = (np.abs(allp[:, None, 0] - x[None, :, 0]) <= h) & (np.abs(allp[:, None, 1] - x[None, :, 1]) <= h)
        found = inside.any(1); k = int(found.sum()); nf += k; nfs += int(found[:len(S)].sum()); r3 += k >= 3; r6 += k >= 6
        fp += int((~inside.any(0)).sum())
    return dict(recall=nf / max(na, 1), recall_certain=nfs / max(ns, 1), false=fp, cand=nc, false_per_photo=fp / len(P),
                reg3=r3, reg6=r6, photos=len(P))


if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs(FEAT, exist_ok=True); os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    t0 = time.time()
    with Pool() as pool:
        pool.map(one, TUNE + HOLD, chunksize=1)
    print(f"detection done in {time.time() - t0:.0f} s", flush=True)
    F = {p: dict(np.load(os.path.join(FEAT, p + ".npz"))) for p in TUNE + HOLD}
    groups = {g: [p for p in HOLD if (_T[p]["split"] == "glare") == (g == "glare")] for g in ("test", "glare")}
    best, grid = None, {}
    for prm in GRID:
        key = "_".join(map(str, prm)); r = evaluate(TUNE, key, F)
        grid[key] = {"tune": r, **{g: evaluate(P, key, F) for g, P in groups.items()}}
        prec = 1 - r["false"] / max(r["cand"], 1); f1 = 2 * prec * r["recall"] / max(prec + r["recall"], 1e-9)
        if best is None or f1 > best[0]:
            best = (f1, prm, key)
    res = {"params": {"circularity": best[1][0], "inertia": best[1][1], "convexity": best[1][2]}, **grid[best[2]]}
    json.dump({"sbd": res}, open(os.path.join(WORK, "results", "blobs_result.json"), "w"), indent=1)
    json.dump(grid, open(os.path.join(WORK, "results", "blobs_grid.json"), "w"), indent=1)
    for k in ("tune", "test", "glare"):
        r = res[k]
        print(f"{k:5s} recall {100 * r['recall']:.0f}%  certain {100 * r['recall_certain']:.0f}%  false/photo {r['false_per_photo']:.1f}"
              f"  >=3 {r['reg3']}/{r['photos']}  >=6 {r['reg6']}/{r['photos']}")
    print("filters chosen on the tuning photographs:", res["params"])
