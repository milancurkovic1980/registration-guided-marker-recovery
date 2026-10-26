"""Registration with three centres of the same markers (Table of the centres in the article).

    python experiments/scanner/centres.py

For every view pair with at least five corresponding markers that have a 3D point for all three centres
(quadrilateral of the classical detector, box of YOLOv12-S, centroid of the dot), every marker is left out in turn,
the views are aligned on the others, and the RMS distance of the left-out markers is computed. The differences
between centres get 95% and 90% intervals from a bootstrap over views (equivalence: 90% interval within 0.05 mm)."""
import itertools
import numpy as np
from common import SESSIONS, Session, group, save_json
from recovery import kabsch, match_markers

M = ("quad", "box", "dot")


def loo_pairs(ses, name):
    rows = []
    for p in ses.projects():
        vs = ses.views_of(p); data = {v: {m: ses.markers3d(v, m) for m in M} for v in vs}
        for vi, vj in itertools.combinations(vs, 2):
            cons = lambda v: {k: np.mean([data[v][m][k][1] for m in M], 0) for k in set.intersection(*[set(data[v][m]) for m in M])}
            mt = match_markers(cons(vi), cons(vj))
            if len(mt) < 5:
                continue
            r = dict(view_i=vi, view_j=vj, n=len(mt), group=group(name, vi))
            for m in M:
                P = np.array([data[vi][m][a][1] for a in mt]); Q = np.array([data[vj][m][b][1] for b in mt.values()]); e = []
                for i in range(len(P)):
                    k = np.arange(len(P)) != i; R, t = kabsch(P[k], Q[k]); e.append(np.linalg.norm(P[i] @ R.T + t - Q[i]))
                r[m] = float(np.sqrt(np.mean(np.square(e))))
            rows.append(r)
    return rows


def bootstrap(rows, a, b, reps=5000, seed=0):
    rng = np.random.default_rng(seed); keys = sorted({r["view_i"] for r in rows} | {r["view_j"] for r in rows})
    d = np.array([r[a] - r[b] for r in rows]); bs = []
    for _ in range(reps):
        cnt = np.bincount(rng.choice(len(keys), len(keys)), minlength=len(keys)); w = dict(zip(keys, cnt))
        ww = np.array([w[r["view_i"]] * w[r["view_j"]] for r in rows], float)
        if ww.sum() > 0: bs.append((ww * d).sum() / ww.sum())
    return dict(mean=float(d.mean()), ci95=[float(x) for x in np.percentile(bs, [2.5, 97.5])], ci90=[float(x) for x in np.percentile(bs, [5, 95])])


if __name__ == "__main__":
    out = {}
    for name, root in SESSIONS.items():
        rows = loo_pairs(Session(root), name); res = dict(pairs=len(rows), rms={m: float(np.mean([r[m] for r in rows])) for m in M}, by_group={})
        for g in sorted({r["group"] for r in rows}):
            rr = [r for r in rows if r["group"] == g]; res["by_group"][g] = dict(pairs=len(rr), **{m: float(np.mean([r[m] for r in rr])) for m in M})
        for a, b in (("quad", "box"), ("quad", "dot"), ("box", "dot")):
            s = bootstrap(rows, a, b); s["equivalent_0.05mm"] = bool(s["ci90"][0] > -0.05 and s["ci90"][1] < 0.05); res[f"{a}-{b}"] = s
        out[name] = res
        print(name, res["pairs"], "pairs, LOO RMS (mm):", {m: round(v, 3) for m, v in res["rms"].items()})
        for k in ("quad-box", "quad-dot", "box-dot"):
            s = res[k]; print(f"   {k}: {s['mean']:+.3f} mm, 95% [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}], equivalent: {s['equivalent_0.05mm']}")
    save_json(out, "centres.json")
