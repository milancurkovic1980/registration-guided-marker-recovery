"""Registration with a given number of markers and the expected target registration error (Table and Figure of the
number of markers in the article).

    python experiments/scanner/number_of_markers.py

View pairs of the first session with at least 12 corresponding markers (quadrilateral centres) are aligned with
300 random subsets of 3 to 10 markers, and the RMS distance is measured on the remaining markers. The prediction
uses the TRE model of Fitzpatrick et al. (1998), with the FLE estimated from the residuals of the full alignments,
plus the localisation error of the left-out marker."""
import itertools
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import SESSIONS, RESULTS, Session, save_json
from recovery import kabsch, match_markers

M = ("quad", "box", "dot")


def tre_factor(F, r):
    c = F.mean(0); X = F - c; _, _, Vt = np.linalg.svd(X, full_matrices=False); s = 0
    for ax in Vt:
        f2 = np.mean(np.sum(X ** 2, 1) - (X @ ax) ** 2); y = r - c; s += (np.sum(y ** 2) - (y @ ax) ** 2) / f2
    return 1 + s / 3


ses = Session(SESSIONS["session1"]); pairs = []
for p in ses.projects():
    vs = ses.views_of(p); data = {v: {m: ses.markers3d(v, m) for m in M} for v in vs}
    for vi, vj in itertools.combinations(vs, 2):
        cons = lambda v: {k: np.mean([data[v][m][k][1] for m in M], 0) for k in set.intersection(*[set(data[v][m]) for m in M])}
        mt = match_markers(cons(vi), cons(vj))
        if len(mt) >= 4:
            pairs.append((np.array([data[vi]["quad"][a][1] for a in mt]), np.array([data[vj]["quad"][b][1] for b in mt.values()])))
num = den = 0
for P, Q in pairs:
    n = len(P); R, t = kabsch(P, Q); num += np.sum((P @ R.T + t - Q) ** 2); den += n * (1 - 2 / n)
FLE = float(np.sqrt(num / den)); sel = [(P, Q) for P, Q in pairs if len(P) >= 12]
rng = np.random.default_rng(0); out = dict(pairs=len(sel), fle_mm=FLE, k={})
for k in range(3, 11):
    mm, pm, p90, f1, f2 = [], [], [], [], []
    for P, Q in sel:
        n = len(P); me, pe = [], []
        for _ in range(300):
            idx = rng.permutation(n); tr, te = idx[:k], idx[k:]
            R, t = kabsch(P[tr], Q[tr]); me.append(np.sqrt(np.mean(np.sum((P[te] @ R.T + t - Q[te]) ** 2, 1))))
            pe.append(np.sqrt(np.mean([FLE ** 2 * (1 + tre_factor(P[tr], P[j]) / k) for j in te])))
        me = np.array(me); mm.append(np.median(me)); pm.append(np.median(pe)); p90.append(np.percentile(me, 90)); f1.append((me > 1).mean()); f2.append((me > 2).mean())
    out["k"][k] = dict(median=float(np.mean(mm)), p90=float(np.mean(p90)), predicted=float(np.mean(pm)), above_1mm=float(np.mean(f1)), above_2mm=float(np.mean(f2)))
    print(k, {a: round(b, 3) for a, b in out["k"][k].items()}, flush=True)
print("pairs", len(sel), "FLE", round(FLE, 3))
save_json(out, "number_of_markers.json")
K = sorted(out["k"]); g = lambda f: [out["k"][k][f] for k in K]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
ax[0].plot(K, g("median"), "o-", label="measured, median"); ax[0].plot(K, g("p90"), "s--", mfc="white", label="measured, 90th percentile")
ax[0].plot(K, g("predicted"), "^-", label="predicted (TRE model)"); ax[0].set_ylabel("RMS on the other markers (mm)")
ax[1].plot(K, [100 * x for x in g("above_1mm")], "o-", label="> 1 mm"); ax[1].plot(K, [100 * x for x in g("above_2mm")], "s-", label="> 2 mm"); ax[1].set_ylabel("Alignments (%)")
for a in ax: a.set_xlabel("Markers used for the alignment"); a.legend(frameon=False, fontsize=7); a.set_xticks(K)
plt.tight_layout(); plt.savefig(f"{RESULTS}/number_of_markers.png", dpi=200)
