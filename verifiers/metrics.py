import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score


def confusion(y, d):
    y = np.asarray(y).astype(int); d = np.asarray(d).astype(int)
    tp = int(((d == 1) & (y == 1)).sum()); fp = int(((d == 1) & (y == 0)).sum())
    fn = int(((d == 0) & (y == 1)).sum()); tn = int(((d == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def point(y, d, s=None):
    tp, fp, fn, tn = confusion(y, d)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    F1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    out = dict(TP=tp, FP=fp, FN=fn, TN=tn, P=P, R=R, F1=F1,
               MCC=matthews_corrcoef(y, d) if len(set(d)) > 1 or len(set(y)) > 1 else 0.0)
    if s is not None and len(set(y)) > 1:
        out["AUC"] = roc_auc_score(y, s); out["AP"] = average_precision_score(y, s)
    return out


def group_bootstrap(y, d, groups, stat="MCC", s=None, B=2000, seed=0):
    """95% percentile CI, resampling whole photographs (clusters)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); idx = {u: np.where(groups == u)[0] for u in ug}
    vals = []
    for _ in range(B):
        pick = np.concatenate([idx[u] for u in rng.choice(ug, len(ug), replace=True)])
        yy = y[pick]
        if len(set(yy)) < 2:
            continue
        r = point(yy, d[pick], None if s is None else s[pick])
        vals.append(r[stat])
    vals = np.array(vals, float)
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def paired_permutation(diff, B=200000, seed=0):
    """Two-sided paired permutation test over photographs: diff = per-photograph difference of two methods."""
    d = np.asarray(diff, float); obs = abs(d.sum())
    if obs == 0:
        return 1.0
    s = np.random.default_rng(seed).choice([-1, 1], size=(B, len(d)))
    return float((np.abs(s @ d) >= obs - 1e-9).mean())
