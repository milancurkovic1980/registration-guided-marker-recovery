"""Verifiers on the labelled candidates: grouped five-fold cross-validation on the training scenes, then final
models trained on all training scenes and scored on the test scenes and the glare scene.

    python experiments/train_verifiers.py                          all settings and verifiers
    python experiments/train_verifiers.py adaptive "ResNet-18|bin"
    FAST=1 ...                                                     skip the calibrated threshold

Output: work/results/<setting>_final.npz (scores, accept if >= 0), <setting>_final.log, work/models/<setting>/."""
import os, sys, time, numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed

from common import ROOT, WORK
from verifiers import models
from verifiers.metrics import point

WEIGHTS = os.path.join(ROOT, "weights")
MODEL_IMAGE = os.path.join(ROOT, "models", "template", "markers.bmp")
VERIFIERS = [("TM", "bin50"), ("Ellipse", "bin"), ("HOG-SVM", "grey"), ("HOG-SVM", "bin"), ("MobileNetV3-S", "grey"),
             ("MobileNetV3-S", "bin"), ("ResNet-18", "grey"), ("ResNet-18", "bin")]
CNN = dict(weights_dir=WEIGHTS, size=64, epochs=12, batch=32, lr=3e-4, threads=4)
SEEDS_FINAL = 3


def make(name, inp):
    if name == "TM":
        return models.TemplateMatching(MODEL_IMAGE)
    return models.make(name, inp != "grey", True, **(CNN if name in ("MobileNetV3-S", "ResNet-18") else {}))


MODEL_DIR = os.path.join(WORK, "models")


def save_model(m, run, name, inp, seed):
    """CNN: state_dict (.pth) and settings (.json); HOG-SVM: scaler and SVM weights (.npz); ellipse test: limits (.json)."""
    import json
    d = os.path.join(MODEL_DIR, run); os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{name}_{inp}_s{seed}")
    if name in ("MobileNetV3-S", "ResNet-18"):
        import torch
        torch.save({k: v.detach().cpu() for k, v in m.net.state_dict().items()}, stem + ".pth")
        json.dump(dict(arch=m.arch, input=inp, crop=64, resize_to=m.size, grey_to_rgb=True,
                       normalise_mean=[0.485, 0.456, 0.406], normalise_std=[0.229, 0.224, 0.225],
                       output="sigmoid", threshold=m.threshold, seed=seed,
                       note="final score = mean of the sigmoid outputs of all seeds; marker if >= threshold"),
                  open(stem + ".json", "w"), indent=1)
    elif name == "Ellipse":
        from markerdet.verify import ELLIPSE_LIMITS
        json.dump(dict(zip(ELLIPSE_LIMITS, map(float, m.limits)), input=inp), open(os.path.join(d, f"{name}_{inp}.json"), "w"), indent=1)
    elif name == "HOG-SVM":
        import joblib
        joblib.dump(m, stem + ".joblib")
        sc, svc = m.pipe[0], m.pipe[-1]
        np.savez(stem + ".npz", mean=sc.mean_, scale=sc.scale_, coef=svc.coef_.ravel(), intercept=svc.intercept_,
                 C=m.C, threshold=m.threshold, input=inp)


def best_threshold(y, s):
    """Threshold with the highest MCC."""
    from sklearn.metrics import matthews_corrcoef
    u = np.unique(np.round(s, 6))
    cand = np.concatenate([[u[0] - 1], (u[:-1] + u[1:]) / 2, [u[-1] + 1]])
    if len(cand) > 400:
        cand = np.quantile(s, np.linspace(0, 1, 401))
    mcc = [matthews_corrcoef(y, (s >= t).astype(int)) for t in cand]
    return float(cand[int(np.argmax(mcc))])


FAST = os.environ.get("FAST", "0") == "1"


def calibrate(name, inp, X, y, g, seed):
    """Threshold chosen on about 20 % of the setups held out from the training set."""
    if FAST:
        return 0.0
    if name == "TM":
        m = make(name, inp); return best_threshold(y, m.score(X) - m.threshold)
    if name == "Ellipse":
        return 0.0
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=100 + seed).split(np.zeros(len(y)), y, g))
    m = make(name, inp).fit(X[tr], y[tr], g[tr], seed=seed)
    return best_threshold(y[va], m.score(X[va]) - m.threshold)


def fit_score(name, inp, Xtr, ytr, gtr, Xte, seed):
    t = calibrate(name, inp, Xtr, ytr, gtr, seed)
    m = make(name, inp).fit(Xtr, ytr, gtr, seed=seed)
    return m.score(Xte) - m.threshold, t


def run(RUN, sel, log):
    z = np.load(os.path.join(WORK, "data", f"{RUN}_scenes.npz")); lab = z["y"] >= 0
    tr = lab & (z["split"] == "train"); te = lab & (z["split"] == "test")
    y, yt, groups = z["y"][tr], z["y"][te], z["setup"][tr]
    XS = {"grey": z["G"], "bin": z["B"], "bin50": z["B50"]}
    gfile = os.path.join(WORK, "data", f"{RUN}_glare.npz")
    zg = np.load(gfile) if os.path.exists(gfile) else None
    if zg is not None:
        lg = zg["y"] >= 0
        XG = {"grey": zg["G"][lg], "bin": zg["B"][lg], "bin50": zg["B50"][lg]}
    folds = np.full(len(y), -1)
    for k, (_, f) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y, groups)):
        folds[f] = k
    out = os.path.join(WORK, "results", f"{RUN}_final.npz")
    res = dict(np.load(out, allow_pickle=True)) if os.path.exists(out) else {}
    res.update(y_cv=y, groups_cv=groups, folds=folds, photo_cv=z["photo"][tr], y_test=yt,
               groups_test=z["setup"][te], photo_test=z["photo"][te], idx_cv=np.where(tr)[0], idx_test=np.where(te)[0])
    if zg is not None:
        res.update(y_glare=zg["y"][lg], photo_glare=zg["photo"][lg], idx_glare=np.where(lg)[0])
    log(f"{RUN}: train {len(y)} ({int(y.sum())} markers, {len(set(groups))} setups) | test {len(yt)} ({int(yt.sum())} markers)")
    for name, inp in sel:
        key = f"{name}|{inp}"
        if f"testcal|{key}" in res:
            if zg is not None and (f"glare|{key}" not in res or len(res[f"glare|{key}"]) != len(res["y_glare"])):
                nseed = SEEDS_FINAL if name in ("MobileNetV3-S", "ResNet-18") else 1
                sg = 0
                for sd in range(nseed):
                    m = make(name, inp).fit(XS[inp][tr], y, groups, seed=100 + sd)
                    save_model(m, RUN, name, inp, 100 + sd)
                    sg = sg + (m.score(XG[inp]) - m.threshold) / nseed
                res[f"glare|{key}"] = sg
                r = point(res["y_glare"], (sg >= 0).astype(int), sg)
                log(f"glare {key:21s} MCC {r['MCC']:.3f} TP {r['TP']} FP {r['FP']} FN {r['FN']}")
                np.savez(out, **res)
            continue
        X, Xt = XS[inp][tr], XS[inp][te]
        cnn = name in ("MobileNetV3-S", "ResNet-18")
        t0 = time.time(); s = np.zeros(len(y)); sc = np.zeros(len(y))
        jobs = [(X[folds != k], y[folds != k], groups[folds != k], X[folds == k], k) for k in range(5)]
        parts = Parallel(n_jobs=1 if cnn else 5)(delayed(fit_score)(name, inp, *j) for j in jobs)
        for k, (p, t) in enumerate(parts):
            s[folds == k] = p; sc[folds == k] = p - t
        res[f"cv|{key}"] = s; res[f"cvcal|{key}"] = sc
        for tag, v in (("cv   ", s), ("cvcal", sc)):
            r = point(y, (v >= 0).astype(int), v)
            log(f"{tag} {key:22s} MCC {r['MCC']:.3f} AUC {r.get('AUC', 0):.3f} TP {r['TP']} FP {r['FP']} FN {r['FN']}  ({time.time() - t0:.0f} s)")
        nseed = SEEDS_FINAL if cnn else 1
        tcal = calibrate(name, inp, X, y, groups, 0)
        st = 0; sg = 0
        for sd in range(nseed):
            m = make(name, inp).fit(X, y, groups, seed=100 + sd)
            save_model(m, RUN, name, inp, 100 + sd)
            st = st + (m.score(Xt) - m.threshold) / nseed
            if zg is not None:
                sg = sg + (m.score(XG[inp]) - m.threshold) / nseed
        if zg is not None:
            res[f"glare|{key}"] = sg
            r = point(res["y_glare"], (sg >= 0).astype(int), sg)
            log(f"glare {key:21s} MCC {r['MCC']:.3f} TP {r['TP']} FP {r['FP']} FN {r['FN']}")
        res[f"test|{key}"] = st; res[f"testcal|{key}"] = st - tcal; res[f"thr|{key}"] = tcal
        r = point(yt, (st - tcal >= 0).astype(int), st)
        log(f"tstcal {key:21s} MCC {r['MCC']:.3f} TP {r['TP']} FP {r['FP']} FN {r['FN']}")
        single, batch = models.time_per_crop(m, Xt)
        res[f"cost|{key}"] = np.array([single, batch, m.n_params()])
        r = point(yt, (st >= 0).astype(int), st)
        log(f"test {key:22s} MCC {r['MCC']:.3f} AUC {r.get('AUC', 0):.3f} TP {r['TP']} FP {r['FP']} FN {r['FN']}  "
            f"{single:.2f} ms/candidate (CPU, 1 thread), {m.n_params()} parameters")
        np.savez(out, **res)


if __name__ == "__main__":
    import torch
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    runs = [sys.argv[1]] if len(sys.argv) > 1 else ["fixed", "adaptive", "adaptive_smoothed"]
    sel = [tuple(s.split("|")) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else VERIFIERS
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    for RUN in runs:
        lf = open(os.path.join(WORK, "results", f"{RUN}_final.log"), "a")
        def log(msg, lf=lf):
            print(msg, flush=True); lf.write(msg + "\n"); lf.flush()
        run(RUN, sel, log)
    print("DONE")
