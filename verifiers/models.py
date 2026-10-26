"""Verifiers with a common interface: fit(X, y, groups, seed), score(X) (larger = more marker-like), threshold
(accept if score >= threshold), n_params(). TM (template matching), HOG-SVM, MobileNetV3-Small and ResNet-18."""
import os, time
import numpy as np
import cv2

# ----------------------------------------------------------------------------- augmentation
def scale(a, s):
    """zoom the 64x64 crop by factor s, padding with the median of its 4 px border."""
    n = max(8, int(round(64 * s)))
    b = cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA if n < 64 else cv2.INTER_LINEAR)
    if n < 64:
        ring = np.concatenate([a[:4].ravel(), a[-4:].ravel(), a[:, :4].ravel(), a[:, -4:].ravel()])
        p = 64 - n; t = p // 2
        b = cv2.copyMakeBorder(b, t, p - t, t, p - t, cv2.BORDER_CONSTANT, value=int(np.median(ring)))
    elif n > 64:
        o = (n - 64) // 2; b = b[o:o + 64, o:o + 64]
    return b


def rotate(a, ang):
    M = cv2.getRotationMatrix2D((31.5, 31.5), ang, 1.0)
    return cv2.warpAffine(a, M, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def augment(a, rng, binary, scale_aug=True, s_range=(0.45, 1.1)):
    """random flip, rotation, scale and (grey input only) brightness/contrast change."""
    if scale_aug:
        a = scale(a, rng.uniform(*s_range))
    if rng.random() < 0.5:
        a = a[:, ::-1]
    if rng.random() < 0.5:
        a = a[::-1]
    a = rotate(np.ascontiguousarray(a), rng.uniform(-180, 180))
    if binary:
        a = np.where(a > 127, 255, 0).astype(np.uint8)
    else:
        c, b = rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3)
        m = a.mean(); a = np.clip((a.astype(np.float32) - m) * c + m * b, 0, 255).astype(np.uint8)
    return a


# ----------------------------------------------------------------------------- TM
class TemplateMatching:
    """Input: 50 x 50 crops of the binarized image."""
    name = "TM"
    def __init__(self, model_image):
        from . import tm
        self.tm = tm
        self.models, self.size = tm.load_models(model_image)
        self.threshold = -tm.threshold(self.models)
    def fit(self, X, y, groups=None, seed=0):
        return self
    def score(self, X):
        return -self.tm.errors(X, self.models)
    def n_params(self):
        return 0

# ----------------------------------------------------------------------------- ellipse test
class EllipseTest:
    """Input: binarized 64 x 64 crops. Six limits on the ellipse features chosen by grid search for the best MCC."""
    name = "Ellipse"
    GRID = ((0.01, 0.02, 0.04, 0.06, 0.08), (0.40, 0.55, 0.70, 1.0), (0.05, 0.10, 0.15, 0.20, 0.30),
            (0.2, 0.3, 0.4, 0.5, 0.6, 0.7), (0.05, 0.10, 0.15, 0.20, 0.30, 0.5), (0.02, 0.05, 0.10, 0.20, 0.30, 0.5))
    def __init__(self):
        self.threshold = 0.0
    def fit(self, X, y, groups=None, seed=0):
        import itertools
        from markerdet.verify import ellipse_features, ellipse_accept
        P = np.array(list(itertools.product(*self.GRID))); F = ellipse_features(X); yb = np.asarray(y).astype(bool)
        D = ellipse_accept(F, P)
        tp = (D & yb).sum(1).astype(float); fp = (D & ~yb).sum(1).astype(float)
        fn = (~D & yb).sum(1).astype(float); tn = (~D & ~yb).sum(1).astype(float)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0)
        self.limits = P[int(np.argmax(mcc))]
        return self
    def score(self, X):
        from markerdet.verify import ellipse_features, ellipse_accept
        return np.where(ellipse_accept(ellipse_features(X), self.limits)[0], 1.0, -1.0)
    def n_params(self):
        return 6

# ----------------------------------------------------------------------------- HOG + SVM
class HOGSVM:
    name = "HOG-SVM"
    def __init__(self, binary, scale_aug=True, n_aug=4, Cs=(0.0003, 0.001, 0.003, 0.01)):
        self.binary, self.scale_aug, self.n_aug, self.Cs = binary, scale_aug, n_aug, Cs
        self.threshold = 0.0
    @staticmethod
    def feats(X):
        from markerdet.verify import hog_features
        return hog_features(X)

    def _augmented(self, X, y, rng):
        XA, yA = [X], [y]
        for _ in range(self.n_aug):
            XA.append(np.stack([augment(a, rng, self.binary, self.scale_aug) for a in X])); yA.append(y)
        return np.concatenate(XA), np.concatenate(yA)
    def fit(self, X, y, groups=None, seed=0):
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import GroupKFold, StratifiedKFold
        from sklearn.metrics import matthews_corrcoef
        rng = np.random.default_rng(seed)
        # C chosen by an inner grouped split of the training data
        if groups is not None and len(set(groups)) >= 3:
            splits = list(GroupKFold(3).split(X, y, groups))
        else:
            splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
        best = (-2, self.Cs[0])
        for C in self.Cs:
            s = []
            for tr, va in splits:
                Xa, ya = self._augmented(X[tr], y[tr], rng)
                p = make_pipeline(StandardScaler(), LinearSVC(C=C, class_weight="balanced", dual=False, max_iter=5000))
                p.fit(self.feats(Xa), ya); s.append(matthews_corrcoef(y[va], p.predict(self.feats(X[va]))))
            best = max(best, (np.mean(s), C))
        self.C = best[1]
        Xa, ya = self._augmented(X, y, rng)
        self.pipe = make_pipeline(StandardScaler(), LinearSVC(C=self.C, class_weight="balanced", dual=False, max_iter=5000))
        self.pipe.fit(self.feats(Xa), ya)
        return self
    def score(self, X):
        return self.pipe.decision_function(self.feats(X))
    def n_params(self):
        svc = self.pipe[-1]; return int(svc.coef_.size + svc.intercept_.size)

# ----------------------------------------------------------------------------- CNNs
WEIGHT_FILES = {"mobilenet_v3_small": "mobilenet_v3_small-047dcff4.pth", "resnet18": "resnet18-f37072fd.pth"}


class TorchNet:
    def __init__(self, arch, binary, scale_aug=True, weights_dir="weights", size=96, epochs=15,
                 batch=32, lr=3e-4, threads=2, require_pretrained=True):
        self.arch, self.binary, self.scale_aug = arch, binary, scale_aug
        self.weights = os.path.join(weights_dir, WEIGHT_FILES[arch])
        self.size, self.epochs, self.batch, self.lr, self.threads = size, epochs, batch, lr, threads
        self.require_pretrained = require_pretrained
        self.name = {"mobilenet_v3_small": "MobileNetV3-S", "resnet18": "ResNet-18"}[arch]
        self.threshold = 0.5

    def _build(self):
        import torch, torchvision
        net = getattr(torchvision.models, self.arch)(weights=None)
        if os.path.exists(self.weights):
            net.load_state_dict(torch.load(self.weights, map_location="cpu", weights_only=True))
        else:   # the same ImageNet weights (IMAGENET1K_V1), downloaded by torchvision
            net = getattr(torchvision.models, self.arch)(weights="IMAGENET1K_V1")
        self.pretrained = True
        if self.arch == "resnet18":
            net.fc = torch.nn.Linear(net.fc.in_features, 1)
        else:
            net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
        return net

    def _tensor(self, X):
        import torch, torch.nn.functional as F
        t = torch.from_numpy(np.ascontiguousarray(X)).float().div(255.0).unsqueeze(1)
        t = F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        t = t.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (t - mean) / std

    def fit(self, X, y, groups=None, seed=0):
        import torch
        torch.set_num_threads(self.threads); torch.manual_seed(seed); rng = np.random.default_rng(seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = True
        self.net = self._build().to(dev); net = self.net
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * int(np.ceil(len(X) / self.batch))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.15)
        pos_w = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w.to(dev))
        yt = torch.from_numpy(y.astype(np.float32)).to(dev)
        for ep in range(self.epochs):
            net.train(); perm = rng.permutation(len(X))
            for s in range(0, len(X), self.batch):
                idx = perm[s:s + self.batch]
                xb = self._tensor(np.stack([augment(X[i], rng, self.binary, self.scale_aug) for i in idx])).to(dev)
                opt.zero_grad(); loss = crit(net(xb).squeeze(1), yt[idx]); loss.backward(); opt.step(); sch.step()
        net.eval()
        return self

    def score(self, X, bs=256):
        import torch
        dev = next(self.net.parameters()).device
        out = []
        with torch.no_grad():
            for s in range(0, len(X), bs):
                out.append(torch.sigmoid(self.net(self._tensor(X[s:s + bs]).to(dev)).squeeze(1)).cpu().numpy())
        return np.concatenate(out)

    def to_cpu(self):
        self.net = self.net.cpu()
        return self

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))


def make(name, binary, scale_aug=True, **kw):
    if name == "Ellipse":
        return EllipseTest()
    if name == "HOG-SVM":
        return HOGSVM(binary, scale_aug)
    if name == "MobileNetV3-S":
        return TorchNet("mobilenet_v3_small", binary, scale_aug, **kw)
    if name == "ResNet-18":
        return TorchNet("resnet18", binary, scale_aug, **kw)
    raise ValueError(name)


def time_per_crop(model, X, threads=1, reps=3):
    """CPU time per crop in ms for single crops and for a batch of 64 (one thread)."""
    try:
        import torch; torch.set_num_threads(threads)
    except ImportError:
        pass
    if hasattr(model, "to_cpu"):
        model.to_cpu()                       # cost is always measured on one CPU thread
    x1, x64 = X[:1], X[:64]
    model.score(x64)  # warm-up
    t = time.perf_counter()
    for _ in range(reps * 20):
        model.score(x1)
    single = (time.perf_counter() - t) / (reps * 20) * 1000
    t = time.perf_counter()
    for _ in range(reps):
        model.score(x64)
    batch = (time.perf_counter() - t) / (reps * len(x64)) * 1000
    return single, batch
