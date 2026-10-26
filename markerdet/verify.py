"""Trained verifiers that decide which candidates are markers.

    v = load_verifier("models/adaptive/HOG-SVM_bin")     # or Ellipse_bin, or ResNet-18_bin (needs PyTorch)
    is_marker = v.accept(det.bin_crops)                 # det.grey_crops for verifiers on grey crops"""
import glob, json, os
import numpy as np


# HOG of 64 x 64 crops: 8 x 8 px cells, 9 orientation bins, 2 x 2 cell blocks, L2-Hys -> 1764 values
def hog_features(X, chunk=512):
    X = np.asarray(X)
    return np.concatenate([_hog(X[i:i + chunk]) for i in range(0, len(X), chunk)]) if len(X) else np.zeros((0, 1764), np.float32)


def _hog(X):
    I = np.asarray(X, np.float32)
    gx = np.zeros_like(I); gy = np.zeros_like(I)
    gx[:, :, 1:-1] = I[:, :, 2:] - I[:, :, :-2]; gy[:, 1:-1, :] = I[:, 2:, :] - I[:, :-2, :]
    mag = np.hypot(gx, gy); ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    pos = ang / 20.0 - 0.5; b0 = np.floor(pos).astype(int); w1 = pos - b0; b0 %= 9; b1 = (b0 + 1) % 9
    n = len(I); H = np.zeros((n, 8, 8, 9), np.float32)
    cy, cx = np.meshgrid(np.arange(64) // 8, np.arange(64) // 8, indexing="ij")
    idx_n = np.repeat(np.arange(n), 64 * 64)
    for b, w in ((b0, 1 - w1), (b1, w1)):
        np.add.at(H, (idx_n, np.tile(cy.ravel(), n), np.tile(cx.ravel(), n), b.ravel()), (mag * w).ravel())
    blocks = np.concatenate([H[:, :-1, :-1], H[:, :-1, 1:], H[:, 1:, :-1], H[:, 1:, 1:]], axis=3)
    v = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + 1e-6)
    v = np.minimum(v, 0.2); v = v / np.sqrt((v ** 2).sum(-1, keepdims=True) + 1e-6)
    return v.reshape(n, -1).astype(np.float32)


# Ellipse test of the white dot on binarized 64 x 64 crops
ELLIPSE_LIMITS = ("amin", "amax", "dmax", "qmin", "fill", "omax")


def ellipse_features(X):
    """Per crop: area of the white region covering the centre, distance of its centroid from the centre, axis ratio
    of its moment ellipse, its area / ellipse area, and the white area outside it (areas as fractions of the crop)."""
    from scipy import ndimage
    X = np.asarray(X); n, h, w = X.shape if X.ndim == 3 else (0, 64, 64)
    F = np.zeros((n, 5)); yy, xx = np.mgrid[0:h, 0:w]; c0 = ((w - 1) / 2, (h - 1) / 2)
    for i in range(n):
        white = X[i] > 127
        lab, k = ndimage.label(white, np.ones((3, 3), int))
        core = lab[h * 3 // 8:h * 5 // 8, w * 3 // 8:w * 5 // 8]; ids = np.unique(core[core > 0])
        if len(ids) == 0:
            F[i] = (0, 1, 0, 0, white.mean()); continue
        j = ids[np.argmax(ndimage.sum(white, lab, ids))]; m = lab == j; a = m.sum()
        x, y = xx[m], yy[m]
        cov = np.cov(np.vstack([x, y])) if a > 2 else np.eye(2) * 1e-3
        ev = np.clip(np.linalg.eigvalsh(cov), 1e-6, None)
        F[i] = (a / (h * w), np.hypot(x.mean() - c0[0], y.mean() - c0[1]) / w, np.sqrt(ev[0] / ev[1]),
                a / (4 * np.pi * np.sqrt(ev[0] * ev[1])), (white.sum() - a) / (h * w))
    return F


def ellipse_accept(F, p):
    """F: (n, 5) features, p: (m, 6) limits -> (m, n) decisions."""
    a, d, q, f, o = np.asarray(F, float).T; p = np.atleast_2d(p)
    return ((a >= p[:, :1]) & (a <= p[:, 1:2]) & (d <= p[:, 2:3]) & (q >= p[:, 3:4]) &
            (np.abs(f - 1) <= p[:, 4:5]) & (o <= p[:, 5:6]))


class EllipseVerifier:
    def __init__(self, path):
        z = json.load(open(path)); self.limits = np.array([z[k] for k in ELLIPSE_LIMITS], float)
        self.input = "bin"; self.threshold = 0.0

    def score(self, crops):
        return np.where(ellipse_accept(ellipse_features(crops), self.limits)[0], 1.0, -1.0)

    def accept(self, crops):
        return self.score(crops) >= self.threshold


class HOGSVMVerifier:
    def __init__(self, npz):
        z = np.load(npz)
        self.mean, self.scale, self.coef = z["mean"], z["scale"], z["coef"]
        self.intercept = float(np.ravel(z["intercept"])[0]); self.threshold = float(z["threshold"])
        self.input = str(z["input"])

    def score(self, crops):
        f = (hog_features(crops) - self.mean) / self.scale
        return f @ self.coef + self.intercept

    def accept(self, crops):
        return self.score(crops) >= self.threshold


class CNNVerifier:
    ARCH = {"ResNet-18": "resnet18", "MobileNetV3-S": "mobilenet_v3_small"}

    def __init__(self, files):
        import torch, torchvision
        self.nets = []
        for f in files:
            meta = json.load(open(os.path.splitext(f)[0] + ".json"))
            net = getattr(torchvision.models, meta["arch"])(weights=None)
            if meta["arch"] == "resnet18":
                net.fc = torch.nn.Linear(net.fc.in_features, 1)
            else:
                net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, 1)
            net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True)); net.eval()
            self.nets.append(net)
        self.meta = meta; self.input = meta["input"]; self.threshold = float(meta["threshold"])

    def _tensor(self, X):
        import torch, torch.nn.functional as F
        m = self.meta; s = int(m["resize_to"])
        t = torch.from_numpy(np.ascontiguousarray(X)).float().div(255.0).unsqueeze(1)
        t = F.interpolate(t, size=(s, s), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        mean = torch.tensor(m["normalise_mean"]).view(1, 3, 1, 1); std = torch.tensor(m["normalise_std"]).view(1, 3, 1, 1)
        return (t - mean) / std

    def score(self, crops, bs=256):
        import torch
        out = np.zeros(len(crops))
        with torch.no_grad():
            for net in self.nets:
                for s in range(0, len(crops), bs):
                    out[s:s + bs] += torch.sigmoid(net(self._tensor(crops[s:s + bs])).squeeze(1)).numpy() / len(self.nets)
        return out

    def accept(self, crops):
        return self.score(crops) >= self.threshold


def load_verifier(stem):
    """stem: path without seed and extension, e.g. models/adaptive/ResNet-18_bin."""
    if os.path.basename(stem).startswith("Ellipse"):
        return EllipseVerifier(stem + ".json")
    files = sorted(glob.glob(stem + "_s*.npz")) or sorted(glob.glob(stem + "_s*.pth"))
    if not files:
        raise FileNotFoundError(f"no model files for {stem}")
    return HOGSVMVerifier(files[0]) if files[0].endswith(".npz") else CNNVerifier(files)
