"""Image loading, binarization and smoothing."""
import numpy as np
from numba import njit, prange


def load_rgb(path):
    """RGB image (H x W x 3, uint8), rotated according to the EXIF orientation."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def load_grey(path):
    """Grey image uint8(0.3 R + 0.59 G + 0.11 B) from a file or an RGB array."""
    rgb = path if isinstance(path, np.ndarray) else load_rgb(path)
    if rgb.ndim == 2:
        return rgb.astype(np.uint8).copy()
    r = rgb[..., 0].astype(np.float32); g = rgb[..., 1].astype(np.float32); b = rgb[..., 2].astype(np.float32)
    v = (np.float32(0.3) * r + np.float32(0.59) * g) + np.float32(0.11) * b
    return v.astype(np.uint8)


def norm_palette_u8(img):
    """Stretch to 0..255: subtract the minimum, multiply by floor(255 / range)."""
    mn = int(img.min()); mx = int(img.max())
    out = img.astype(np.int32) - mn
    rng = mx - mn
    f = 255 // rng if rng > 0 else 0
    return ((out * f) & 0xFF).astype(np.uint8)


_PCT = np.array([np.float32(0.7 + 0.2 / (1.0 + np.exp(-(i - 127) / 30.0))) for i in range(256)], np.float32)


@njit(parallel=True, cache=True)
def _binarize(pad, H, W, R, pct, out):
    hl = (2 * R + 1) * (2 * R + 1)
    for j in prange(H):
        hist = np.zeros(256, np.int64)
        s = 0
        for jj in range(j, j + 2 * R + 1):
            for ii in range(0, 2 * R + 1):
                v = pad[jj, ii]; hist[v] += 1; s += v
        for i in range(W):
            if i > 0:
                for k in range(2 * R + 1):
                    vl = pad[j + k, i - 1]; vr = pad[j + k, i + 2 * R]
                    hist[vl] -= 1; hist[vr] += 1; s += vr - vl
            avg = s // hl
            med = 0; acc = 0
            while acc < hl // 2:
                acc += hist[med]; med += 1
            if med != 0:
                med -= 1
            # the first column of a row uses pixel (R, j) of the padded image
            p = pad[j, R] if i == 0 else pad[j + R, i + R]
            if np.float32(p) < pct[med] * np.float32(avg) and p < avg:
                out[j, i] = 0
            else:
                out[j, i] = 255


def image_binarization(img, R):
    """Local binarization with a (2R+1) x (2R+1) window."""
    H, W = img.shape
    if R <= 0 or R >= W / 2 or R >= H / 2:
        raise ValueError("binarization: window radius is too large")
    pad = np.pad(img, R, mode="edge")
    out = np.empty((H, W), np.uint8)
    _binarize(pad, H, W, R, _PCT, out)
    return out


def box_blur3(img, sigma):
    """Three passes of a box filter (approximate Gaussian)."""
    r = max(1, int(np.round(np.float32(sigma) * np.float32(0.57735))))
    im = img.astype(np.int64)
    k = 2 * r + 1
    for _ in range(3):
        for axis in (1, 0):
            p = np.pad(im, [(0, 0), (r, r)] if axis == 1 else [(r, r), (0, 0)], mode="edge")
            c = np.cumsum(p, axis=axis)
            c = np.concatenate([np.zeros_like(c.take([0], axis=axis)), c], axis=axis)
            n = im.shape[axis]
            s = c.take(np.arange(k, k + n), axis=axis) - c.take(np.arange(0, n), axis=axis)
            im = s // k
    return im.astype(img.dtype)


def adapt_image_binarization(img):
    """Window radius = 5 x median diameter of the dark components of a first binarization."""
    from .components import connected_components
    H, W = img.shape
    first = image_binarization(img, min(W, H) // 30)
    _, comps = connected_components(first, 10)
    sizes = sorted((len(c), i) for i, c in enumerate(comps) if first[c[0, 1], c[0, 0]] != 255)
    n = len(sizes)
    diam = []
    for size, i in sizes[n // 5: 4 * n // 5]:
        c = comps[i]
        dx = int(c[:, 0].max() - c[:, 0].min()); dy = int(c[:, 1].max() - c[:, 1].min())
        if max(dx, dy) < 3 * min(dx, dy):
            diam.append(int((dx + dy) / 2.0))
    if not diam:
        return first
    diam.sort()
    return image_binarization(img, 5 * diam[len(diam) // 2])
