"""Corners of the binarized image (Hessian-based Harris-type response, 7 x 7 non-maximum suppression)."""
import numpy as np
from scipy import ndimage
from .imaging import box_blur3


def detect_corners(binary):
    im = box_blur3(binary, 1.8).astype(np.float64)
    H, W = im.shape
    s = int(1.8 * 4)
    c = im[s:H - s, s:W - s]
    Dxx = im[s:H - s, s + 1:W - s + 1] + im[s:H - s, s - 1:W - s - 1] - 2.0 * c
    Dyy = im[s + 1:H - s + 1, s:W - s] + im[s - 1:H - s - 1, s:W - s] - 2.0 * c
    Dxy = 0.25 * (im[s + 1:H - s + 1, s + 1:W - s + 1] + im[s - 1:H - s - 1, s - 1:W - s - 1]
                  - im[s + 1:H - s + 1, s - 1:W - s - 1] - im[s - 1:H - s - 1, s + 1:W - s + 1])
    tr = Dxx + Dyy; det = Dxx * Dyy - Dxy * Dxy; disc = tr * tr - 4.0 * det
    ok = disc >= 0.0
    root = np.sqrt(np.where(ok, disc, 0.0))
    l1 = 0.5 * (tr + root); l2 = 0.5 * (tr - root)
    ok &= ~(np.abs(l1 - l2) < 1e-10)
    ok &= (l1 > 1e-3) & (l2 > 1e-3) & (np.maximum(l1, l2) < 8 * np.minimum(l1, l2))
    Hm = np.zeros((H, W)); Hm[s:H - s, s:W - s] = np.where(ok, np.maximum(0.0, l1 * l2 - 0.04 * tr * tr), 0.0)
    mn = Hm.min(); mx = Hm.max()
    if abs(mn) > 1e-10:
        Hm -= mn
    mx -= mn
    f = 1.0 / mx if abs(mx) > 1e-10 else 0.0
    Hm *= f
    # strict maximum in a 7 x 7 window, response >= 0.02, border of 3 pixels
    fp = np.ones((7, 7), bool); fp[3, 3] = False
    nb = ndimage.maximum_filter(Hm, footprint=fp, mode="nearest")
    m = (Hm >= 0.02) & (Hm > nb)
    m[:3, :] = False; m[H - 3:, :] = False; m[:, :3] = False; m[:, W - 3:] = False
    ys, xs = np.nonzero(m)
    return np.stack([xs, ys], 1)
