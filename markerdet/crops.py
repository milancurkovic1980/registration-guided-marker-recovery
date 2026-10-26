"""Rectification of a candidate quadrilateral into a square crop (inverse bilinear mapping)."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _inside(qx, qy, px, py):
    cnt = 0
    for k in range(4):
        x1 = qx[k]; y1 = qy[k]; x2 = qx[(k + 1) % 4]; y2 = qy[(k + 1) % 4]
        if py > min(y1, y2) and py <= max(y1, y2) and px <= max(x1, x2) and y1 != y2:
            xin = (py - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or px <= xin:
                cnt += 1
    return cnt % 2 == 1


@njit(cache=True)
def _map(mu, x, y, x0, y0, x1, y1, x2, y2, x3, y3, S, thr):
    """Returns (ok, skip_pixel, sx, sy)."""
    cxl = x - x0 - mu * (x1 - x0); cxr = (x3 + mu * (x2 - x3) - x0 - mu * (x1 - x0))
    cyl = y - y0 - mu * (y1 - y0); cyr = (y3 + mu * (y2 - y3) - y0 - mu * (y1 - y0))
    if abs(cxr) < 1e-10 and abs(cyr) < 1e-6:
        return False, True, 0.0, 0.0
    n1x = S - 1.0; n2x = S - 1.0; n2y = S - 1.0; n3y = S - 1.0
    if abs(cxr) > thr:
        d = cxl / cxr
        ax = mu * n1x; ay = 0.0
        bx = mu * n2x; by = n3y + mu * (n2y - n3y)
    else:
        d = cyl / cyr
        ax = mu * n1x; ay = 0.0
        bx = mu * n2x; by = n3y + mu * (n2y - n3y)
    sx = ax + d * (bx - ax); sy = ay + d * (by - ay)
    return True, False, sx, sy


@njit(cache=True)
def _stretch(src, qx, qy, S, out):
    H, W = src.shape
    x0 = qx[0]; y0 = qy[0]; x1 = qx[1]; y1 = qy[1]; x2 = qx[2]; y2 = qy[2]; x3 = qx[3]; y3 = qy[3]
    per = 0.0
    for k in range(4):
        dx = qx[k] - qx[(k + 1) % 4]; dy = qy[k] - qy[(k + 1) % 4]
        per += math.sqrt(dx * dx + dy * dy)
    rb = max(1, int((2 * S + 2 * S) / per))
    left = max(0, int(min(min(x0, x1), min(x2, x3)))); right = min(W - 1, int(max(max(x0, x1), max(x2, x3))) + 1)
    up = max(0, int(min(min(y0, y1), min(y2, y3)))); down = min(H - 1, int(max(max(y0, y1), max(y2, y3))) + 1)
    a = (y1 - y0) * (x2 - x3 - x1 + x0) - (x1 - x0) * (y2 - y3 - y1 + y0)
    for j in range(up, down):
        for i in range(left, right):
            x = float(i); y = float(j)
            if not _inside(qx, qy, x, y):
                continue
            b = (x - x0) * (y2 - y3 - y1 + y0) - (x1 - x0) * (y3 - y0) - (y - y0) * (x2 - x3 - x1 + x0) + (y1 - y0) * (x3 - x0)
            c = (x - x0) * (y3 - y0) - (y - y0) * (x3 - x0)
            if abs(a) < 1e-6:
                if abs(b) < 1e-6:
                    continue
                mu1 = -c / b; mu2 = mu1
            else:
                r = math.sqrt(b * b - 4 * a * c) if b * b - 4 * a * c >= 0 else np.nan
                mu1 = (-b - r) / (2 * a); mu2 = (-b + r) / (2 * a)
            skip = False
            for t in range(2):
                if t == 1 and not (mu1 != mu2):
                    break
                mu = mu1 if t == 0 else mu2
                ok, sk, sx, sy = _map(mu, x, y, x0, y0, x1, y1, x2, y2, x3, y3, S, 1e-10 if t == 0 else 1e-6)
                if sk:
                    skip = True
                    break
                if sx >= 0 and sx < S and sy >= 0 and sy < S:
                    v = src[j, i]; ix = int(sx); iy = int(sy)
                    for jj in range(iy - rb, iy + rb + 1):
                        for ii in range(ix - rb, ix + rb + 1):
                            if jj >= 0 and jj < S and ii >= 0 and ii < S:
                                out[jj, ii] = v
            if skip:
                continue


def rectify(src, quad, size=64):
    """size x size crop of image src (uint8) for the quadrilateral quad = [(x, y)] * 4 (P0..P3)."""
    q = np.asarray(quad, np.float64)
    out = np.zeros((size, size), np.uint8)
    _stretch(np.ascontiguousarray(src), q[:, 0].copy(), q[:, 1].copy(), size, out)
    return out
