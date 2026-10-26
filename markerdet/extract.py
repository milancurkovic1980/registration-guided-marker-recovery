"""Candidate extraction: quadrilaterals of dark components with four corners, geometric checks and an appearance test."""
from dataclasses import dataclass
import math
import numpy as np
from numba import njit

from .imaging import load_grey, norm_palette_u8, image_binarization, adapt_image_binarization, box_blur3
from .components import connected_components
from .harris import detect_corners


@dataclass
class Params:
    max_side_ratio: float = 2.0         # longest / shortest side of the quadrilateral
    max_diag_ratio: float = 2.0         # longer / shorter diagonal
    max_adjacent_ratio: float = 5.0     # ratio of the sides P0P1 and P1P2
    texture_corner_factor: float = 2.0  # component rejected as texture if corners > factor * sqrt(area)
    small_fallback: bool = False        # quadrilateral from extreme points when a component has fewer than 4 corners
    fallback_min_area: int = 30
    fallback_min_white: float = 0.10
    fallback_max_white: float = 0.50
    fallback_min_black: float = 0.35
    fallback_max_dot_offset: float = 0.30
    fallback_min_side: float = 10.0
    appearance_test: bool = True
    min_side: float = 12.0              # mean side of a candidate in pixels at a shorter image side of 2252 px,
                                        # scaled with the image (about 0.5 % of the shorter side)
    min_white: float = 0.12             # fraction of bright pixels inside the quadrilateral
    max_white: float = 0.55
    max_dot_offset: float = 0.15        # distance of the bright centroid from the centre / radius

    @staticmethod
    def legacy():
        """Limits of the original extraction."""
        return Params(max_side_ratio=1.5, max_diag_ratio=1.5, max_adjacent_ratio=4.0, appearance_test=False)


# ----------------------------------------------------------------------------------------------- geometry
def inside_mask(quad, xs, ys):
    """Points (xs, ys) inside the quadrilateral (crossing-number test)."""
    xs = np.asarray(xs, np.float64); ys = np.asarray(ys, np.float64)
    cnt = np.zeros(np.broadcast(xs, ys).shape, np.int32)
    for i in range(4):
        x1, y1 = quad[i]; x2, y2 = quad[(i + 1) % 4]
        if y1 == y2:
            continue
        m = (ys > min(y1, y2)) & (ys <= max(y1, y2)) & (xs <= max(x1, x2))
        if x1 != x2:
            xin = (ys - y1) * (x2 - x1) / (y2 - y1) + x1
            m &= xs <= xin
        cnt += m
    return (cnt & 1) == 1


def _grid_inside(quad, x0, x1, y0, y1):
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    return xs, ys, inside_mask(quad, xs, ys)


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))


@njit(cache=True)
def _clear_odds_idx(c):
    n = c.shape[0]; m = n * (n - 1) // 2
    vx = np.empty(m, np.int64); vy = np.empty(m, np.int64); pi = np.empty(m, np.int64); pj = np.empty(m, np.int64)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            vx[k] = c[i, 0] - c[j, 0]; vy[k] = c[i, 1] - c[j, 1]; pi[k] = i; pj[k] = j; k += 1
    best = 0.0; b0 = 0; b1 = 1
    for i in range(m - 1):
        for j in range(i + 1, m):
            z = float(vx[i] * vy[j] - vy[i] * vx[j]); a = z * z
            if a > best:
                best = a; b0 = i; b1 = j
    return pi[b0], pj[b0], pi[b1], pj[b1]


def _clear_odds(corners):
    """Keeps the two corner pairs whose difference vectors span the largest parallelogram."""
    idx = _clear_odds_idx(np.array(corners, np.int64))
    return [corners[int(i)] for i in idx]


def _extreme_points(comp):
    dx = (1, -1, 0, 0, 1, -1, 1, -1); dy = (0, 0, 1, -1, 1, -1, -1, 1)
    best = []
    for d in range(8):
        v = dx[d] * comp[:, 0] + dy[d] * comp[:, 1]
        best.append(tuple(int(t) for t in comp[int(np.argmax(v))]))
    out = []
    for p in best:
        if not any(abs(q[0] - p[0]) + abs(q[1] - p[1]) <= 1 for q in out):
            out.append(p)
    return out


def _angle_align(P):
    cx = 0.25 * (P[0][0] + P[1][0] + P[2][0] + P[3][0]); cy = 0.25 * (P[0][1] + P[1][1] + P[2][1] + P[3][1])
    phi = []
    for k in range(4):
        a = math.atan2(P[k][1] - cy, P[k][0] - cx)
        if a < 0:
            a += 2 * math.pi
        phi.append((a, float(k)))
    phi.sort()
    return [P[int(k)] for _, k in phi]


def _radius(P, C):
    return max(math.sqrt((p[0] - C[0]) * (p[0] - C[0]) + (p[1] - C[1]) * (p[1] - C[1])) for p in P)


def _page(P):
    # mean side used to choose between duplicates
    d0 = math.sqrt((P[0][0] - P[1][0]) ** 2 + (P[0][1] - P[1][0]) ** 2)
    d1 = _dist(P[1], P[2]); d2 = _dist(P[2], P[3]); d3 = _dist(P[3], P[1])
    return (d0 + d1 + d2 + d3) / 4


def _check_square(P, prm):
    d1 = _dist(P[0], P[2]); d2 = _dist(P[1], P[3])
    if max(d1, d2) > prm.max_diag_ratio * min(d1, d2):
        return False
    s = sorted([_dist(P[0], P[1]), _dist(P[1], P[2]), _dist(P[2], P[3]), _dist(P[0], P[3])])
    return not s[-1] > prm.max_side_ratio * s[0]


def _appearance_ok(P, binary, prm):
    H, W = binary.shape
    X = [p[0] for p in P]; Y = [p[1] for p in P]
    cx = 0.25 * (X[0] + X[1] + X[2] + X[3]); cy = 0.25 * (Y[0] + Y[1] + Y[2] + Y[3])
    side = 0.0; rad = 0.0
    for i in range(4):
        j = (i + 1) % 4
        side += 0.25 * math.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
        rad = max(rad, math.sqrt((X[i] - cx) ** 2 + (Y[i] - cy) ** 2))
    if side < prm.min_side * min(W, H) / 2252.0:
        return False
    x0 = max(0, int(math.floor(min(X)))); x1 = min(W - 1, int(math.ceil(max(X))))
    y0 = max(0, int(math.floor(min(Y)))); y1 = min(H - 1, int(math.ceil(max(Y))))
    xs, ys, m = _grid_inside(P, x0, x1, y0, y1)
    n = int(m.sum())
    w = m & (binary[y0:y1 + 1, x0:x1 + 1] > 127)
    nw = int(w.sum())
    if n == 0 or nw == 0:
        return False
    white = nw / n
    if white < prm.min_white or white > prm.max_white:
        return False
    wx = float(xs[w].sum()); wy = float(ys[w].sum())
    off = math.sqrt((wx / nw - cx) ** 2 + (wy / nw - cy) ** 2)
    return off <= prm.max_dot_offset * rad


def _stretch(P, C, coef, W, H):
    Q = [((p[0] - C[0]) * coef + C[0], (p[1] - C[1]) * coef + C[1]) for p in P]
    for q in Q:
        if q[0] < 0 or q[0] >= W or q[1] < 0 or q[1] >= H:
            return P, False
    return Q, True


# ----------------------------------------------------------------------------------------------- extraction
def binarize(grey, mode="fixed", sigma=0.0):
    """mode "fixed": window radius min(w, h)/20, "adaptive": radius from the size of dark components."""
    H, W = grey.shape
    if mode == "fixed":
        return image_binarization(grey, min(W, H) // 20)
    src = box_blur3(grey, sigma) if sigma > 0 else grey
    return adapt_image_binarization(src)


def find_quads(binary, prm=None):
    """Candidate quadrilaterals [(x, y)] * 4 on a binarized image (0 dark, 255 bright)."""
    prm = prm or Params()
    binary = np.where(np.asarray(binary) > 127, 255, 0).astype(np.uint8)
    H, W = binary.shape
    mr = min(W, H) // 20
    corners = detect_corners(binary)
    cmask = np.zeros((H, W), bool); cmask[corners[:, 1], corners[:, 0]] = True
    _, comps = connected_components(binary, 10)

    allsq = []
    for comp in comps:
        n = len(comp)
        if n == 0 or n > (10 * mr) * (10 * mr):
            continue
        if binary[comp[0, 1], comp[0, 0]] == 255:
            continue
        sel = cmask[comp[:, 1], comp[:, 0]]
        cr = [tuple(int(t) for t in p) for p in comp[sel]]
        if len(cr) > prm.texture_corner_factor * math.sqrt(n):
            continue
        if len(cr) > 4 and len(cr) < mr:
            cr = _clear_odds(cr)
        fallback = False
        if prm.small_fallback and len(cr) < 4 and prm.fallback_min_area <= n <= (2 * mr) * (2 * mr):
            ext = _extreme_points(comp)
            if len(ext) >= 4:
                cr = _clear_odds(ext) if len(ext) > 4 else ext
                fallback = True
        if len(cr) != 4:
            continue
        P = [(float(x), float(y)) for x, y in cr]
        C = (0.25 * (P[0][0] + P[1][0] + P[2][0] + P[3][0]), 0.25 * (P[0][1] + P[1][1] + P[2][1] + P[3][1]))
        R = int(_radius(P, C))
        if R > 7 * mr:
            continue
        P = _angle_align(P)
        # histogram of the binarized pixels inside: rejected only if the quadrilateral is completely dark
        x0 = int(C[0]) - R; y0 = int(C[1]) - R
        xs, ys, m = _grid_inside(P, x0, int(C[0]) + R, y0, int(C[1]) + R)
        vals = binary[np.clip(ys[m], 0, H - 1), np.clip(xs[m], 0, W - 1)]
        count = len(vals); nwhite = int((vals == 255).sum())
        accept = count > 0 and nwhite > 0
        if accept and fallback:
            white = nwhite / count; black = (count - nwhite) / count
            side = 0.25 * (_dist(P[0], P[1]) + _dist(P[1], P[2]) + _dist(P[2], P[3]) + _dist(P[3], P[0]))
            accept = (side >= prm.fallback_min_side and prm.fallback_min_white <= white <= prm.fallback_max_white
                      and black >= prm.fallback_min_black)
            if accept:
                wm = vals == 255
                cx = float(xs[m][wm].sum()) / nwhite; cy = float(ys[m][wm].sum()) / nwhite
                accept = _dist((cx, cy), C) <= prm.fallback_max_dot_offset * _radius(P, C)
        if accept:
            allsq.append([P, C])

    # duplicates: identical quadrilaterals, or quadrilaterals sharing a corner (the larger one is kept)
    Z = [(0.0, 0.0)] * 4
    sq = [list(a[0]) for a in allsq]
    for i in range(len(sq) - 1):
        if sq[i] == Z:
            continue
        for j in range(i + 1, len(sq)):
            if sq[i] == sq[j]:
                sq[j] = list(Z)
            elif any(a == b for a in sq[i] for b in sq[j]):
                if _page(sq[i]) > _page(sq[j]):
                    sq[j] = list(Z)
                else:
                    sq[i] = list(Z)
    quads = [(sq[i], allsq[i][1]) for i in range(len(sq)) if sq[i] != Z and _check_square(sq[i], prm)]

    out = []
    for P, C in quads:
        ok = True
        while ok and all(binary[int(p[1]), int(p[0])] == 0 for p in P):
            P, ok = _stretch(P, C, 1.005, W, H)
        out.append(P)
    if prm.appearance_test:
        out = [P for P in out if _appearance_ok(P, binary, prm)]
    # very elongated quadrilaterals (ratio of two adjacent sides) are rejected
    res = []
    for P in out:
        a = _dist(P[0], P[1]); b = _dist(P[2], P[1])
        if not prm.max_adjacent_ratio * min(a, b) < max(a, b):
            res.append(P)
    return res


def find_markers(image, mode="fixed", sigma=0.0, prm=None, return_binary=False):
    """Image (path or array) -> candidate quadrilaterals."""
    grey = norm_palette_u8(load_grey(image))
    binary = binarize(grey, mode, sigma)
    quads = find_quads(binary, prm)
    return (quads, binary) if return_binary else quads
