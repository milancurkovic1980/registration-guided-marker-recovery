"""Connected components (8-connectivity) in a fixed order."""
import numpy as np
from numba import njit


@njit(cache=True)
def _root(lab, i):
    while lab[i] != i:
        lab[i] = lab[lab[i]]
        i = lab[i]
    return i


@njit(cache=True)
def _label(img):
    H, W = img.shape
    L = np.empty((H, W), np.int64)
    par = np.empty(H * W + 1, np.int64); n = 0
    L[0, 0] = n; par[n] = n; n += 1
    for r in range(1, H):
        if img[r, 0] == img[r - 1, 0]:
            L[r, 0] = L[r - 1, 0]
        else:
            L[r, 0] = n; par[n] = n; n += 1
    for c in range(1, W):
        if img[0, c] == img[0, c - 1]:
            L[0, c] = L[0, c - 1]
        else:
            L[0, c] = n; par[n] = n; n += 1
        for r in range(1, H):
            my = -1
            if img[r, c] == img[r - 1, c]:
                my = L[r - 1, c]
            for d in range(-1, 1):
                if img[r, c] == img[r + d, c - 1]:
                    if my >= 0:
                        a = _root(par, my); b = _root(par, L[r + d, c - 1])
                        if a != b:
                            par[_root(par, my)] = _root(par, L[r + d, c - 1])
                    else:
                        my = L[r + d, c - 1]
            if my >= 0:
                L[r, c] = my
            else:
                L[r, c] = n; par[n] = n; n += 1
            if img[r - 1, c] == img[r, c - 1]:
                a = _root(par, L[r - 1, c]); b = _root(par, L[r, c - 1])
                if a != b:
                    par[_root(par, L[r - 1, c])] = _root(par, L[r, c - 1])
    for i in range(n):
        par[i] = _root(par, i)
    return L, par[:n]


def connected_components(img, min_size=10):
    """(tag image, components); each component lists its pixels (x, y), smaller ones are dropped."""
    L, par = _label(np.ascontiguousarray(img))
    root = par[L]                                   # root label of every pixel
    roots = np.flatnonzero(par == np.arange(len(par)))
    tag_of_root = np.full(len(par), -1, np.int64); tag_of_root[roots] = np.arange(len(roots))
    tag = tag_of_root[root]
    sizes = np.bincount(tag.ravel(), minlength=len(roots))
    keep = np.flatnonzero(sizes >= min_size)
    remap = np.full(len(roots), -1, np.int64); remap[keep] = np.arange(len(keep))
    tag = remap[tag]
    flat = tag.ravel()
    order = np.argsort(flat, kind="stable")          # row-major order is preserved inside a component
    order = order[flat[order] >= 0]
    counts = np.bincount(flat[flat >= 0], minlength=len(keep))
    W = img.shape[1]
    ys, xs = np.divmod(order, W)
    comps = np.split(np.stack([xs, ys], 1), np.cumsum(counts)[:-1]) if len(keep) else []
    return tag, comps
