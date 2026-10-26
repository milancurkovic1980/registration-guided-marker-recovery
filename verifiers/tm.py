"""Template matching of the detector (TM): edge map and chamfer distance of the binarized crop compared with
ring models (models/template/markers.bmp); marker if the smallest squared error / 255^2 <= 3 x crop size."""
import numpy as np, cv2
from scipy import ndimage
from PIL import Image


def clear_frame(a, R):
    a = a.copy(); H, W = a.shape
    if W <= R or H <= R:
        return a
    m = np.zeros_like(a, bool)
    m[R + 1:H - R, R + 1:W - R] = True
    a[~m] = 0
    return a


def _sobel_nms(g):
    g = g.astype(np.float64); H, W = g.shape
    # convolution: sum_{xx,yy} src(x+xx, y+yy) * k[xx+1][yy+1], x = column
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)   # k[xx][yy]
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float)
    sx = np.zeros_like(g); sy = np.zeros_like(g)
    for xx in (-1, 0, 1):
        for yy in (-1, 0, 1):
            sh = np.roll(np.roll(g, -yy, 0), -xx, 1)
            sx += sh * kx[xx + 1, yy + 1]; sy += sh * ky[xx + 1, yy + 1]
    mag = np.hypot(sx, sy); ori = np.degrees(np.arctan2(sy, sx))
    ori = np.where(ori < 0, -(ori + 180.0), ori); a = np.abs(ori)
    q = np.zeros_like(ori)
    q[a < 22.5] = 1; q[a >= 157.5] = 1; q[(a >= 22.5) & (a < 67.5)] = 2
    q[(a >= 67.5) & (a < 112.5)] = 3; q[(a >= 112.5) & (a < 157.5)] = 4
    border = np.zeros_like(g, bool); border[1:-1, 1:-1] = True
    mag[~border] = 0
    M = mag
    def at(dx, dy):  # magnitude at (x+dx, y+dy)
        return np.roll(np.roll(M, -dy, 0), -dx, 1)
    keep = np.zeros_like(M, bool)
    keep |= (q == 3) & (at(-1, 0) < M) & (M > at(1, 0))
    keep |= (q == 1) & (at(0, -1) < M) & (M > at(0, 1))
    keep |= (q == 4) & (at(-1, 1) < M) & (M > at(1, -1))
    keep |= (q == 2) & (at(-1, -1) < M) & (M > at(1, 1))    
    out = M.copy(); inner = border.copy()
    out[inner & ~keep] = 0
    return out


def canny_edge(a, sigma=3.0, use_median=True, tdown=0.25, tup=10.0):
    im = a.astype(np.uint8)
    if use_median:
        im = cv2.medianBlur(im, 5)
    gp = cv2.GaussianBlur(im.astype(np.float64), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    mag = _sobel_nms(gp)
    weak = mag >= tdown; strong = mag >= tup
    lab, n = ndimage.label(weak, structure=np.ones((3, 3)))
    good = np.unique(lab[strong]); good = good[good > 0]
    edges = np.isin(lab, good)
    out = np.where(edges, 0, 255).astype(np.uint8)
    H, W = out.shape
    o = out  # gap closing (x = column, y = row)
    for x in range(2, W - 2):
        for y in range(2, H - 2):
            if o[y, x] == 0:
                if o[y, x + 1] == 0 or o[y - 1, x + 1] == 0 or o[y + 1, x + 1] == 0: continue
                if o[y - 1, x + 2] == 0 or o[y + 1, x + 2] == 0: o[y, x + 1] = 0
    for x in range(W - 3, 1, -1):
        for y in range(2, H - 2):
            if o[y, x] == 0:
                if o[y, x - 1] == 0 or o[y - 1, x - 1] == 0 or o[y + 1, x - 1] == 0: continue
                if o[y - 1, x - 2] == 0 or o[y + 1, x - 2] == 0: o[y, x - 1] = 0
    for y in range(2, H - 2):
        for x in range(2, W - 2):
            if o[y, x] == 0:
                if o[y - 1, x] == 0 or o[y - 1, x - 1] == 0 or o[y - 1, x + 1] == 0: continue
                if o[y - 2, x - 1] == 0 or o[y - 2, x + 1] == 0: o[y - 1, x] = 0
    for y in range(H - 3, 1, -1):
        for x in range(2, W - 2):
            if o[y, x] == 0:
                if o[y + 1, x] == 0 or o[y + 1, x - 1] == 0 or o[y + 1, x + 1] == 0: continue
                if o[y + 2, x - 1] == 0 or o[y + 2, x + 1] == 0: o[y + 1, x] = 0
    return o


def chamfer(a, coef=0.2):
    """Two-pass chamfer distance in 8-bit arithmetic."""
    c = a.astype(np.float64).tolist(); H = len(c); W = len(c[0])
    eps = 255 * coef; eps2 = eps / np.sqrt(2)
    for y in range(H):
        row = c[y]; prev = c[y - 1] if y > 0 else None
        for x in range(W - 1, -1, -1):
            p = row[x]
            if x > 0 and y > 0: p = min(p, eps + prev[x - 1])
            if x < W - 1 and y > 0: p = min(p, eps + prev[x + 1])
            if y > 0: p = min(p, eps2 + prev[x])
            if x < W - 1: p = min(p, eps2 + row[x + 1])
            row[x] = float(int(p))
    for y in range(H - 1, -1, -1):
        row = c[y]; nxt = c[y + 1] if y < H - 1 else None
        for x in range(W):
            p = row[x]
            if x > 0 and y < H - 1: p = min(p, eps + nxt[x - 1])
            if x < W - 1 and y < H - 1: p = min(p, eps + nxt[x + 1])
            if y < H - 1: p = min(p, eps2 + nxt[x])
            if x > 0: p = min(p, eps2 + row[x - 1])
            row[x] = float(int(p))
    return np.array(c)

def load_models(path):
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    g = (0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]).astype(np.uint8)
    ms = g.shape[1]; n = g.shape[0] // ms
    models = []
    for j in range(n):
        m = g[j * ms:(j + 1) * ms, :ms].astype(np.float64)
        for r in range(4):
            models.append(np.rot90(m, -r).copy())
    return np.stack(models), ms


def chain(crop, ms):
    a = clear_frame(crop, max(1, ms // 10))
    a = canny_edge(a, 3.0, True)
    return chamfer(a, 0.2)


def errors(crops, models):
    ms = models.shape[1]
    out = np.empty(len(crops))
    for i, c in enumerate(crops):
        m = chain(c, ms)
        out[i] = min(((M - m) ** 2).sum() / 65025.0 for M in models)
    return out


def threshold(models):
    return 3.0 * models.shape[1]
