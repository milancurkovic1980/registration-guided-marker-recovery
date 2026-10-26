import csv, glob, json, os, re
import itertools
import numpy as np
from PIL import Image

from markerdet import detect
from markerdet.extract import binarize, find_quads, Params, inside_mask
from markerdet.imaging import norm_palette_u8, load_grey
from markerdet.crops import rectify
from markerdet.verify import load_verifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ELLIPSE = os.path.join(ROOT, "models", "adaptive", "Ellipse_bin")
METHODS = {"quad": "markers3d_quad.csv", "box": "markers3d_box.csv", "dot": "markers3d_dot.csv"}

_ell = None


def ellipse_test():
    global _ell
    if _ell is None:
        _ell = load_verifier(ELLIPSE)
    return _ell


# ----------------------------------------------------------------------------------------------- data of a session
class Session:
    """A scanning session: data/scans/<session>/<project>/objectN/ with

    01.png              image under the white light of the projector (the first image of the view)
    annotation.json     annotated markers ("sure" and "unsure", one click per marker)
    markers3d_*.csv     markers found by a detector with their 3D points (id, x, y, X, Y, Z, n, rms);
                        quad = quadrilateral of the classical detector, box = box of YOLOv12-S,
                        dot = intensity-weighted centroid of the dot. The same id is the same marker in one view."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.views = []
        for d in sorted(glob.glob(os.path.join(self.root, "*", "object*"))):
            if os.path.exists(os.path.join(d, "annotation.json")):
                self.views.append(os.path.relpath(d, self.root).replace(os.sep, "/"))
        self.views.sort(key=lambda v: (v.split("/")[0], int(re.findall(r"\d+", v.split("/")[1])[0])))
        self._cam = {}

    def projects(self):
        return sorted({v.split("/")[0] for v in self.views})

    def views_of(self, project):
        return [v for v in self.views if v.split("/")[0] == project]

    def path(self, view, name="01.png"):
        return os.path.join(self.root, view, name)

    def annotation(self, view):
        a = json.load(open(self.path(view, "annotation.json")))
        return np.array(a["sure"] + a["unsure"], float).reshape(-1, 2)

    def markers3d(self, view, method="quad"):
        """{id: (xy, XYZ)} of one detector, or of all three ("all", first quad, then box, then dot)."""
        if method == "all":
            out = {}
            for m in ("quad", "box", "dot"):
                for k, val in self.markers3d(view, m).items():
                    out.setdefault(k, val)
            return out
        out = {}
        p = self.path(view, METHODS[method])
        if not os.path.exists(p):
            return out
        for r in csv.DictReader(open(p)):
            if r["X"] in ("NaN", "nan", ""):
                continue
            out[int(r["id"])] = (np.array([float(r["x"]), float(r["y"])]), np.array([float(r["X"]), float(r["Y"]), float(r["Z"])]))
        return out

    def camera(self, project):
        """(f, cx, cy) of the camera, estimated from the markers of the project (no calibration file is needed)."""
        if project not in self._cam:
            # focal length and principal point fitted to the pixels and 3D points of all markers of the project
            A, b = [], []
            for v in self.views_of(project):
                for x, X in self.markers3d(v, "quad").values():
                    A += [[-X[0] / X[2], 1, 0], [X[1] / X[2], 0, 1]]; b += [x[0], x[1]]
            self._cam[project] = tuple(float(t) for t in np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0])
        return self._cam[project]


# ------------------------------------------------------------------------------------------------------ geometry
def kabsch(P, Q):
    """Least-squares rigid transformation (R, t) with Q ~ R P + t."""
    cp, cq = P.mean(0), Q.mean(0); H = (P - cp).T @ (Q - cq); U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def match_markers(Pi, Pj, tol=1.0, inlier=1.5, min_dist=15.0):
    """Correspondences {id_i: id_j} between two sets of 3D markers {id: XYZ} by random sample consensus on the
    distances between markers (two pairs with the same distance within tol mm and a third consistent marker define a
    rigid transformation, the one that brings most markers within inlier mm of a marker of the other view is kept)."""
    ki, kj = list(Pi), list(Pj)
    A = np.array([Pi[k] for k in ki]); B = np.array([Pj[k] for k in kj])
    if len(A) < 3 or len(B) < 3:
        return {}
    DA = np.linalg.norm(A[:, None] - A[None], axis=2); DB = np.linalg.norm(B[:, None] - B[None], axis=2)
    best = {}
    for a1, a2 in itertools.combinations(range(len(A)), 2):
        if DA[a1, a2] < min_dist:
            continue
        for b1, b2 in np.argwhere(np.abs(DB - DA[a1, a2]) < tol):
            if b1 == b2:
                continue
            for a3 in range(len(A)):
                if a3 in (a1, a2):
                    continue
                ok = np.where((np.abs(DB[b1] - DA[a1, a3]) < tol) & (np.abs(DB[b2] - DA[a2, a3]) < tol))[0]
                for b3 in ok:
                    if b3 in (b1, b2):
                        continue
                    R, t = kabsch(A[[a1, a2, a3]], B[[b1, b2, b3]])
                    d = np.linalg.norm((A @ R.T + t)[:, None] - B[None], axis=2)
                    nn = d.argmin(1); good = d[np.arange(len(A)), nn] < inlier
                    mt = {ki[x]: kj[nn[x]] for x in np.where(good)[0]}
                    if len(set(mt.values())) == len(mt) and len(mt) > len(best):
                        best = mt
                    break
                break
        if len(best) >= min(len(A), len(B)) - 1 and len(best) >= 6:
            break
    return best


def project(X, camera):
    """Pixel of a 3D point in the camera frame of the scanner (convention of its calibration files)."""
    f, cx, cy = camera
    return np.array([cx - f * X[0] / X[2], cy + f * X[1] / X[2]])


# -------------------------------------------------------------------------------------------------- detections
def classical_detections(image):
    """Classical detector: adaptive binarization, candidate extraction, ellipse test on the binarized crop.
    Returns the accepted quadrilaterals (N, 4, 2)."""
    r = detect(image, "adaptive", 0.0)
    if len(r.quads) == 0:
        return np.zeros((0, 4, 2))
    return r.quads[ellipse_test().accept(r.bin_crops)]


def quad_side(quads, default=20.0):
    if len(quads) == 0:
        return default
    return float(np.median([np.mean([np.hypot(*(q[i] - q[(i + 1) % 4])) for i in range(4)]) for q in quads]))


def box_detections(boxes, threshold):
    """Boxes (x0, y0, x1, y1, confidence) of a learned detector above a confidence threshold."""
    b = np.array([x for x in boxes if x[4] >= threshold], float).reshape(-1, 5)
    return b


def found_by_quads(quads, pts, g=1.3):
    """found[i]: annotated marker i lies in a quadrilateral enlarged g times, false[k]: quadrilateral k contains none."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(quads) == 0:
        return np.zeros(len(pts), bool), np.zeros(0, bool)
    ins = np.zeros((len(pts), len(quads)), bool)
    for k, q in enumerate(np.asarray(quads, float)):
        c = q.mean(0); qq = c + g * (q - c)
        ins[:, k] = inside_mask([tuple(t) for t in qq], pts[:, 0], pts[:, 1])
    return ins.any(1), ~ins.any(0)


def found_by_boxes(boxes, pts, g=1.3):
    b = np.asarray(boxes, float).reshape(-1, 5); pts = np.asarray(pts, float).reshape(-1, 2)
    if len(b) == 0:
        return np.zeros(len(pts), bool), np.zeros(0, bool)
    c = (b[:, :2] + b[:, 2:4]) / 2; h = (b[:, 2:4] - b[:, :2]) / 2 * g
    ins = (np.abs(pts[:, None, 0] - c[None, :, 0]) <= h[None, :, 0]) & (np.abs(pts[:, None, 1] - c[None, :, 1]) <= h[None, :, 1])
    return ins.any(1), ~ins.any(0)


# ------------------------------------------------------------------------------------------------- local search
def local_search(grey, c, side, canvas=1080, scaled=300):
    """Local search around a predicted centre c: a crop of four marker sides is scaled to `scaled` px and placed in
    the middle of a canvas x canvas image filled with its median grey level, binarized with a fixed window, and the
    extracted candidate closest to c (within one side) that passes the ellipse test is returned (or None)."""
    s = 2.0 * side
    x0, y0 = int(max(c[0] - s, 0)), int(max(c[1] - s, 0))
    x1, y1 = int(min(c[0] + s, grey.shape[1] - 1)), int(min(c[1] + s, grey.shape[0] - 1))
    crop = grey[y0:y1 + 1, x0:x1 + 1]
    if crop.size < 100:
        return None
    f = scaled / max(crop.shape)
    g = np.asarray(Image.fromarray(crop).resize((round(crop.shape[1] * f), round(crop.shape[0] * f)), Image.BICUBIC))
    o = (canvas - scaled) // 2
    can = np.full((canvas, canvas), int(np.median(g)), np.uint8); can[o:o + g.shape[0], o:o + g.shape[1]] = g
    b = binarize(norm_palette_u8(can), "fixed")
    best = None
    for q in find_quads(b, Params(min_side=0.0)):
        q = np.array(q, float)
        if not ellipse_test().accept(rectify(b, q, 64)[None])[0]:
            continue
        cq = (q.mean(0) - o) / f + [x0, y0]
        d = np.hypot(*(cq - c))
        if d < side and (best is None or d < best[0]):
            best = (d, cq)
    return None if best is None else best[1]


# ------------------------------------------------------------------------------------------------------ recovery
def recover_session(session, detections, mode="sequential", method="quad", searcher=None):
    """Registration-guided recovery on all views of a session.

    detections: {view: dict(centres=(N, 2) detected centres, side=marker side in px)}
    mode:       "sequential" (only views recorded before) or "global" (all other views of the project)
    method:     3D points used for the registration (markers3d_<method>.csv, the markers of the detector)
    searcher:   function(view, grey, centre, side) -> centre or None (default: local_search)
    Returns {view: dict(predicted=[...], recovered=[...], used_views=n)}."""
    out = {}
    for prj in session.projects():
        views = session.views_of(prj); cam = session.camera(prj)
        K = {v: session.markers3d(v, method) for v in views}
        order = {v: int(re.findall(r"\d+", v.split("/")[1])[0]) for v in views}
        for vj in views:
            J = detections[vj]; cands = []; used = 0
            for vi in views:
                if vi == vj or (mode == "sequential" and order[vi] > order[vj]):
                    continue
                mt = match_markers({k: X for k, (x, X) in K[vi].items()}, {k: X for k, (x, X) in K[vj].items()})
                if len(mt) < 3:
                    continue
                used += 1
                R, t = kabsch(np.array([K[vi][a][1] for a in mt]), np.array([K[vj][b][1] for b in mt.values()]))
                for k, (x, X) in K[vi].items():
                    if k in mt:
                        continue
                    c = project(R @ X + t, cam)
                    if not (0 <= c[0] < 1920 and 0 <= c[1] < 1080):
                        continue
                    if len(J["centres"]) and np.min(np.linalg.norm(J["centres"] - c, axis=1)) < J["side"]:
                        continue
                    if any(np.hypot(*(c - cc)) < 0.5 * J["side"] for cc in cands):
                        continue
                    cands.append(c)
            new = []
            if cands:
                grey = norm_palette_u8(load_grey(session.path(vj)))
                for c in cands:
                    r = searcher(vj, grey, c, J["side"]) if searcher else local_search(grey, c, J["side"])
                    if r is not None and not any(np.hypot(*(r - n)) < 0.5 * J["side"] for n in new):
                        new.append(r)
            out[vj] = dict(predicted=[c.tolist() for c in cands], recovered=[r.tolist() for r in new], used_views=used)
    return out


def evaluate_recovery(session, detections, found0, result):
    """Labels the recovered markers against the annotation: recovered (an annotated marker missed by the detector,
    within half a marker side), duplicate or false. Returns per-view counts."""
    rows = {}
    for v in session.views:
        pts = session.annotation(v); f = found0[v].copy(); side = detections[v]["side"]; lab = []
        for r in result[v]["recovered"]:
            d = np.linalg.norm(pts - r, axis=1) if len(pts) else np.array([1e9]); i = int(d.argmin())
            if d[i] < 0.5 * side and not f[i]:
                f[i] = True; lab.append("recovered")
            elif d[i] < 0.5 * side:
                lab.append("duplicate")
            else:
                lab.append("false")
        rows[v] = dict(n=len(pts), found=int(found0[v].sum()), found_after=int(f.sum()), labels=lab, found_mask=f.tolist())
    return rows
