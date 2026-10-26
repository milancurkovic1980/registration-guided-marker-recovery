"""Registration-guided recovery on both scanning sessions (Table of the recovery in the article).

    python experiments/scanner/recovery_experiment.py          classical detector, sequential and global
    python experiments/scanner/recovery_experiment.py --learned  also on top of YOLOv12-N/S (global and sequential)

For each session and variant: recall before and after the recovery with a 95% bootstrap interval of the gain,
recovered and false markers, recall per surface or object, and the view pairs (at least four annotated common
markers) whose markers found in both views number at least three and six."""
import itertools, sys
import numpy as np
from common import SESSIONS, THRESHOLDS, Session, group, load_json, save_json, bootstrap_gain
from recovery import recover_session, match_markers
from recovery.core import found_by_quads, found_by_boxes, box_detections, quad_side


def annotated_ids(ses, v, side):
    """3D id of every annotated marker (nearest centre of any detector within 0.6 sides) -> {id: (index, XYZ)}."""
    pts = ses.annotation(v); ids = {}
    for m in ("box", "quad", "dot"):
        for k, (x, X) in ses.markers3d(v, m).items():
            if k in ids or not len(pts):
                continue
            d = np.linalg.norm(pts - x, axis=1); i = int(d.argmin())
            if d[i] < 0.6 * side:
                ids[k] = (i, X)
    return ids


def view_pairs(ses, masks, sides):
    """Common markers found in both views, for all view pairs with at least four annotated common markers."""
    rows = []
    for p in ses.projects():
        vs = ses.views_of(p); ids = {v: annotated_ids(ses, v, sides[v]) for v in vs}
        for vi, vj in itertools.combinations(vs, 2):
            mt = match_markers({k: X for k, (i, X) in ids[vi].items()}, {k: X for k, (i, X) in ids[vj].items()})
            if len(mt) < 4:
                continue
            rows.append({s: sum(1 for a, b in mt.items() if M[vi][ids[vi][a][0]] and M[vj][ids[vj][b][0]]) for s, M in masks.items()} | {"annotated": len(mt)})
    return rows


def run(name, det_name, modes, zoom=None):
    ses = Session(SESSIONS[name]); D = load_json(f"{det_name}_{name}.json"); dets, found0 = {}, {}
    for v in ses.views:
        pts = ses.annotation(v)
        if det_name == "classical":
            q = np.array(D[v], float).reshape(-1, 4, 2); found0[v] = found_by_quads(q, pts)[0]
            dets[v] = dict(centres=q.mean(1) if len(q) else np.zeros((0, 2)), side=quad_side(q))
        else:
            b = box_detections(D[v], THRESHOLDS[det_name]); found0[v] = found_by_boxes(b, pts)[0]
            dets[v] = dict(centres=(b[:, :2] + b[:, 2:4]) / 2, side=float(np.median(np.mean(b[:, 2:4] - b[:, :2], 1))) if len(b) else 15.0)
    method = "quad" if det_name == "classical" else "all"   # learned detectors: 3D points of all detectors of the view
    summary = {}; masks = {"detector": found0}
    for mode in modes:
        res = recover_session(ses, dets, mode=mode, method=method, searcher=zoom)
        n0, n1, n, false, rec = [], [], [], 0, 0; per = {}; mask = {}
        for v in ses.views:
            pts = ses.annotation(v); f = found0[v].copy()
            for r in res[v]["recovered"]:
                d = np.linalg.norm(pts - r, axis=1) if len(pts) else np.array([1e9]); i = int(d.argmin())
                if d[i] < 0.5 * dets[v]["side"] and not f[i]: f[i] = True; rec += 1
                elif d[i] >= 0.5 * dets[v]["side"]: false += 1
            n0.append(int(found0[v].sum())); n1.append(int(f.sum())); n.append(len(pts)); mask[v] = f
            g = per.setdefault(group(name, v), [0, 0, 0]); g[0] += n0[-1]; g[1] += n1[-1]; g[2] += n[-1]
        gain, ci = bootstrap_gain(n0, n1, n); masks[mode] = mask
        summary[mode] = dict(recall_before=100 * sum(n0) / sum(n), recall_after=100 * sum(n1) / sum(n), gain_points=100 * gain,
                             gain_ci95=[100 * c for c in ci], recovered=rec, false=false, markers=sum(n),
                             per_group={k: [round(100 * a / c, 1), round(100 * b / c, 1)] for k, (a, b, c) in per.items()})
        print(f"{name:15s} {det_name:9s} {mode:10s} recall {summary[mode]['recall_before']:.1f} -> {summary[mode]['recall_after']:.1f}%"
              f"  gain {100 * gain:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]  recovered {rec}  false {false}  {summary[mode]['per_group']}", flush=True)
    if det_name == "classical":
        masks["annotated"] = {v: np.ones(len(ses.annotation(v)), bool) for v in ses.views}
        rows = view_pairs(ses, masks, {v: dets[v]["side"] for v in ses.views})
        summary["view_pairs"] = {s: dict(ge3=int(sum(r[s] >= 3 for r in rows)), ge6=int(sum(r[s] >= 6 for r in rows))) for s in masks} | {"pairs": len(rows)}
        print(f"{name:15s} view pairs ({len(rows)}):", {s: (v['ge3'], v['ge6']) for s, v in summary['view_pairs'].items() if s != 'pairs'}, flush=True)
    return summary


def yolo_zoom(session, net):
    """Local search with the learned detector on the scaled crop (3x, in a 640 x 640 canvas)."""
    from ultralytics import YOLO
    from PIL import Image
    import os
    from common import ROOT
    m = YOLO(os.path.join(ROOT, "models", "learned", net + ".pt")); cache = {}

    def search(view, grey, c, side, z=3):
        if view not in cache: cache[view] = Image.open(session.path(view)).convert("RGB")
        h = int(3 * side); x0, y0 = int(c[0] - h), int(c[1] - h)
        crop = cache[view].crop((x0, y0, x0 + 2 * h, y0 + 2 * h)).resize((2 * h * z, 2 * h * z), Image.BICUBIC)
        can = Image.new("RGB", (640, 640), tuple(int(t) for t in np.median(np.asarray(crop).reshape(-1, 3), 0))); o = (640 - crop.size[0]) // 2
        can.paste(crop, (o, o)); b = m.predict(can, imgsz=640, conf=0.05, iou=0.5, verbose=False, device="cpu")[0].boxes; best = None
        for (a0, b0, a1, b1), cf in zip(b.xyxy.numpy(), b.conf.numpy()):
            cc = np.array([((a0 + a1) / 2 - o) / z + x0, ((b0 + b1) / 2 - o) / z + y0])
            if np.hypot(*(cc - c)) < side and (best is None or cf > best[0]): best = (cf, cc)
        return None if best is None else best[1]
    return search


if __name__ == "__main__":
    out = {}
    for name in SESSIONS:
        out[f"{name}|classical"] = run(name, "classical", ("sequential", "global"))
    if "--learned" in sys.argv:
        for name in SESSIONS:
            for net in ("yolo12n", "yolo12s"):
                out[f"{name}|{net}"] = run(name, net, ("sequential", "global"), zoom=yolo_zoom(Session(SESSIONS[name]), net))
    save_json(out, "recovery.json")
