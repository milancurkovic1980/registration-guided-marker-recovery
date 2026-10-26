"""Recall, false detections per view and views with at least 3 and 6 markers found, per surface or object.

    python experiments/scanner/evaluate_detection.py      (after detect.py)"""
import os
import numpy as np
from common import SESSIONS, RESULTS, THRESHOLDS, Session, group, load_json, save_json
from recovery.core import found_by_quads, found_by_boxes, box_detections

out = {}
for name, root in SESSIONS.items():
    ses = Session(root); acc = {}
    dets = {"classical": load_json(f"classical_{name}.json")}
    for n in THRESHOLDS:
        if os.path.exists(os.path.join(RESULTS, f"{n}_{name}.json")):
            dets[n] = load_json(f"{n}_{name}.json")
    for det, D in dets.items():
        for v in ses.views:
            pts = ses.annotation(v)
            f, fa = found_by_quads(np.array(D[v], float).reshape(-1, 4, 2), pts) if det == "classical" else \
                found_by_boxes(box_detections(D[v], THRESHOLDS[det]), pts)
            for g in (group(name, v), "all"):
                t = acc.setdefault((det, g), dict(markers=0, found=0, false=0, views=0, reg3=0, reg6=0))
                t["markers"] += len(pts); t["found"] += int(f.sum()); t["false"] += int(fa.sum()); t["views"] += 1
                t["reg3"] += int(f.sum() >= 3); t["reg6"] += int(f.sum() >= 6)
    res = {f"{d}|{g}": dict(recall=round(100 * t["found"] / t["markers"], 1), false_per_view=round(t["false"] / t["views"], 2),
                            views_ge3=t["reg3"], views_ge6=t["reg6"], views=t["views"], markers=t["markers"]) for (d, g), t in acc.items()}
    out[name] = res
    print("==", name)
    for k, r in sorted(res.items()):
        print(f"  {k:28s} recall {r['recall']:5.1f}%  false {r['false_per_view']:5.2f}  >=3 {r['views_ge3']:3d}  >=6 {r['views_ge6']:3d}  ({r['views']} views, {r['markers']} markers)")
save_json(out, "detection.json")
