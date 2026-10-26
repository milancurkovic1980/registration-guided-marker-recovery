"""Recall of the extraction against the hand-marked markers of the evaluation photographs.

    python experiments/recall.py      every folder work/candidates/<tag>/ -> work/results/recall.json

Recall (all and certain markers), false candidates per photograph and photographs with at least 3 and 6 found
markers, for the test scenes and the glare scene."""
import glob, json, os
import numpy as np
from common import WORK, photo_table, annotations, match
from markerdet import read_candidates


def stats(tag, photos):
    na = ns = f = fs = fp = r3 = r6 = 0
    for p in photos:
        c, u = annotations(p)
        found, false = match(read_candidates(os.path.join(WORK, "candidates", tag, p + ".txt")), np.vstack([c, u]))
        k = int(found.sum()); na += len(found); ns += len(c); f += k; fs += int(found[:len(c)].sum())
        fp += int(false.sum()); r3 += k >= 3; r6 += k >= 6
    return dict(recall=f / na, recall_certain=fs / ns, false_per_photo=fp / len(photos), reg3=r3, reg6=r6, photos=len(photos))


if __name__ == "__main__":
    T = photo_table(); out = {}
    groups = {g: sorted(p for p, r in T.items() if r["annotation"] == "evaluation" and (r["split"] == "glare") == (g == "glare"))
              for g in ("test", "glare")}
    for d in sorted(glob.glob(os.path.join(WORK, "candidates", "*"))):
        tag = os.path.basename(d)
        for g, P in groups.items():
            if all(os.path.exists(os.path.join(d, p + ".txt")) for p in P):
                s = out[f"{g}|{tag}"] = stats(tag, P)
                print(f"{g:5s} {tag:28s} recall {100 * s['recall']:3.0f}%  certain {100 * s['recall_certain']:3.0f}%  "
                      f"false/photo {s['false_per_photo']:5.1f}  >=3 {s['reg3']}/{s['photos']}  >=6 {s['reg6']}/{s['photos']}")
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(WORK, "results", "recall.json"), "w"), indent=1)
