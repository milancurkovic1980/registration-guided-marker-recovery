"""Detections on all views of both scanning sessions.

    python experiments/scanner/detect.py [--learned]

Classical detector (adaptive binarization, extraction, ellipse test): accepted quadrilaterals of every view
-> results/scanner/classical_<session>.json. With --learned, also the boxes of YOLOv12-N/S (Ultralytics) and of
RF-DETR-S (rfdetr) at confidence 0.01 -> results/scanner/<detector>_<session>.json. The weights are expected in
models/learned/ (yolo12n.pt, yolo12s.pt, rfdetr_s.pth, the last from the release of this repository)."""
import os, sys
import numpy as np
from common import SESSIONS, ROOT, Session, save_json
from recovery import classical_detections

LEARNED = os.path.join(ROOT, "models", "learned")


def yolo(session, name):
    from ultralytics import YOLO
    from PIL import Image
    m = YOLO(os.path.join(LEARNED, name + ".pt")); out = {}
    for v in session.views:
        r = m.predict(Image.open(session.path(v)).convert("RGB"), imgsz=1920, conf=0.01, iou=0.5, max_det=1000, verbose=False, device="cpu")[0]
        out[v] = np.c_[r.boxes.xyxy.numpy(), r.boxes.conf.numpy()].round(2).tolist()
    return out


def rfdetr(session):
    import torch, torchvision
    from PIL import Image
    from rfdetr import RFDETRSmall
    rf = RFDETRSmall(pretrain_weights=os.path.join(LEARNED, "rfdetr_s.pth"), resolution=1024); T, S = 1024, 896; out = {}
    for v in session.views:
        im = Image.open(session.path(v)).convert("RGB"); s = 4000 / im.size[0]
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC); W, H = im.size
        xs = list(range(0, W - T + 1, S)); ys = list(range(0, H - T + 1, S))
        if xs[-1] + T < W: xs.append(W - T)
        if ys[-1] + T < H: ys.append(H - T)
        bb = []
        for x in xs:
            for y in ys:
                d = rf.predict(im.crop((x, y, x + T, y + T)), threshold=0.01)
                if len(d): bb.append(np.c_[d.xyxy + [x, y, x, y], d.confidence])
        if bb:
            b = torch.tensor(np.vstack(bb), dtype=torch.float32); b = b[torchvision.ops.nms(b[:, :4], b[:, 4], 0.5)].numpy() / [s, s, s, s, 1]
        else:
            b = np.zeros((0, 5))
        out[v] = np.round(b, 2).tolist()
    return out


if __name__ == "__main__":
    for name, root in SESSIONS.items():
        ses = Session(root)
        save_json({v: classical_detections(ses.path(v)).round(3).tolist() for v in ses.views}, f"classical_{name}.json")
        print(name, "classical", len(ses.views), "views", flush=True)
        if "--learned" in sys.argv:
            for n in ("yolo12n", "yolo12s"):
                save_json(yolo(ses, n), f"{n}_{name}.json"); print(name, n, flush=True)
            save_json(rfdetr(ses), f"rfdetr_s_{name}.json"); print(name, "rfdetr_s", flush=True)
