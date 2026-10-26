"""Recent end-to-end learned detectors as reference: YOLOv12 (NeurIPS 2025) and RF-DETR (ICLR 2026), trained on the
labelled markers of the training scenes.

    pip install ultralytics rfdetr
    python experiments/learned_detectors.py prepare            data sets in work/detectors/
    python experiments/learned_detectors.py train              YOLOv12-N, YOLOv12-S and RF-DETR-S (GPU)
    python experiments/learned_detectors.py predict [folder]   boxes of all test and glare photographs and of the
                                                               images in an optional extra folder
                                                               -> work/detectors/pred_<model>.json

Training boxes are the candidates labelled as markers in any of the three extraction settings (merged), on the
photographs of the training scenes. One scene and lighting session is kept out for validation. YOLOv12 sees the whole
photograph at 1920 px, RF-DETR tiles of 1024 px at the full resolution."""
import csv, glob, json, os, sys, time
import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W = os.path.join(ROOT, "work", "detectors")
YOLO_MODELS = ["yolo12n", "yolo12s"]
YOLO_IMGSZ = 1920
TILE, STEP = 1024, 896
VAL_SETUP = ("pc_tower_plant", "daylight")      # scene and lighting session kept out for validation


def photos():
    return {r["photo"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "photos.csv")))}


def marker_boxes():
    """Union of the candidates labelled as markers in the three settings, as axis-aligned boxes (x0, y0, x1, y1)."""
    out = {}
    for f in glob.glob(os.path.join(ROOT, "data", "labels", "*.csv")):
        for r in csv.DictReader(open(f)):
            if r["label"] != "1":
                continue
            q = np.array([float(r[k]) for k in ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3")]).reshape(4, 2)
            box = np.r_[q.min(0), q.max(0)]; c = box.reshape(2, 2).mean(0); s = (box[2:] - box[:2]).mean()
            L = out.setdefault(r["photo"], [])
            if all(np.hypot(*(c - b.reshape(2, 2).mean(0))) > 0.5 * s for b in L):
                L.append(box)
    return out


def tiles(w, h):
    xs = list(range(0, max(w - TILE, 0) + 1, STEP)); ys = list(range(0, max(h - TILE, 0) + 1, STEP))
    if xs[-1] + TILE < w: xs.append(w - TILE)
    if ys[-1] + TILE < h: ys.append(h - TILE)
    return [(x, y) for y in ys for x in xs]


def yolo_line(b, w, h):
    return f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}\n"


def prepare():
    T = photos(); B = marker_boxes(); n = {}
    for p, r in T.items():
        part = "test" if r["split"] != "train" else ("val" if (r["scene"], r["lighting"]) == VAL_SETUP else "train")
        src = glob.glob(os.path.join(ROOT, "photos", p + ".*"))[0]
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB"); w, h = im.size
        for d in ("full", "tiles"):
            for s in ("images", "labels"):
                os.makedirs(os.path.join(W, d, s, part), exist_ok=True)
        im.save(os.path.join(W, "full", "images", part, p + ".jpg"), quality=95)
        boxes = B.get(p, []) if part != "test" else []
        open(os.path.join(W, "full", "labels", part, p + ".txt"), "w").writelines(yolo_line(b, w, h) for b in boxes)
        if part != "test":                                   # tiles at the full resolution for RF-DETR
            for x, y in tiles(w, h):
                inside = [b - [x, y, x, y] for b in boxes if b[0] >= x and b[1] >= y and b[2] <= x + TILE and b[3] <= y + TILE]
                name = f"{p}_{x}_{y}"
                im.crop((x, y, x + TILE, y + TILE)).save(os.path.join(W, "tiles", "images", part, name + ".jpg"), quality=95)
                open(os.path.join(W, "tiles", "labels", part, name + ".txt"), "w").writelines(yolo_line(b, TILE, TILE) for b in inside)
        n[part] = n.get(part, 0) + 1
    for d in ("full", "tiles"):
        open(os.path.join(W, d, "data.yaml"), "w").write(f"path: {os.path.join(W, d)}\ntrain: images/train\nval: images/val\nnames:\n  0: marker\n")
    print("photos", n, "validation setup", VAL_SETUP)


def train(which="all"):
    import torch
    dev = 0 if torch.cuda.is_available() else "cpu"
    if which in ("all", "yolo"):
        from ultralytics import YOLO
        for m in YOLO_MODELS:
            YOLO(m + ".pt").train(data=os.path.join(W, "full", "data.yaml"), imgsz=YOLO_IMGSZ, epochs=150, patience=40,
                                  batch=-1, device=dev, project=os.path.join(W, "runs"), name=m, exist_ok=True, seed=0, workers=2)
    if which in ("all", "rfdetr"):
        from rfdetr import RFDETRSmall
        RFDETRSmall().train(dataset_dir=os.path.join(W, "tiles"), epochs=50, batch_size=4, grad_accum_steps=4, resolution=TILE,
                            output_dir=os.path.join(W, "runs", "rfdetr_s"), early_stopping=True, seed=0)


def nms(b, iou=0.5):
    import torch, torchvision
    if len(b) == 0:
        return b
    t = torch.tensor(b, dtype=torch.float32)
    return t[torchvision.ops.nms(t[:, :4], t[:, 4], iou)].numpy()


def predict(extra=None):
    files = sorted(glob.glob(os.path.join(W, "full", "images", "test", "*.jpg")) + glob.glob(os.path.join(W, "full", "images", "val", "*.jpg")))
    if extra:
        files += sorted(glob.glob(os.path.join(extra, "*.bmp")) + glob.glob(os.path.join(extra, "*.jpg")))
    runs = []
    from ultralytics import YOLO
    for m in YOLO_MODELS:
        model = YOLO(os.path.join(W, "runs", m, "weights", "best.pt"))
        runs.append((m, lambda im, model=model: np.c_[(r := model.predict(im, imgsz=YOLO_IMGSZ, conf=0.01, iou=0.5, max_det=1000, verbose=False)[0]).boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]))
    from rfdetr import RFDETRSmall
    ck = sorted(glob.glob(os.path.join(W, "runs", "rfdetr_s", "checkpoint_best*.pth")))[0]
    rf = RFDETRSmall(pretrain_weights=ck, resolution=TILE)

    def rf_predict(im):
        w0 = im.size[0]; s = 4000 / w0 if w0 < 3000 else 1.0          # HD frames scaled to the training resolution
        if s != 1.0:
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC)
        out = []
        for x, y in tiles(*im.size):
            d = rf.predict(im.crop((x, y, x + TILE, y + TILE)), threshold=0.01)
            if len(d):
                out.append(np.c_[d.xyxy + [x, y, x, y], d.confidence])
        return nms(np.vstack(out)) / [s, s, s, s, 1] if out else np.zeros((0, 5))
    runs.append(("rfdetr_s", rf_predict))
    for m, fn in runs:
        out, t = {}, []
        for f in files:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            t0 = time.perf_counter(); b = fn(im); t.append(time.perf_counter() - t0)
            out[os.path.splitext(os.path.basename(f))[0]] = np.round(b, 2).tolist()
        out["_time_per_image_s"] = float(np.median(t[1:]))
        json.dump(out, open(os.path.join(W, f"pred_{m}.json"), "w"))
        print(m, len(files), "images,", round(out["_time_per_image_s"], 3), "s per image", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        prepare()
    elif cmd == "train":
        train(sys.argv[2] if len(sys.argv) > 2 else "all")
    elif cmd == "predict":
        predict(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
