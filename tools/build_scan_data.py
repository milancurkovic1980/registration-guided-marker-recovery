"""Copies the data of a scanner project into the layout of this repository (data/scans/<session>/<project>/).

    python tools/build_scan_data.py <scanner project folder> <session name>

From every view objectN that has an annotation (01.bmp.json) it takes the first image (01.bmp, saved losslessly as
01.png), the annotation and the marker files of the scanner (objectN__A_3d.csv quadrilateral of the classical
detector, objectN__B_3d.csv box of YOLOv12-S, objectN__E_3d.csv centroid of the dot)."""
import json, os, shutil, sys, glob
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAMES = {"A": "markers3d_quad.csv", "B": "markers3d_box.csv", "E": "markers3d_dot.csv"}


def main(src, session):
    src = os.path.abspath(src); project = os.path.basename(src.rstrip("/\\"))
    dst = os.path.join(ROOT, "data", "scans", session, project)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for js in sorted(glob.glob(os.path.join(src, "object*", "01.bmp.json"))):
        d = os.path.dirname(js); v = os.path.basename(d); o = os.path.join(dst, v); os.makedirs(o, exist_ok=True)
        a = json.load(open(js))
        json.dump(dict(image="01.png", image_size=a.get("image_size", [1920, 1080]), sure=a["sure"], unsure=a["unsure"]),
                  open(os.path.join(o, "annotation.json"), "w"))
        Image.open(os.path.join(d, "01.bmp")).convert("RGB").save(os.path.join(o, "01.png"), optimize=True)
        for m, name in NAMES.items():
            p = os.path.join(d, f"{v}__{m}_3d.csv")
            if os.path.exists(p):
                shutil.copy(p, os.path.join(o, name))
        n += 1
    print(project, n, "views ->", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
