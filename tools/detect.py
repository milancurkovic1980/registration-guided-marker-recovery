"""Find the markers in photographs.

    python tools/detect.py examples/20261021-104535.jpg
    python tools/detect.py photos/                      (all photographs in a folder)

For every photograph a folder results/<photo>/ is written with
    preview.jpg        the photograph with the candidates: green = accepted as marker, red = rejected
    candidates.txt     all candidate quadrilaterals of the detector (corners P0..P3, x y in pixels)
    markers.txt        the candidates accepted by the verifier
    crops/             64 x 64 crop of every candidate, k_grey.png (grey image) and k_bin.png (binarized image),
                       k = number of the candidate in candidates.txt

Options
    --mode fixed|adaptive|adaptive_smoothed   setting of the detector (default adaptive)
    --verifier PATH       trained verifier, e.g. models/adaptive/ResNet-18_bin (default: models/<mode>/HOG-SVM_bin)
    --no-verifier         keep all candidates (no decision)
    --out FOLDER          output folder (default results)"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image, ImageDraw
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from markerdet import detect, write_candidates, load_rgb

MODES = {"fixed": ("fixed", 0.0), "adaptive": ("adaptive", 0.0), "adaptive_smoothed": ("adaptive", 0.6)}


def preview(path, rgb, quads, ok):
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    w = max(2, rgb.shape[1] // 800)
    for q, a in zip(quads, ok):
        d.line([tuple(p) for p in q] + [tuple(q[0])], fill=(0, 230, 0) if a else (240, 30, 30), width=w)
    im.save(path, quality=92)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="photographs or folders")
    ap.add_argument("--mode", default="adaptive", choices=list(MODES))
    ap.add_argument("--verifier"); ap.add_argument("--no-verifier", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    photos = []
    for p in a.inputs:
        photos += sorted(glob.glob(os.path.join(p, "*.jpg")) + glob.glob(os.path.join(p, "*.png"))) if os.path.isdir(p) else [p]
    ver = None
    if not a.no_verifier:
        from markerdet.verify import load_verifier
        ver = load_verifier(a.verifier or os.path.join(ROOT, "models", a.mode, "HOG-SVM_bin"))
    for p in photos:
        name = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(a.out, name); os.makedirs(os.path.join(out, "crops"), exist_ok=True)
        det = detect(p, *MODES[a.mode])
        ok = np.ones(len(det.quads), bool)
        if ver is not None and len(det.quads):
            ok = ver.accept(det.bin_crops if ver.input == "bin" else det.grey_crops)
        write_candidates(os.path.join(out, "candidates.txt"), det.quads)
        write_candidates(os.path.join(out, "markers.txt"), det.quads[ok])
        for k in range(len(det.quads)):
            Image.fromarray(det.grey_crops[k]).save(os.path.join(out, "crops", f"{k}_grey.png"))
            Image.fromarray(det.bin_crops[k]).save(os.path.join(out, "crops", f"{k}_bin.png"))
        preview(os.path.join(out, "preview.jpg"), load_rgb(p), det.quads, ok)
        print(f"{p}: {len(det.quads)} candidates, {int(ok.sum())} markers -> {out}", flush=True)


if __name__ == "__main__":
    main()
