"""Candidates of the detector.

    python experiments/extract.py                   all photographs, three settings, with crops
    python experiments/extract.py --original        original extraction limits
    python experiments/extract.py --binarizations   extraction on other binarizations of the annotated photographs

Output: work/candidates/<tag>/<photo>.txt and work/crops/<setting>/<photo>.npz (quads, G, B, B50).
Finished photographs are skipped."""
import os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "1")      # photographs run in parallel
from multiprocessing import Pool
import numpy as np
from common import WORK, SETTINGS, METHODS, photo_table, photo_path, grey_stretched, doxa_binarize
from markerdet import Params, binarize, find_quads, rectify, load_grey, norm_palette_u8, write_candidates

ORIGINAL = "--original" in sys.argv


def save(tag, photo, quads):
    txt = os.path.join(WORK, "candidates", tag, photo + ".txt")
    os.makedirs(os.path.dirname(txt), exist_ok=True)
    write_candidates(txt + ".tmp", quads); os.replace(txt + ".tmp", txt)


def done(tag, photo):
    return os.path.exists(os.path.join(WORK, "candidates", tag, photo + ".txt"))


def detector(photo):
    grey = None
    for name, (mode, sigma) in SETTINGS.items():
        tag = name + ("_original" if ORIGINAL else "")
        if done(tag, photo):
            continue
        grey = norm_palette_u8(load_grey(photo_path(photo))) if grey is None else grey
        binary = binarize(grey, mode, sigma)
        quads = np.array(find_quads(binary, Params.legacy() if ORIGINAL else Params()), float).reshape(-1, 4, 2)
        if not ORIGINAL:
            crop = lambda img, s=64: np.array([rectify(img, q, s) for q in quads], np.uint8).reshape(-1, s, s)
            os.makedirs(os.path.join(WORK, "crops", name), exist_ok=True)
            np.savez_compressed(os.path.join(WORK, "crops", name, photo + ".npz"),
                                quads=quads, G=crop(grey), B=crop(binary), B50=crop(binary, 50))
        save(tag, photo, quads)
    return photo


def other_binarizations(photo):
    """Otsu and three windows (half, equal, twice the fixed window) of the local methods, with and without the appearance test."""
    g = grey_stretched(photo); r0 = min(g.shape) // 20
    for m in METHODS:
        for w in [None] if m == "OTSU" else [2 * (r0 // 2) + 1, 2 * r0 + 1, 4 * r0 + 1]:
            b = None
            for app in (True, False):
                tag = f"b-{m.lower()}" + ("" if w is None else f"-w{w}") + ("" if app else "-noapp")
                if not done(tag, photo):
                    b = doxa_binarize(g, m, w) if b is None else b
                    save(tag, photo, find_quads(b, Params(appearance_test=app)))
    return photo


if __name__ == "__main__":
    T = photo_table()
    if "--binarizations" in sys.argv:
        fn, photos = other_binarizations, sorted(p for p, r in T.items() if r["annotation"])
    else:
        fn, photos = detector, sorted(T)
    with Pool() as pool:
        for k, p in enumerate(pool.imap_unordered(fn, photos), 1):
            print(f"[{k}/{len(photos)}] {p}", flush=True)
