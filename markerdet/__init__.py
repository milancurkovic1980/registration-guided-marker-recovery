"""markerdet: detector of photogrammetric markers (dark square with a bright dot) based on local binarization.

    from markerdet import detect
    det = detect("photo.jpg", mode="adaptive")    # det.quads, det.grey_crops, det.bin_crops (64 x 64)
"""
from dataclasses import dataclass
import numpy as np

from .imaging import load_rgb, load_grey, norm_palette_u8
from .extract import Params, binarize, find_quads, find_markers
from .crops import rectify

__version__ = "1.0.0"


@dataclass
class Detection:
    quads: np.ndarray       # (n, 4, 2) corners P0..P3, (x, y) in pixels
    grey: np.ndarray        # grey image
    binary: np.ndarray      # binarized image
    grey_crops: np.ndarray  # (n, 64, 64)
    bin_crops: np.ndarray   # (n, 64, 64)


def detect(image, mode="adaptive", sigma=0.0, params=None, crop_size=64):
    """Candidates and their crops. mode: "fixed" or "adaptive"; sigma > 0 smooths before the adaptive binarization."""
    grey = norm_palette_u8(load_grey(image))
    binary = binarize(grey, mode, sigma)
    quads = np.array(find_quads(binary, params), float).reshape(-1, 4, 2)
    crops = lambda img: (np.stack([rectify(img, q, crop_size) for q in quads]) if len(quads)
                         else np.zeros((0, crop_size, crop_size), np.uint8))
    return Detection(quads, grey, binary, crops(grey), crops(binary))


def write_candidates(path, quads):
    """Candidate file: "Marker k:" followed by four lines "x y"."""
    with open(path, "w") as f:
        for k, q in enumerate(quads):
            f.write(f"Marker {k}:\n" + "".join(f"{x:.6f} {y:.6f}\n" for x, y in q) + "\n")


def read_candidates(path):
    quads, cur = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith("Marker"):
            if cur:
                quads.append(cur)
            cur = []
        elif line:
            cur.append(tuple(map(float, line.split()[:2])))
    if cur:
        quads.append(cur)
    return np.array(quads, float).reshape(-1, 4, 2)
