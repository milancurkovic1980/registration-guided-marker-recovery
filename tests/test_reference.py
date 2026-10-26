"""The Python detector reproduces the candidates of the original compiled detector exactly.

    python -m pytest tests            (needs photos/20261020-100439.jpg and photos/20261021-195136.jpg)"""
import os, sys
import numpy as np
import pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from markerdet import detect, read_candidates

CASES = [(p, s) for p in ("20261020-100439", "20261021-195136") for s in ("fixed", "s000", "s060")]
MODE = {"fixed": ("fixed", 0.0), "s000": ("adaptive", 0.0), "s060": ("adaptive", 0.6)}


@pytest.mark.parametrize("photo,setting", CASES)
def test_candidates(photo, setting):
    img = os.path.join(ROOT, "photos", photo + ".jpg")
    if not os.path.exists(img):
        pytest.skip("photographs missing in photos/")
    ref = read_candidates(os.path.join(ROOT, "tests", "reference", f"{photo}__{setting}.txt"))
    q = detect(img, *MODE[setting]).quads
    assert q.shape == ref.shape
    assert np.abs(q - ref).max() < 1e-5
