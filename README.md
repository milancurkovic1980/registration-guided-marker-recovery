# registration-guided-marker-recovery

Code and data of the article

> M. Ćurković, A. Ćurković, *Registration-Guided Recovery of Markers Lost to Projector Glare in Smartphone
> Structured-Light Scanning*, Optics and Lasers in Engineering (submitted).

A structured-light scanner registers its views through markers attached to the object. Under the white light of the
projector, specular highlights on glossy surfaces hide many of them. The registration-guided recovery projects the
markers of views that are already registered into a view and detects them again with a local binarization around
the predicted position, without training data or a graphics processing unit.

The repository contains

* **`markerdet/`**: Python implementation of the classical marker detector (local binarization, candidate
  extraction, 64 x 64 crops of the candidates);
* **`recovery/`**: registration of views from the 3D points of the markers and the registration-guided recovery;
* **`models/`**: the ellipse test and HOG-SVM verifiers, and YOLOv12-N/S in `models/learned/`
  (RF-DETR-S and ResNet-18 are attached to the release, see below);
* **`data/scans/`**: the two scanning sessions of the article, 69 views in all (see below);
* **`photos/`** and **`data/annotations`, `data/labels`, `data/photos.csv`**: the 215 photographs used to compare
  the verifiers and detectors, with the labels of all candidates and the annotated markers;
* **`experiments/`**: the experiments of the article (`experiments/scanner/` for the scanning sessions, the other
  scripts for the photographs);
* **`tools/`**: detection on photographs (`detect.py`) and conversion of scanner projects (`build_scan_data.py`).

The markers are dark squares with a bright circular dot (dot diameter 45% of the side).

## Scanning sessions

```
data/scans/session1/<project>/objectN/        38 views: flat side and side with spheres of the calibration
                                              object, grey desk pad, replica of a head sculpture
data/scans/session2_glare/<project>/objectN/  31 views: glazed porcelain plate (objects 4-11), bowl (1-3, 12-15)
                                              and cup (16-31) under projector glare
```

Every view contains

* `01.png`: the image under the white light of the projector on which the markers are detected (1920 x 1080 px);
* `annotation.json`: the annotated markers (`sure` and `unsure`, one click per marker, not an exact centre);
* `markers3d_quad.csv`, `markers3d_box.csv`, `markers3d_dot.csv`: the markers found with the classical detector
  (centre of the quadrilateral), with YOLOv12-S (centre of the box) and the centroid of the dot, each with its
  3D point computed by the scanner (`id, x, y, X, Y, Z, n, rms`; `n` and `rms` describe the plane fitted to the
  reconstructed points around the marker). The same `id` is the same marker in the three files of a view.

The focal length and the principal point of the camera, needed to project the markers into a view, are fitted to
the pixels and 3D points of the markers of each project (`Session.camera`). The reconstructed surfaces and the
calibration of the scanner are not included.

## Installation

```
git clone https://github.com/milancurkovic1980/registration-guided-marker-recovery
cd registration-guided-marker-recovery
pip install -r requirements.txt
```

The learned detectors need `ultralytics` (YOLOv12) and `rfdetr` (RF-DETR) with PyTorch, see `requirements.txt`.

## Reproducing the results on the scanning sessions

```
python experiments/scanner/detect.py                 # classical detector on all views (add --learned for YOLO, RF-DETR)
python experiments/scanner/evaluate_detection.py     # recall, false detections, views with 3 and 6 markers
python experiments/scanner/recovery_experiment.py    # recovery, sequential and global (add --learned)
python experiments/scanner/centres.py                # registration with the three centres
python experiments/scanner/number_of_markers.py      # registration with 3 to 10 markers and the TRE model
```

The results are written to `results/scanner/`. The predictions of the learned detectors used in the article are
included there, so the evaluation and the recovery can be run without them.

The recovery predicts the pixel of a missing marker from its 3D
point, so it needs the focal length and the principal point of the camera. The calibration of the scanner is not
included, and these values are fitted to the pixels and 3D points of the markers of each project. With the fitted
values the recovery on the glare session differs slightly from the article (sequential 79.4% instead of 79.8% with
15 instead of 16 recovered markers, global 82.8% instead of 83.6% with 24 instead of 26). On the first session only the number of false
markers of the global variant differs (8 instead of 7), and the numbers of view pairs are the same.

## Recovery on your own data

```python
from recovery import Session, classical_detections, recover_session
from recovery.core import quad_side

ses = Session("data/scans/session2_glare")
dets = {}
for v in ses.views:
    q = classical_detections(ses.path(v))
    dets[v] = dict(centres=q.mean(1) if len(q) else q.reshape(0, 2), side=quad_side(q))
result = recover_session(ses, dets, mode="sequential")      # or "global"
print(result["26_10_2026_17_03_53/object10"]["recovered"])  # centres of the recovered markers
```

## Photographs

The experiments on the photographs (candidate extraction, verifiers, learned detectors) are in `experiments/`
(`extract.py`, `recall.py`, `train_verifiers.py`, `learned_detectors.py`). Markers in a photograph are found with

```
python tools/detect.py photos/20261021-104535.jpg --verifier models/adaptive/Ellipse_bin
```

## Release files

* `rfdetr_s.pth`: RF-DETR-S fine-tuned on the training scenes (copy to `models/learned/`);
* `models.zip`: the fine-tuned ResNet-18 verifiers.

## Citation

See `CITATION.cff`. License: MIT (code). The data are released under CC BY 4.0.
