"""Registration-guided recovery of markers in structured-light scanning.

A view is registered to other views of the same scan through the 3D points of the markers found in both
(random sample consensus on the distances between markers, then least-squares rigid alignment). Markers of the
other view that have no counterpart are projected into the image of the view, and a local search with its own
binarization looks for a marker where none was detected."""
from .core import (Session, kabsch, match_markers, project, local_search, recover_session, classical_detections,
                   box_detections)

__all__ = ["Session", "kabsch", "match_markers", "project", "local_search", "recover_session",
           "classical_detections", "box_detections"]
