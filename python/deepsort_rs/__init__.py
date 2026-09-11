from dataclasses import dataclass

import numpy as np

from ._deepsort_rs import CHI2_95_4DOF, KalmanFilter
from ._deepsort_rs import Tracker as _RustTracker

__all__ = ["KalmanFilter", "Tracker", "Track", "CHI2_95_4DOF"]


@dataclass
class Track:
    track_id: int
    ltrb: np.ndarray
    age: int
    confirmed: bool

    def is_confirmed(self) -> bool:
        """Present so a `Track` can stand in for `deep_sort_realtime`'s own
        track object, whose callers filter on this method."""
        return self.confirmed

    def to_ltrb(self) -> np.ndarray:
        """Same reason as `is_confirmed`: mirrors the reference API."""
        return self.ltrb


class Tracker:
    """Tracker DeepSORT. `confidences` est accepté pour la stabilité de
    l'API (comme dans deep_sort_realtime) mais n'intervient pas dans le
    matching cascade/IoU, uniquement dans un éventuel NMS en amont côté
    appelant."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.2,
        nn_budget: int | None = 100,
        max_iou_distance: float = 0.7,
    ):
        self._inner = _RustTracker(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            max_iou_distance=max_iou_distance,
        )

    def update(self, boxes, confidences, embeddings) -> list[Track]:
        del confidences
        boxes = np.ascontiguousarray(boxes, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        raw = self._inner.update(boxes, embeddings)
        return [
            Track(
                track_id=int(row[0]),
                ltrb=row[1:5],
                age=int(row[5]),
                confirmed=bool(row[6]),
            )
            for row in raw
        ]
