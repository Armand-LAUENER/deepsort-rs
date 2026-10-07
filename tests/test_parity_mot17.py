"""Parité sur données réelles : détections publiques MOT17 et embeddings
MobileNetV2 enregistrés dans `tests/fixtures/` (cf. `scripts/record_fixture.py`
et `tests/fixtures/README.md`).

Les deux trackers rejouent exactement les mêmes entrées ; ils doivent
produire les mêmes pistes (IDs, âge, confirmation, boîtes à 1e-4), pistes
tentatives comprises, à chaque frame.
"""
from pathlib import Path

import numpy as np
import pytest
from deep_sort_realtime.deepsort_tracker import DeepSort

from deepsort_rs import Tracker
from test_parity import assert_frame_parity

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("mot17-*.npz"))


def frames(fixture):
    """Par frame : (boîtes ltwh, confiances, embeddings float32), dans l'ordre enregistré."""
    data = np.load(fixture)
    frame, ltwh, conf = data["frame"], data["ltwh"], data["conf"]
    embedding = data["embedding"].astype(np.float32)
    for frame_id in range(1, int(data["n_frames"]) + 1):
        rows = np.flatnonzero(frame == frame_id)
        yield ltwh[rows], conf[rows], embedding[rows]


def test_fixtures_are_present():
    assert len(FIXTURES) >= 2, "fixtures MOT17 manquantes dans tests/fixtures/"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
@pytest.mark.parametrize("max_age,n_init", [(30, 3), (5, 2)])
def test_tracker_matches_deep_sort_realtime_on_mot17(fixture, max_age, n_init):
    params = dict(max_age=max_age, n_init=n_init, max_cosine_distance=0.2, nn_budget=100, max_iou_distance=0.7)
    rust_tracker = Tracker(**params)
    ref_tracker = DeepSort(**params, embedder=None, nms_max_overlap=1.0)

    for frame, (ltwh, conf, embeds) in enumerate(frames(fixture), start=1):
        boxes_xyxy = np.column_stack([ltwh[:, :2], ltwh[:, :2] + ltwh[:, 2:]]).astype(np.float32)
        rust_tracks = rust_tracker.update(boxes_xyxy, conf, embeds)

        # La référence reçoit les boîtes que voit le Rust, reconstruites depuis
        # le même xyxy float32 : sinon l'arrondi f32 de `l + w` (pas de 1,2e-4
        # vers x = 1200) dépasse à lui seul la tolérance de 1e-4.
        xyxy = boxes_xyxy.astype(np.float64)
        ref_ltwh = np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]])
        raw_detections = [(list(box), float(c), "person") for box, c in zip(ref_ltwh, conf)]
        ref_tracks = ref_tracker.update_tracks(raw_detections, embeds=list(embeds))

        assert_frame_parity(frame, rust_tracks, ref_tracks)
