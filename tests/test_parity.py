"""Jalon 2 : le Tracker Rust doit produire exactement les mêmes IDs et des
boîtes équivalentes (tolérance 1e-4, cf. PROJECT.md §4) que `deep_sort_realtime`
sur une séquence synthétique multi-objets.

Note : ceci valide la fidélité algorithmique (cascade, gating, cycle de vie,
budget de features) contre la vraie référence de parité choisie par le
projet. Ce n'est pas un substitut à la séquence VisionCam + MOT17 prévue au
Jalon 0 (non disponibles dans cet environnement) : la parité sur données
réelles reste à vérifier séparément une fois ces séquences enregistrées.
"""
import numpy as np
import pytest
from deep_sort_realtime.deepsort_tracker import DeepSort

from deepsort_rs import Tracker

BOX_TOLERANCE = 1e-4


def ltwh_to_xyxy(ltwh):
    l, t, w, h = ltwh
    return [l, t, l + w, t + h]


class ObjectTrack:
    """Trajectoire synthétique à vitesse constante bruitée, avec pattern de
    présence/absence pour exercer la cascade et le cycle de vie."""

    def __init__(self, start, velocity, embedding, present_from=0, present_until=None, gaps=()):
        self.start = np.array(start, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.embedding = np.array(embedding, dtype=np.float32)
        self.present_from = present_from
        self.present_until = present_until
        self.gaps = set(gaps)

    def is_present(self, frame):
        if frame < self.present_from:
            return False
        if self.present_until is not None and frame >= self.present_until:
            return False
        return frame not in self.gaps

    def ltwh(self, frame, rng):
        pos = self.start + self.velocity * frame + rng.normal(scale=0.3, size=2)
        w, h = 40.0, 80.0
        return [pos[0], pos[1], w, h]


def make_scene():
    rng = np.random.default_rng(123)
    dim = 16
    embeddings = rng.normal(size=(4, dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # bien séparées

    return [
        ObjectTrack([0.0, 0.0], [3.0, 0.5], embeddings[0]),
        ObjectTrack([400.0, 300.0], [-2.0, -1.0], embeddings[1]),
        ObjectTrack([200.0, 50.0], [1.0, 2.0], embeddings[2], gaps={10, 11, 12}),
        ObjectTrack([50.0, 250.0], [2.0, -0.5], embeddings[3], present_from=8, present_until=30),
    ]


def run_both(n_frames, max_age, n_init, max_cosine_distance=0.2, max_iou_distance=0.7, nn_budget=100):
    scene = make_scene()
    rng = np.random.default_rng(7)

    rust_tracker = Tracker(
        max_age=max_age,
        n_init=n_init,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget,
        max_iou_distance=max_iou_distance,
    )
    ref_tracker = DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget,
        max_iou_distance=max_iou_distance,
        embedder=None,
        nms_max_overlap=1.0,
    )

    for frame in range(n_frames):
        ltwhs = []
        embeds = []
        for obj in scene:
            if obj.is_present(frame):
                ltwhs.append(obj.ltwh(frame, rng))
                embeds.append(obj.embedding)

        boxes_xyxy = np.array([ltwh_to_xyxy(b) for b in ltwhs], dtype=np.float32)
        embeds_arr = np.array(embeds, dtype=np.float32) if embeds else np.zeros((0, 16), dtype=np.float32)
        confidences = [1.0] * len(ltwhs)

        rust_tracks = rust_tracker.update(boxes_xyxy, confidences, embeds_arr)

        raw_detections = [(ltwh, 1.0, None) for ltwh in ltwhs]
        ref_tracks = ref_tracker.update_tracks(raw_detections, embeds=embeds if embeds else np.zeros((0, 16)))

        yield frame, rust_tracks, ref_tracks


def assert_frame_parity(frame, rust_tracks, ref_tracks):
    rust_by_id = {t.track_id: t for t in rust_tracks}
    ref_by_id = {int(t.track_id): t for t in ref_tracks}

    assert set(rust_by_id) == set(ref_by_id), (
        f"frame {frame}: ensembles de pistes différents — "
        f"rust={sorted(rust_by_id)} ref={sorted(ref_by_id)}"
    )

    for track_id, rust_track in rust_by_id.items():
        ref_track = ref_by_id[track_id]
        assert rust_track.age == ref_track.age, f"frame {frame}, piste {track_id}: age différent"
        assert rust_track.is_confirmed() == ref_track.is_confirmed(), (
            f"frame {frame}, piste {track_id}: état de confirmation différent"
        )
        np.testing.assert_allclose(
            rust_track.ltrb,
            ref_track.to_ltrb(),
            atol=BOX_TOLERANCE,
            err_msg=f"frame {frame}, piste {track_id}: boîte différente",
        )


@pytest.mark.parametrize(
    "max_age,n_init",
    [
        (30, 3),  # defaults PROJECT.md §5
        (5, 2),  # cycle de vie resserré : confirmation/suppression/rattrapage IoU plus fréquents
    ],
)
def test_tracker_matches_deep_sort_realtime(max_age, n_init):
    for frame, rust_tracks, ref_tracks in run_both(n_frames=40, max_age=max_age, n_init=n_init):
        assert_frame_parity(frame, rust_tracks, ref_tracks)
