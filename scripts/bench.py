"""Jalon 4 : mesure du temps par frame du tracker seul, à 10/50/200 objets,
contre `deep_sort_realtime` et `norfair` — méthode décrite dans PROJECT.md §8.

Ce script mesure uniquement la partie tracking : les détections et
embeddings sont pré-calculés (pas de détecteur ni d'embedder dans la boucle
chronométrée), comme l'exige §8.

À exécuter avec un venv qui a les 3 dépendances installées. `norfair` impose
`numpy<2.0`, ce qui downgraderait le venv de dev principal (`.venv`) utilisé
par `pytest` — utiliser un venv séparé (voir README, section Benchmark) :

    python -m venv .venv-bench
    .venv-bench/Scripts/python -m pip install maturin deep_sort_realtime norfair
    .venv-bench/Scripts/python -m maturin develop --release
    .venv-bench/Scripts/python scripts/bench.py

norfair ne fait pas de matching par apparence (pas de ReID intégré dans son
usage standard) : il est configuré ici en IoU pur. C'est un écart de
configuration documenté (§8), pas une tentative de reproduire DeepSORT avec
norfair.
"""
import statistics
import time

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from norfair import Detection as NorfairDetection
from norfair import Tracker as NorfairTracker

from deepsort_rs import Tracker as RustTracker

WARMUP_FRAMES = 50
MEASURED_FRAMES = 500
DENSITIES = (10, 50, 200)
EMBEDDING_DIM = 32
FRAME_SIZE = (1920, 1080)


def make_frames(n_objects, n_frames, seed):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n_objects, EMBEDDING_DIM)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    starts = rng.uniform(
        [0, 0], [FRAME_SIZE[0] - 60, FRAME_SIZE[1] - 120], size=(n_objects, 2)
    )
    velocities = rng.normal(scale=2.0, size=(n_objects, 2))

    frames = []
    for frame_idx in range(n_frames):
        centers = starts + velocities * frame_idx + rng.normal(scale=0.5, size=(n_objects, 2))
        w, h = 40.0, 80.0
        boxes_xyxy = np.stack(
            [centers[:, 0], centers[:, 1], centers[:, 0] + w, centers[:, 1] + h], axis=1
        ).astype(np.float32)
        frames.append((boxes_xyxy, embeddings.copy()))
    return frames


def time_frames(step_fn, frames):
    for boxes, embeddings in frames[:WARMUP_FRAMES]:
        step_fn(boxes, embeddings)

    timings = []
    for boxes, embeddings in frames[WARMUP_FRAMES : WARMUP_FRAMES + MEASURED_FRAMES]:
        start = time.perf_counter()
        step_fn(boxes, embeddings)
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def bench_deepsort_rs(frames):
    tracker = RustTracker()
    confidences = None

    def step(boxes, embeddings):
        nonlocal confidences
        if confidences is None or len(confidences) != len(boxes):
            confidences = [1.0] * len(boxes)
        tracker.update(boxes, confidences, embeddings)

    return time_frames(step, frames)


def bench_deep_sort_realtime(frames):
    # nn_budget=100 pour matcher le défaut de deepsort_rs (PROJECT.md §5) :
    # laisser nn_budget=None (illimité, défaut de deep_sort_realtime) fait
    # grossir l'historique de features sans borne sur toute la mesure, ce qui
    # fausserait la comparaison de vitesse en croissance linéaire avec le
    # nombre de frames plutôt qu'en régime stationnaire.
    tracker = DeepSort(embedder=None, nms_max_overlap=1.0, nn_budget=100)

    def step(boxes, embeddings):
        raw_detections = [
            ([b[0], b[1], b[2] - b[0], b[3] - b[1]], 1.0, None) for b in boxes
        ]
        tracker.update_tracks(raw_detections, embeds=list(embeddings))

    return time_frames(step, frames)


def bench_norfair(frames):
    tracker = NorfairTracker(distance_function="iou", distance_threshold=0.5)

    def step(boxes, embeddings):
        detections = [
            NorfairDetection(points=np.array([[b[0], b[1]], [b[2], b[3]]]))
            for b in boxes
        ]
        tracker.update(detections)

    return time_frames(step, frames)


def summarize(name, timings_ms):
    return {
        "candidate": name,
        "median_ms": statistics.median(timings_ms),
        "p95_ms": statistics.quantiles(timings_ms, n=100)[94],
    }


def main():
    print(f"Warm-up : {WARMUP_FRAMES} frames, mesure : {MEASURED_FRAMES} frames, médiane + p95 par densité.\n")
    results = []
    for density in DENSITIES:
        frames = make_frames(density, WARMUP_FRAMES + MEASURED_FRAMES, seed=density)

        rows = [
            summarize("deepsort_rs", bench_deepsort_rs(frames)),
            summarize("deep_sort_realtime", bench_deep_sort_realtime(frames)),
            summarize("norfair (IoU only)", bench_norfair(frames)),
        ]
        rust_median = rows[0]["median_ms"]

        print(f"=== {density} objets/frame ===")
        for row in rows:
            speedup = row["median_ms"] / rust_median if row["candidate"] != "deepsort_rs" else 1.0
            print(
                f"  {row['candidate']:<20} médiane {row['median_ms']:7.3f} ms  "
                f"p95 {row['p95_ms']:7.3f} ms  (x{speedup:.1f} vs deepsort_rs)"
            )
            results.append({"density": density, **row})
        print()

    return results


if __name__ == "__main__":
    main()
