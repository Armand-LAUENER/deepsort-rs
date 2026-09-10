# Changelog

## Jalon 2 — Tracker complet

- Tracker DeepSORT complet en Rust : `metrics.rs` (IoU, cosinus), `assignment.rs` (Hungarian via `pathfinding::kuhn_munkres_min` + gating chi², padding pour matrices rectangulaires), `cascade.rs` (matching cascade par âge + passage IoU), `track.rs` (cycle de vie tentative/confirmé/supprimé), `tracker.rs` (orchestration par frame, banque de features bornée par `nn_budget`).
- API Python `Tracker(max_age, n_init, max_cosine_distance, nn_budget, max_iou_distance).update(boxes, confidences, embeddings) -> list[Track]` conforme au §5/§6 de `PROJECT.md`.
- `tests/test_parity.py` : comparaison directe contre `deep_sort_realtime` réellement installé (`embedder=None`) sur une séquence synthétique multi-objets (apparition/disparition, occlusion temporaire, cycle de vie resserré) — IDs de piste et boîtes identiques à 1e-4 près, sur les deux jeux de paramètres testés (défauts, et `max_age=5/n_init=2` pour forcer confirmation/suppression/rattrapage IoU).
- **Limite assumée** : le critère de fin officiel du Jalon 2 (§7) porte sur la séquence VisionCam + 2 séquences MOT17 (Jalon 0), qui ne sont pas disponibles dans cet environnement. La parité algorithmique est validée contre la vraie référence sur données synthétiques ; la parité sur données réelles reste à faire une fois ces séquences enregistrées.

## Jalon 1 — Kalman

- Squelette maturin (`Cargo.toml`, `pyproject.toml`, `python/deepsort_rs/`) qui compile et s'installe en editable dans un venv.
- `KalmanFilter` (état 8D, mesure 4D, modèle à vitesse constante) implémenté en Rust (`src/kalman.rs`) depuis les équations du papier Wojke et al. 2017, exposé à Python via PyO3 (`src/lib.rs`).
- `tests/test_kalman.py` : parité contre une référence NumPy indépendante (`tests/reference_kalman.py`) sur trajectoires synthétiques — `initiate`, `predict`, `project`, `update`, `gating_distance` — écart < 1e-5 (5/5 tests passent).
- `docs/veille-similari.md` : vérification préalable de `similari` (pas de matching cascade par âge ni de gating chi² dans son code) et versions de crates figées via `cargo add --dry-run`.
