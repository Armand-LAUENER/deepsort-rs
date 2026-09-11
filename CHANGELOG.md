# Changelog

## Jalon 3 — Intégration VisionCam

- `confirmed` exposé en 7ᵉ colonne de sortie, et `is_confirmed()` / `to_ltrb()`
  ajoutés au dataclass `Track` : `update` renvoie aussi les pistes tentatives,
  que l'appelant doit pouvoir écarter comme il le fait avec la référence.
- **Correctif de parité** (`src/assignment.rs`) : `min_cost_matching` renvoyait
  ses non-appariés dans l'ordre croissant des indices, alors que la référence
  liste d'abord ceux que l'assignation n'a jamais retenus, puis ceux qu'elle a
  retenus avant de les rejeter sur le seuil de coût. Le tracker crée les
  nouvelles pistes dans cet ordre : deux `track_id` finissaient intervertis.
  Invisible sur la séquence synthétique du Jalon 2 (le cas demande, dans un
  même appel, une détection jamais retenue *et* une rejetée sur le coût), mais
  visible dès MOT17-04 : 99 frames divergentes sur 300. Couvert par des tests
  unitaires Rust dans `assignment.rs`.
- Parité vérifiée sur données réelles via `tools/bench_tracker.py` de VisionCam
  (YOLOv8-Pose + MobileNetV2) : **0 divergence sur 350 frames** comparées, IDs
  et boîtes, pistes tentatives comprises, sur MOT17-04 et MOT17-09.
- Vitesse mesurée sur le chemin d'intégration réel (embedder MobileNetV2
  inclus, identique des deux côtés) : 24,4 ms → 17,2 ms par frame sur MOT17-04
  (×1,42), 31,1 ms → 25,9 ms sur MOT17-09 (×1,20). L'embedder domine ce temps,
  c'est lui qui borne le gain visible dans le pipeline — le ×5,8 à ×20,6 du
  Jalon 4 porte sur l'association seule.
- **Non fait** : MOTA/IDF1 (MOT17 avec annotations non disponible), wheels PyPI,
  CI.

## Jalon 4 (partiel) — Bench vitesse

- `scripts/bench.py` : mesure tracker seul (détections/embeddings pré-calculés, warm-up 50 frames, médiane+p95 sur 500 frames) à 10/50/200 objets/frame, contre `deep_sort_realtime` et `norfair` réellement installés — méthode conforme à `PROJECT.md` §8.
- Résultats publiés dans `README.md` : `deepsort_rs` de 5,8× à 20,6× plus rapide que `deep_sort_realtime` selon la densité (l'écart se réduit à haute densité car l'assignation Hungarian est O(n³) des deux côtés — plafond algorithmique partagé, documenté tel quel).
- Bug de méthodologie trouvé et corrigé en cours de route : la distance cosinus (`src/metrics.rs`) renormalisait chaque vecteur à chaque paire comparée au lieu d'une normalisation unique en entrée de frame (coût quadratique inutile, masquait le vrai gain à haute densité).
- **Non fait** : MOTA/IDF1 (nécessite MOT17, non disponible), gain end-to-end pipeline VisionCam (nécessite le Jalon 3), publication des wheels sur PyPI, CI.

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
