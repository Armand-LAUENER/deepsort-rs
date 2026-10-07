# Changelog

## Parité MOT17 dans le dépôt, qualité MOT17, CI

- **Correctif de parité** (`src/tracker.rs`) : `deep_sort_realtime` stocke
  chaque détection en float32 (`Detection.ltwh`) et calcule `to_xyah` en
  float32 ; le tracker Rust calculait la mesure en f64. Écart de 2 à 3 ulp f32
  par mise à jour, qui dépassait la tolérance de 1e-4 px sur les pistes longues
  à grandes coordonnées (1,7e-4 px vers x ≈ 650 à la frame 150 de MOT17-09).
  Antérieur au passage à `sgemm` (même écart, au chiffre près, avec le binaire
  de `d894462`), invisible sur la scène synthétique à petites coordonnées. Le
  tracker reproduit désormais ces arrondis.
- `tests/test_parity_mot17.py` + `tests/fixtures/` : parité frame par frame
  (IDs, âge, état, boîtes à 1e-4, pistes tentatives comprises) sur les
  détections publiques et embeddings MobileNetV2 de MOT17-04 (300 frames) et
  MOT17-09 (525 frames), avec `(max_age, n_init)` = (30, 3) et (5, 2).
  Embeddings projetés en 128-d float16 pour la taille du dépôt (3,4 Mo) ;
  fixtures sous CC BY-NC-SA 3.0 comme MOT17. Générées par
  `scripts/record_fixture.py`.
- MOTA/IDF1 publiés dans le README (`tools.eval_mot` de VisionCam, révision
  `8484623`) : identiques à la référence sur MOT17-04 (73,6 % / 72,3 %, 102
  changements d'ID) et MOT17-09 (51,6 % / 55,8 %, 44).
- CI GitHub Actions : `cargo test`, `cargo clippy -D warnings`, et `pytest`
  (parité) sous Python 3.11 et 3.13.
- Nom `deepsort-rs` retenu, libre sur crates.io et PyPI.
- **Non fait** : wheels PyPI.

## Perf — association rapide à forte densité

- **Constat** : à 40 personnes/frame (embeddings 512-d, `nn_budget=100`),
  l'association prenait 48,6 ms contre 11,0 ms pour `deep_sort_realtime`.
  Le coût venait des distances d'apparence de la cascade : un produit scalaire
  `f64` par paire (échantillon, détection), réduction séquentielle non
  vectorisée, sur des `Vec<Vec<f64>>` non contigus.
- `src/metrics.rs` : `FeatureMatrix`, lignes de features `f32` contiguës,
  utilisée pour la banque d'apparence (`Tracker::feature_bank`),
  `Track::features` et la matrice des embeddings de la frame (normalisée une
  fois, norme accumulée en `f64`).
- `nn_cosine_costs` : par piste, `C = S · Qᵀ` en un appel
  `matrixmultiply::sgemm`, puis `cost[j] = 1 - max_i C[i][j]` — même schéma
  que le produit matriciel numpy de la référence. Nouvelle dépendance directe
  `matrixmultiply = "0.3"` (déjà présente en transitif via nalgebra).
- `PyTracker::update` relâche le GIL pendant le calcul (`py.detach`) : deux
  trackers dans deux threads Python passent de 12,25 s en série à 6,28 s
  (×1,95).
- **Parité** : `tests/test_parity.py` vert ; 0 frame à IDs divergents contre
  `deep_sort_realtime` sur le banc d'association ci-dessous (k = 3, 10, 40).
  Le passage en `f32` n'a pas nécessité le repli `dgemm`.
- Banc d'association (300 frames, 512-d, `OPENBLAS_NUM_THREADS=1`, WSL2,
  release) :

  | Personnes/frame | deep_sort_realtime | avant | après |
  |---:|---:|---:|---:|
  | 3 | 0,82 ms | 0,29 ms | 0,12 ms |
  | 10 | 2,59 ms | 3,15 ms | 0,50 ms |
  | 40 | 10,97 ms | 48,63 ms | 3,56 ms |

- `scripts/bench.py` (médiane, embeddings 32-d, même machine, avant → après) :
  10 objets 0,136 → 0,043 ms ; 50 objets 3,270 → 0,427 ms ; 200 objets
  47,6 → 4,9 ms.
- **Non fait** : MOT17-04 (`tools.eval_mot`) et 4 caméras, à mesurer côté
  VisionCam après le bump de révision.

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
