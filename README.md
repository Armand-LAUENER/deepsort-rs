# deepsort-rs

Tracker DeepSORT (Kalman + association apparence/mouvement + matching cascade) réimplémenté en Rust, exposé en Python via PyO3. Voir `PROJECT.md` pour le contexte, l'hypothèse testée et les critères de succès complets.

> Nom de travail — à confirmer avant publication (voir `PROJECT.md` §1).

## Statut

- **Jalon 1** (Kalman) : fait. `tests/test_kalman.py` (écart < 1e-5 contre une référence NumPy).
- **Jalon 2** (tracker complet) : fait. Parité contre `deep_sort_realtime` réellement installé, sur séquence synthétique (`tests/test_parity.py`) et sur données réelles via VisionCam (YOLOv8-Pose + MobileNetV2) : 0 divergence d'ID ou de boîte sur 350 frames de MOT17-04 et MOT17-09, pistes tentatives comprises.
- **Jalon 3** (intégration VisionCam) : fait. VisionCam utilise `deepsort_rs` via `TRACKER_BACKEND=rust`. Gain mesuré sur le chemin réel, embedder compris, avant l'optimisation de l'association : ×1,42 sur MOT17-04, ×1,20 sur MOT17-09 (l'embedder domine le temps par frame). Le gain après optimisation reste à mesurer côté VisionCam.
- **Jalon 4** (bench, qualité, publication) : partiel.
  - Fait : bench vitesse du tracker seul, ci-dessous.
  - Fait côté VisionCam (`tools.eval_mot`), avant l'optimisation de l'association : sur MOT17-04 (détections publiques), pistes identiques à la référence, MOTA 73,6 %, IDF1 72,3 %, 102 changements d'ID. À remesurer sur la révision actuelle et à reporter ici.
  - Pas fait : wheels sur PyPI, CI.

Voir `CHANGELOG.md` pour le détail de chaque jalon.

## Installation (dev)

```bash
python -m venv .venv
.venv/Scripts/pip install maturin
.venv/Scripts/python -m maturin develop --release
.venv/Scripts/pip install -e ".[test]"
.venv/Scripts/python -m pytest tests/
```

## Benchmark — vitesse du tracker seul

Méthode (voir `scripts/bench.py`, conforme à `PROJECT.md` §8) :

- Détections et embeddings pré-calculés hors boucle chronométrée — seul le tracker est mesuré, jamais un détecteur ou un embedder.
- Warm-up de 50 frames (hors mesure), mesure sur 500 frames, médiane et p95 par frame.
- Trois candidats : `deepsort_rs`, `deep_sort_realtime` (`embedder=None`, `nn_budget=100` pour être comparable), `norfair` (config IoU pure — **écart documenté** : norfair n'a pas de ReID par apparence dans son usage standard, ce n'est donc pas une comparaison à isoparamètres algorithmiques, seulement à isotâche "tracker appelé N fois par frame").
- Trois densités : 10, 50, 200 objets/frame, trajectoires synthétiques à vitesse constante bruitée.
- Une seule machine, une seule exécution par densité (pas de moyenne sur plusieurs runs) — à prendre comme ordre de grandeur, pas comme mesure de production.

**Machine** : AMD Ryzen 5 7600X, WSL2 Ubuntu 24.04, Python 3.13, rustc 1.98.1, build `--release`.

| Densité | Candidat | Médiane (ms) | p95 (ms) | Speedup vs `deepsort_rs` |
|---|---|---:|---:|---:|
| 10 | **deepsort_rs** | 0.043 | 0.065 | 1.0× |
| 10 | deep_sort_realtime | 1.865 | 4.712 | 43.2× plus lent |
| 10 | norfair (IoU seul) | 0.539 | 1.006 | 12.5× plus lent |
| 50 | **deepsort_rs** | 0.427 | 0.613 | 1.0× |
| 50 | deep_sort_realtime | 34.772 | 384.056 | 81.4× plus lent |
| 50 | norfair (IoU seul) | 2.996 | 7.620 | 7.0× plus lent |
| 200 | **deepsort_rs** | 4.913 | 8.892 | 1.0× |
| 200 | deep_sort_realtime | 167.650 | 573.175 | 34.1× plus lent |
| 200 | norfair (IoU seul) | 11.924 | 20.240 | 2.4× plus lent |

**Lecture honnête** :

- Le gain contre `deep_sort_realtime` va de 34× à 81× selon la densité. Le p95 de la référence est très supérieur à sa médiane (variance côté Python, non creusée), et il s'agit d'une seule exécution par densité : ce sont des ordres de grandeur.
- `scripts/bench.py` utilise des embeddings de dimension 32, ce qui sous-estime le poids des distances d'apparence. Avec des embeddings réalistes de dimension 512 (banc d'association décrit dans `CHANGELOG.md`, 300 frames, `OPENBLAS_NUM_THREADS=1`), l'écart est plus modeste : 0,12 ms contre 0,82 ms à 3 personnes, 3,56 ms contre 10,97 ms à 40 personnes (×3,1).
- Une version précédente de ce README attribuait le recul du gain à haute densité à l'assignation Hungarian en O(n³). C'était faux : le goulot était le calcul des distances cosinus (un produit scalaire `f64` séquentiel par paire, sur des `Vec<Vec<f64>>`). Depuis le passage à des matrices `f32` contiguës et à un produit matriciel `sgemm` par piste (`src/metrics.rs`, `nn_cosine_costs`), `deepsort_rs` passe de 47,6 ms à 4,9 ms de médiane à 200 objets sur cette machine, et reste devant `norfair` à toutes les densités.
- Les chiffres de la version précédente (Intel Core i7-1360P, Windows) ne sont pas comparables directement à ceux-ci : machine différente.

Deux bugs de performance ont été trouvés et corrigés en cours de route : la distance cosinus renormalisait d'abord chaque vecteur à chaque paire comparée (corrigé par une normalisation unique en entrée de frame), puis les distances d'apparence restaient calculées paire par paire, sans vectorisation (corrigé par le produit matriciel ci-dessus).

`Tracker.update` relâche le GIL pendant le calcul : plusieurs trackers (une caméra chacun) peuvent tourner en parallèle dans des threads Python.

**Non fait** : MOTA/IDF1 (nécessite MOT17, non disponible ici) et le gain end-to-end sur le pipeline VisionCam complet (nécessite l'intégration du Jalon 3, également non faite). Le chiffre ci-dessus est un speedup **tracker seul**, pas un gain pipeline — `PROJECT.md` §8 est explicite sur le fait que le second sera bien plus faible (loi d'Amdahl).

Reproduire (nécessite un venv séparé car `norfair` impose `numpy<2.0`, ce qui casserait le venv de dev principal) :

```bash
python -m venv .venv-bench
.venv-bench/Scripts/pip install maturin deep_sort_realtime norfair
.venv-bench/Scripts/python -m maturin develop --release
.venv-bench/Scripts/python scripts/bench.py
```

## Licence

MIT — voir [LICENSE](LICENSE).
