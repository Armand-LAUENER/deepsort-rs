# deepsort-rs — Tracker multi-objets DeepSORT en Rust avec bindings Python

> Nom de travail. À remplacer avant publication (vérifier la disponibilité sur crates.io et PyPI).

## 1. Résumé

Réimplémentation du tracker DeepSORT (Kalman + association apparence/mouvement + matching cascade) en Rust, exposée en Python via PyO3, installable par `pip install` sans dépendance à PyTorch. Cible : remplacer `deep_sort_realtime` dans VisionCam et, plus largement, offrir un tracker rapide, thread-safe et léger à tout pipeline de détection Python.

Le projet ne réécrit **pas** l'embedder (MobileNet, TorchReID, ArcFace…) : le calcul des vecteurs d'apparence reste côté Python/GPU. Seule la logique de tracking est portée.

## 2. Problème

Les implémentations Python de DeepSORT (`deep_sort_realtime`, forks du dépôt original de Wojke, `norfair`) sont en Python pur : boucle par track, filtre de Kalman NumPy par objet, matrice de coût calculée en boucles. À 30 fps avec plusieurs dizaines d'objets, le tracker devient un goulot mesurable, et le GIL empêche de paralléliser sur plusieurs flux. Ces packages traînent aussi des dépendances lourdes (torch pour l'embedder intégré) et sont inégalement maintenus.

Il n'existe pas, côté Python, de tracker DeepSORT compilé, léger et bien maintenu. La crate Rust `similari` existe mais son état de maintenance et son orientation (plateforme In-Sight) sont à vérifier — voir §9.

## 3. Hypothèse testée

> Un tracker DeepSORT en Rust produit **exactement les mêmes tracks** que `deep_sort_realtime` sur une même séquence de détections, en étant **10 à 50× plus rapide** sur la partie tracking, sans dépendance à torch.

Les deux moitiés de l'hypothèse comptent. La vitesse sans la parité ne vaut rien : un tracker « plus rapide » qui change les IDs n'est pas comparable.

## 4. Critères de succès

- **Parité** : zéro écart d'ID et de boîte (tolérance 1e-4) contre `deep_sort_realtime` sur au moins trois séquences enregistrées (VisionCam + deux séquences MOT17).
- **Qualité** : MOTA et IDF1 sur MOT17 (détections publiques) égaux à la référence à ±0,5 point.
- **Vitesse** : temps par frame du tracker seul mesuré à 10, 50 et 200 objets contre `deep_sort_realtime` et `norfair`, publié dans le README avec la méthode.
- **Intégration** : VisionCam tourne avec le tracker Rust à la place de l'ancien, gain end-to-end mesuré et documenté.
- **Distribution** : wheels Linux / Windows / macOS publiés sur PyPI via CI, `pip install` fonctionne dans un venv vierge.

Le projet est terminé quand ces cinq points sont cochés. Tout le reste est extension.

## 5. Périmètre

### Inclus (MVP)

- Filtre de Kalman 8 états (cx, cy, aspect, h + vitesses), prédiction / mise à jour, initialisation de covariance conforme à la référence.
- Distance cosinus sur embeddings avec `nn_budget` (features accumulées par track) ; distance de Mahalanobis avec gating au seuil chi² à 4 ddl (9,4877).
- Matching cascade par âge de track, assignation Hungarian par niveau, second passage IoU pour les non-appariés.
- Cycle de vie des tracks : tentative → confirmé → supprimé (`n_init`, `max_age`).
- API Python : `Tracker(max_age=30, n_init=3, max_cosine_distance=0.2, nn_budget=100, max_iou_distance=0.7)` et `update(boxes, confidences, embeddings) -> list[Track]`. Entrées NumPy, zero-copy.
- Harnais de validation : enregistrement de séquences (détections + embeddings) en `.npz`, diff automatique contre la référence.
- Bench reproductible (`scripts/bench.py`) et rapport.

### Exclus

- Embedder (reste en Python).
- Détecteur (YOLO, RT-DETR… reste en Python).
- ByteTrack, OC-SORT, BoT-SORT : extensions possibles après le MVP, pas avant.
- Support GPU côté Rust : inutile, le tracking est CPU-bound et léger.
- Interface async / streaming : hors sujet pour un MVP.

## 6. Architecture

```
deepsort-rs/
├── Cargo.toml
├── pyproject.toml            # maturin
├── src/
│   ├── lib.rs                # module PyO3, conversions NumPy ↔ Rust
│   ├── kalman.rs             # KalmanFilter : predict / update / gating_distance
│   ├── track.rs              # Track : état, features, cycle de vie
│   ├── metrics.rs            # cosinus, IoU, Mahalanobis, matrices de coût
│   ├── assignment.rs         # Hungarian (via crate) + gating
│   ├── cascade.rs            # matching cascade + passage IoU
│   └── tracker.rs            # Tracker : orchestration d'une frame
├── python/deepsort_rs/
│   └── __init__.py           # API publique, dataclass Track côté Python
├── tests/
│   ├── test_kalman.py        # vs. implémentation NumPy de référence
│   ├── test_parity.py        # diff sur séquences .npz
│   └── fixtures/*.npz
├── scripts/
│   ├── record_sequence.py    # génère un .npz depuis VisionCam ou MOT17
│   ├── bench.py              # vitesse vs deep_sort_realtime / norfair
│   └── eval_mot.py           # MOTA / IDF1 via motmetrics
└── .github/workflows/ci.yml  # tests + build wheels (maturin-action)
```

Crates : `nalgebra` (Kalman, matrices statiques 8×8), `pathfinding` ou `lapjv` (assignation), `numpy` + `pyo3` (bindings), `rayon` (matrice de coût en parallèle si N > seuil), `thiserror` (erreurs).

Frontière Python/Rust : une seule fonction `update` par frame. Entrées `boxes: float32[N,4]` (xyxy), `confidences: float32[N]`, `embeddings: float32[N,D]` en lecture seule zero-copy. Sortie : tableau `[M, 6]` (track_id, x1, y1, x2, y2, age) converti en objets côté Python. Pas d'objets Python complexes qui traversent la frontière. Le GIL est libéré pendant `update`.

## 7. Jalons

| # | Livrable | Critère de fin | Estimation |
|---|----------|----------------|------------|
| 0 | Séquence de validation | `.npz` enregistré depuis VisionCam ; sortie de référence de `deep_sort_realtime` figée | 1 soirée |
| 1 | Kalman | Squelette maturin qui compile ; `test_kalman.py` passe (écart < 1e-5 vs NumPy sur trajectoires synthétiques) | 1 semaine |
| 2 | Tracker complet | `test_parity.py` à zéro écart sur la séquence VisionCam, puis sur 2 séquences MOT17 | 2 semaines (compter le double) |
| 3 | API + intégration | VisionCam fonctionne avec `deepsort_rs` ; gain end-to-end mesuré | 3–4 jours |
| 4 | Bench, qualité, publication | Tableau vitesse + MOTA/IDF1 dans le README ; wheels sur PyPI ; CI verte | 1 semaine |

Total réaliste : 5 à 6 semaines à temps partiel. Le jalon 2 est celui qui déborde : les écarts viendront de détails (ordre de traitement dans la cascade, accumulation des features, arrondis), pas du Rust.

Chaque jalon se termine par un commit taggé et une ligne dans `CHANGELOG.md`. On ne passe pas au suivant tant que le critère de fin n'est pas atteint.

## 8. Méthode de benchmark

- Même machine, même séquence, même détections et embeddings pré-calculés (le tracker est mesuré seul, jamais avec le détecteur).
- Warm-up de 50 frames, mesure sur ≥ 500 frames, médiane et p95 du temps par frame.
- Trois candidats : `deepsort_rs`, `deep_sort_realtime`, `norfair` (config équivalente autant que possible, écarts documentés).
- Trois densités : 10, 50, 200 objets par frame (séquences MOT17 choisies en conséquence, ou détections dupliquées/bruitées pour la densité 200).
- Résultat rapporté en deux chiffres distincts : speedup du tracker seul et gain sur le pipeline VisionCam complet. Le second sera bien plus faible (loi d'Amdahl) et c'est celui qu'on met en avant honnêtement.

## 9. Risques et points à vérifier

- **`similari`** : ~~vérifier avant le jalon 1~~ — vérifié (voir `docs/veille-similari.md`). Lecture directe du code (`voting.rs`, `sort.rs`, `track_attributes.rs`) : pas de matching cascade par âge, pas de gating chi², seuil Mahalanobis fixe. Pas un doublon de `deepsort-rs`, pas de repositionnement nécessaire ; à citer en référence dans le README final.
- **Licence** : `deep_sort_realtime` est probablement MIT, le `deep_sort` original de Wojke est GPL-3. À confirmer. Dans tous les cas : implémentation depuis le papier et les équations, pas traduction du code. Licence cible : MIT ou Apache-2.0.
- **Parité impossible à 100 %** : certains écarts peuvent venir de comportements non déterministes ou de bugs de la référence. Si un écart est documenté et justifié (la référence a tort), il est accepté ; sinon il bloque.
- **Spécificité visages** : dans VisionCam, les embeddings ArcFace sont bien plus discriminants qu'un ReID piéton ; `max_cosine_distance` et `nn_budget` devront être réglés. C'est un réglage à documenter, pas une divergence d'algorithme.
- **Apprentissage Rust** : le jalon 1 sert aussi de prise en main (ownership, structs, tests). Si ça bloque plus de deux semaines, réduire le périmètre du jalon 1 au Kalman sans bindings.
- **Over-engineering** : pas de trait générique « Tracker », pas d'abstraction multi-algorithmes avant qu'un second algorithme existe.

## 10. Extensions (après le MVP uniquement)

1. ByteTrack (association en deux passes par score, sans embeddings) — le plus demandé, simple à ajouter sur la même base.
2. OC-SORT / BoT-SORT.
3. Bindings pour d'autres langages (Node via napi-rs) si un usage se présente.

## 11. Références

- Wojke, Bewley, Paulus — *Simple Online and Realtime Tracking with a Deep Association Metric* (ICIP 2017), arXiv:1703.07402.
- Bewley et al. — *Simple Online and Realtime Tracking* (SORT), arXiv:1602.00763.
- `deep_sort_realtime` (levan92, GitHub) — implémentation de référence pour la parité.
- MOT17 — jeu de données et détections publiques pour l'évaluation ; `motmetrics` pour MOTA / IDF1.
- PyO3 + maturin — guide utilisateur ; crate `numpy` pour les bindings zero-copy.
