# Veille technique — préalable au Jalon 1

> Réalisée le 2026-09-10, avant tout code, pour trancher le risque §9 de `PROJECT.md` ("vérifier `similari` avant le jalon 1") et figer les choix de crates. Sources : dépôt GitHub `insight-platform/Similari`, crates.io, code source des fichiers cités (lus directement, pas de résumé de seconde main).

## 1. Verdict sur `similari` — risque levé, pas de repositionnement

**État du dépôt** (`github.com/insight-platform/Similari`) : Apache-2.0, 265 stars, 4 issues ouvertes, non archivé. Dernier commit identifié : **26 mars 2025** (~18 mois avant aujourd'hui). Ni mort ni activement maintenu au rythme d'un projet en croissance — état intermédiaire, à surveiller mais pas bloquant.

**Lecture directe du code** (pas seulement le README) :

- `src/trackers/visual_sort/voting.rs` — le "cascade voting engine" annoncé dans les commentaires n'est **pas** la matching cascade de DeepSORT. C'est un pipeline en deux étapes fixes : vote TopN sur les features visuelles, puis vote Hungarian sur le reste. **Aucune boucle sur l'âge des tracks** (le principe même de la cascade Wojke : matcher d'abord les tracks les plus récemment vus, avant de laisser les tracks plus âgés concourir). Confirmé par lecture du fichier, pas déduit du README.
- `src/trackers/sort.rs` — Hungarian bien utilisé pour l'assignation, mais gating via une constante `MAHALANOBIS_NEW_TRACK_THRESHOLD = 1.0` fixe, pas un seuil chi² à 4 ddl appliqué comme filtre d'exclusion façon Wojke (9,4877). Pas de mention de cascade ni de budget dans ce fichier.
- `src/trackers/visual_sort/track_attributes.rs` — il existe bien un plafond sur l'historique par track (`history_length`, pop_front sur `observed_features`/`observed_boxes` une fois dépassé) : équivalent fonctionnel d'un `nn_budget`, ça au moins est présent.

**Conclusion** : `similari` est un framework bas niveau ("framework to build custom trackers") avec un `VisualSORT` *"DeepSORT-like"* qui n'implémente ni la cascade par âge ni le gating chi², les deux éléments qui conditionnent la parité exacte visée en §3-4 de `PROJECT.md`. Ce n'est donc pas un doublon du projet prévu. Pas de raison de repositionner `deepsort-rs` en contribution à `similari` ou en simple wrapper — la parité stricte contre `deep_sort_realtime` reste un objectif que `similari` ne revendique pas et n'atteint probablement pas tel quel.

**Action retenue** : mentionner `similari` dans le README final comme alternative existante (transparence, honnêteté intellectuelle), sans en dépendre.

**Autres projets Rust scannés par la même occasion** (aucun n'est un concurrent direct) :
- `mot-rs` (LdDl) — IoU/ByteTrack/centroïdes, explicitement sans ReID/embeddings. Hors sujet pour DeepSORT.
- `tracktor` — tracking multi-cible par filtres RFS (Random Finite Sets), approche différente, pas DeepSORT.

## 2. Chaîne de licences — hypothèse du §9 confirmée

- `nwojke/deep_sort` (implémentation originale du papier) : **GPL-3**, confirmé par lecture du fichier `LICENSE` du dépôt.
- `levan92/deep_sort_realtime` (référence de parité choisie) : **MIT**, confirmé.
- Conséquence pratique inchangée par rapport au doc : implémentation de `deepsort-rs` **depuis le papier et les équations**, jamais en traduisant le code de Wojke (GPL-3) ni celui de `deep_sort_realtime` — même si ce dernier est MIT, la logique métier vient du papier pour rester propre. Licence cible MIT/Apache-2.0 confirmée viable.

## 3. Fidélité de `deep_sort_realtime` comme référence de parité

Lecture de `deepsort_tracker.py` — les defaults cités en §5 de `PROJECT.md` sont exacts, vérifiés dans le code source (pas supposés) :

| Paramètre | Défaut confirmé |
|---|---|
| `max_age` | 30 |
| `n_init` | 3 |
| `max_cosine_distance` | 0.2 |
| `nn_budget` | `None` (illimité) |
| `max_iou_distance` | 0.7 |

Point utile pour le harnais de validation (Jalon 0) : `deep_sort_realtime` a un embedder intégré (`mobilenet` par défaut) mais accepte `embedder=None`, auquel cas les embeddings doivent être fournis à `update_tracks()`. C'est le mode à utiliser pour la génération des séquences de référence : on injecte les mêmes embeddings pré-calculés des deux côtés (Rust et Python), sinon un écart de tracking pourrait venir de l'embedder et fausserait le diagnostic de parité.

## 4. Crates Rust — versions résolvables réellement (pas des suppositions)

Obtenues via `cargo add --dry-run` (résolution réelle contre l'index crates.io), pas via recherche web :

| Crate | Version résolue | Rôle |
|---|---|---|
| `nalgebra` | 0.35.0 | Kalman, matrices statiques 8×8 |
| `pyo3` | 0.29.2 | bindings Python |
| `numpy` | 0.29.0 | ponts NumPy zero-copy (aligné sur la version de pyo3) |
| `pathfinding` | 4.16.0 | assignation Hungarian (module `kuhn_munkres`) |
| `rayon` | 1.12.0 | parallélisation matrice de coût |
| `thiserror` | 2.0.20 | erreurs |

**Point d'attention identifié pour `assignment.rs`** : le module `kuhn_munkres` de `pathfinding` travaille sur une structure `Matrix` qui n'impose pas d'être carrée au niveau du type, mais l'algorithme Kuhn-Munkres lui-même est conçu pour un problème d'assignation carré. En pratique, une matrice de coût rectangulaire (nombre de détections ≠ nombre de tracks, cas courant à chaque frame) devra être **paddée à carré avec des coûts factices élevés** avant l'appel, puis les entrées correspondant au padding filtrées après résolution. À vérifier précisément sur le code de la fonction au moment d'écrire `assignment.rs`, plutôt que de le découvrir en debug.

`munkres` et `hungarian` (crates alternatives, ~50k téléchargements pour `munkres`) existent mais `pathfinding` est déjà recommandé par la communauté comme plus rapide et mieux maintenu — confirme le choix du §6.

## 5. Environnement local — à corriger avant le Jalon 1

- `rustc 1.98.1` / `cargo 1.98.1` installés et à jour.
- **`maturin` n'est pas installé** (`pip show maturin` → not found). Nécessaire avant de pouvoir builder les bindings PyO3 : `pip install maturin`.
- `Cargo.toml` actuel a `name = "Ragdeepsort"` (squelette RustRover par défaut) — à renommer en `deepsort-rs` (ou nom définitif retenu, cf. avertissement en tête de `PROJECT.md` sur la disponibilité crates.io/PyPI) avant d'ajouter la structure `src/` décrite en §6.

## 6. Ce que ça change dans `PROJECT.md`

Le risque §9 sur `similari` est **résolu** : pas d'action de repositionnement nécessaire, `similari` sera juste cité en référence dans le README final. Rien d'autre dans le plan n'a besoin de changer — les hypothèses du document (licences, defaults de référence, choix de crates) sont confirmées par lecture directe des sources, pas seulement plausibles.
