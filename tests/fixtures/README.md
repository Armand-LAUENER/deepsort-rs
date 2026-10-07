# Fixtures de parité MOT17

Entrées réelles rejouées par `tests/test_parity_mot17.py` : les deux trackers
reçoivent exactement les mêmes détections et embeddings, et doivent produire
les mêmes pistes.

| Fichier | Séquence | Frames | Détections |
|---|---|---:|---:|
| `mot17-04-sdp.npz` | MOT17-04-SDP (train) | 1–300 | 10 155 |
| `mot17-09-sdp.npz` | MOT17-09-SDP (train) | 1–525 (toute la séquence) | 3 607 |

## Contenu

- `frame` (int32, N) : numéro de frame (base 1) de chaque détection.
- `ltwh` (float32, N×4) : détections publiques SDP de MOT17 (`det/det.txt`),
  ramenées en pixels base 0, boîtes de taille nulle écartées.
- `conf` (float32, N) : confiance de la détection.
- `embedding` (float16, N×128) : embeddings d'apparence MobileNetV2 (1280-d)
  calculés par l'embedder de VisionCam sur les crops que prendrait
  `deep_sort_realtime`, puis projetés sur leurs 128 premières directions
  principales (SVD non centrée) pour limiter la taille du dépôt.
- `n_frames`, `source` : longueur et nom de la séquence.

Régénération : `scripts/record_fixture.py` (docstring pour la commande).
La parité en dimension complète (1280-d) est vérifiée côté VisionCam
(`tools.eval_mot`).

## Licence

Ces fichiers sont dérivés du jeu de données MOT17 (MOTChallenge), distribué
sous licence **Creative Commons BY-NC-SA 3.0** : ils restent sous cette
licence, et non sous la licence MIT du reste du dépôt. Usage non commercial,
partage dans les mêmes conditions, attribution :

> A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler.
> *MOT16: A Benchmark for Multi-Object Tracking.* arXiv:1603.00831, 2016.
