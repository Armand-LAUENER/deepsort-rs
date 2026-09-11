# deepsort-rs

Tracker DeepSORT (Kalman + association apparence/mouvement + matching cascade) réimplémenté en Rust, exposé en Python via PyO3. Voir `PROJECT.md` pour le contexte, l'hypothèse testée et les critères de succès complets.

> Nom de travail — à confirmer avant publication (voir `PROJECT.md` §1).

## Statut

- **Jalon 1** (Kalman) et **Jalon 2** (tracker complet) faits, parité validée contre `deep_sort_realtime` réellement installé sur séquences synthétiques (`tests/test_kalman.py`, `tests/test_parity.py`) — voir les limites ci-dessous.
- **Jalon 4** (bench vitesse) fait pour la partie tracker seul, ci-dessous.
- Séquence VisionCam + MOT17 (Jalon 0), intégration VisionCam (Jalon 3) et MOTA/IDF1 : **pas encore faits**, hors de portée de cet environnement de développement (pas d'accès à VisionCam ni au jeu de données MOT17).

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

**Machine** : Intel Core i7-1360P, Windows, Python 3.11.9, rustc 1.98.1, build `--release`.

| Densité | Candidat | Médiane (ms) | p95 (ms) | Speedup vs `deepsort_rs` |
|---|---|---:|---:|---:|
| 10 | **deepsort_rs** | 0.463 | 0.701 | 1.0× |
| 10 | deep_sort_realtime | 9.551 | 12.157 | 20.6× plus lent |
| 10 | norfair (IoU seul) | 2.721 | 4.050 | 5.9× plus lent |
| 50 | **deepsort_rs** | 7.990 | 11.630 | 1.0× |
| 50 | deep_sort_realtime | 50.887 | 63.531 | 6.4× plus lent |
| 50 | norfair (IoU seul) | 14.103 | 18.034 | 1.8× plus lent |
| 200 | **deepsort_rs** | 83.892 | 141.349 | 1.0× |
| 200 | deep_sort_realtime | 489.132 | 2952.057 | 5.8× plus lent |
| 200 | norfair (IoU seul) | 86.098 | 121.011 | ~1.0× (équivalent) |

**Lecture honnête** : le gain contre `deep_sort_realtime` (la vraie cible de parité) va de 20,6× à 10 objets à 5,8× à 200 objets — l'avantage se réduit avec la densité parce que l'assignation Hungarian est O(n³) des deux côtés, un plafond algorithmique partagé, pas un problème d'implémentation. À 200 objets, `deepsort_rs` rejoint `norfair` (qui ne fait pourtant pas de matching par apparence) : l'écart au bas de la fourchette « 10-50× » de l'hypothèse §3 de `PROJECT.md` n'est donc pas atteint à haute densité — à documenter tel quel plutôt qu'à enjoliver. Le p95 de `deep_sort_realtime` à 200 objets (2952 ms, très supérieur à sa médiane) suggère une forte variance côté référence à cette densité, non creusée ici.

Un bug de méthodologie a été trouvé et corrigé en cours de route : la distance cosinus renormalisait chaque vecteur à chaque paire piste/détection comparée au lieu d'une normalisation unique en entrée de frame — coût quadratique inutile qui faisait apparaître `deepsort_rs` plus lent que `deep_sort_realtime` à 200 objets avant correction (`src/metrics.rs`, `normalize()` appelée une fois dans `Tracker::update`).

**Non fait** : MOTA/IDF1 (nécessite MOT17, non disponible ici) et le gain end-to-end sur le pipeline VisionCam complet (nécessite l'intégration du Jalon 3, également non faite). Le chiffre ci-dessus est un speedup **tracker seul**, pas un gain pipeline — `PROJECT.md` §8 est explicite sur le fait que le second sera bien plus faible (loi d'Amdahl).

Reproduire (nécessite un venv séparé car `norfair` impose `numpy<2.0`, ce qui casserait le venv de dev principal) :

```bash
python -m venv .venv-bench
.venv-bench/Scripts/pip install maturin deep_sort_realtime norfair
.venv-bench/Scripts/python -m maturin develop --release
.venv-bench/Scripts/python scripts/bench.py
```
