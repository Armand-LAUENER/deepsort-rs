# Changelog

## Jalon 1 — Kalman

- Squelette maturin (`Cargo.toml`, `pyproject.toml`, `python/deepsort_rs/`) qui compile et s'installe en editable dans un venv.
- `KalmanFilter` (état 8D, mesure 4D, modèle à vitesse constante) implémenté en Rust (`src/kalman.rs`) depuis les équations du papier Wojke et al. 2017, exposé à Python via PyO3 (`src/lib.rs`).
- `tests/test_kalman.py` : parité contre une référence NumPy indépendante (`tests/reference_kalman.py`) sur trajectoires synthétiques — `initiate`, `predict`, `project`, `update`, `gating_distance` — écart < 1e-5 (5/5 tests passent).
- `docs/veille-similari.md` : vérification préalable de `similari` (pas de matching cascade par âge ni de gating chi² dans son code) et versions de crates figées via `cargo add --dry-run`.
