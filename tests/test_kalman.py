"""Jalon 1 : le KalmanFilter Rust doit reproduire la référence NumPy à
moins de 1e-5 sur des trajectoires synthétiques (cf. PROJECT.md §7)."""
import numpy as np
import pytest

from deepsort_rs import KalmanFilter
from reference_kalman import ReferenceKalmanFilter

TOLERANCE = 1e-5


def synthetic_trajectory(rng, n_frames, start, velocity, noise_std=0.5):
    """Boîte (cx, cy, a, h) en mouvement à vitesse constante bruitée."""
    measurements = []
    state = np.array(start, dtype=np.float64)
    vel = np.array(velocity, dtype=np.float64)
    for _ in range(n_frames):
        state = state + vel
        noisy = state + rng.normal(scale=noise_std, size=4)
        noisy[2] = np.clip(noisy[2], 0.1, 5.0)  # aspect ratio positif
        noisy[3] = max(noisy[3], 1.0)  # hauteur positive
        measurements.append(noisy)
    return np.array(measurements)


TRAJECTORIES = [
    dict(start=[100.0, 200.0, 0.5, 80.0], velocity=[1.5, -0.5, 0.0, 0.2], seed=0),
    dict(start=[50.0, 50.0, 1.0, 40.0], velocity=[-2.0, 3.0, 0.01, -0.1], seed=1),
    dict(start=[300.0, 150.0, 0.4, 120.0], velocity=[0.0, 0.0, 0.0, 0.0], seed=2),
]


@pytest.mark.parametrize("traj", TRAJECTORIES)
def test_predict_update_matches_reference(traj):
    rng = np.random.default_rng(traj["seed"])
    measurements = synthetic_trajectory(rng, 60, traj["start"], traj["velocity"])

    rust_kf = KalmanFilter()
    ref_kf = ReferenceKalmanFilter()

    rust_mean, rust_cov = rust_kf.initiate(measurements[0])
    ref_mean, ref_cov = ref_kf.initiate(measurements[0])

    np.testing.assert_allclose(rust_mean, ref_mean, atol=TOLERANCE)
    np.testing.assert_allclose(rust_cov, ref_cov, atol=TOLERANCE)

    for measurement in measurements[1:]:
        rust_mean, rust_cov = rust_kf.predict(rust_mean, rust_cov)
        ref_mean, ref_cov = ref_kf.predict(ref_mean, ref_cov)
        np.testing.assert_allclose(rust_mean, ref_mean, atol=TOLERANCE)
        np.testing.assert_allclose(rust_cov, ref_cov, atol=TOLERANCE)

        rust_mean, rust_cov = rust_kf.update(rust_mean, rust_cov, measurement)
        ref_mean, ref_cov = ref_kf.update(ref_mean, ref_cov, measurement)
        np.testing.assert_allclose(rust_mean, ref_mean, atol=TOLERANCE)
        np.testing.assert_allclose(rust_cov, ref_cov, atol=TOLERANCE)


def test_project_matches_reference():
    rng = np.random.default_rng(42)
    measurements = synthetic_trajectory(rng, 10, [10.0, 20.0, 0.6, 60.0], [1.0, 1.0, 0.0, 0.0])

    rust_kf = KalmanFilter()
    ref_kf = ReferenceKalmanFilter()

    mean, cov = rust_kf.initiate(measurements[0])
    ref_mean, ref_cov = ref_kf.initiate(measurements[0])

    for measurement in measurements[1:]:
        mean, cov = rust_kf.predict(mean, cov)
        ref_mean, ref_cov = ref_kf.predict(ref_mean, ref_cov)

        proj_mean, proj_cov = rust_kf.project(mean, cov)
        ref_proj_mean, ref_proj_cov = ref_kf.project(ref_mean, ref_cov)
        np.testing.assert_allclose(proj_mean, ref_proj_mean, atol=TOLERANCE)
        np.testing.assert_allclose(proj_cov, ref_proj_cov, atol=TOLERANCE)

        mean, cov = rust_kf.update(mean, cov, measurement)
        ref_mean, ref_cov = ref_kf.update(ref_mean, ref_cov, measurement)


def test_gating_distance_matches_reference():
    rng = np.random.default_rng(7)
    measurements = synthetic_trajectory(rng, 20, [0.0, 0.0, 0.5, 50.0], [2.0, -1.0, 0.0, 0.0])

    rust_kf = KalmanFilter()
    ref_kf = ReferenceKalmanFilter()

    mean, cov = rust_kf.initiate(measurements[0])
    ref_mean, ref_cov = ref_kf.initiate(measurements[0])

    for measurement in measurements[1:10]:
        mean, cov = rust_kf.predict(mean, cov)
        ref_mean, ref_cov = ref_kf.predict(ref_mean, ref_cov)
        mean, cov = rust_kf.update(mean, cov, measurement)
        ref_mean, ref_cov = ref_kf.update(ref_mean, ref_cov, measurement)

    candidates = measurements[10:]

    for only_position in (False, True):
        rust_dist = rust_kf.gating_distance(mean, cov, candidates, only_position)
        ref_dist = ref_kf.gating_distance(ref_mean, ref_cov, candidates, only_position)
        np.testing.assert_allclose(rust_dist, ref_dist, atol=TOLERANCE)

    # Sanité : la mesure suivante de la même trajectoire doit passer le
    # gating chi² à 4 ddl (0.95), une mesure aberrante ne doit pas passer.
    from reference_kalman import CHI2_95_4DOF

    on_track = rust_kf.gating_distance(mean, cov, candidates[:1], False)[0]
    assert on_track < CHI2_95_4DOF

    outlier = candidates[0].copy()
    outlier[:2] += 500.0
    off_track = rust_kf.gating_distance(mean, cov, outlier[None, :], False)[0]
    assert off_track > CHI2_95_4DOF
