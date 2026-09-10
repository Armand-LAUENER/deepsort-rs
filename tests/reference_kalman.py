"""Implémentation NumPy indépendante du filtre de Kalman DeepSORT (Wojke et al.
2017), écrite depuis les équations du papier — sert de référence pour
`test_kalman.py`, pas une traduction du code Rust.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

STD_WEIGHT_POSITION = 1.0 / 20
STD_WEIGHT_VELOCITY = 1.0 / 160
CHI2_95_4DOF = 9.4877


class ReferenceKalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2 * STD_WEIGHT_POSITION * h,
            2 * STD_WEIGHT_POSITION * h,
            1e-2,
            2 * STD_WEIGHT_POSITION * h,
            10 * STD_WEIGHT_VELOCITY * h,
            10 * STD_WEIGHT_VELOCITY * h,
            1e-5,
            10 * STD_WEIGHT_VELOCITY * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        h = mean[3]
        std_pos = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-2,
            STD_WEIGHT_POSITION * h,
        ]
        std_vel = [
            STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * h,
            1e-5,
            STD_WEIGHT_VELOCITY * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        h = mean[3]
        std = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-1,
            STD_WEIGHT_POSITION * h,
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor = cho_factor(projected_cov, lower=True)
        kalman_gain = cho_solve(chol_factor, (covariance @ self._update_mat.T).T).T

        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        chol_factor = cho_factor(covariance, lower=True)
        d = measurements - mean
        z = cho_solve(chol_factor, d.T)
        return np.sum(d.T * z, axis=0)
