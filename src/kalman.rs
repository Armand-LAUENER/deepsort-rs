//! Filtre de Kalman pour le suivi de boîtes englobantes (cx, cy, a, h) avec
//! modèle de mouvement à vitesse constante, tel que décrit dans Wojke et al.
//! 2017 (SORT/DeepSORT). État à 8 dimensions : position + vitesse.

use nalgebra::{SMatrix, SVector};

pub type State = SVector<f64, 8>;
pub type Covariance = SMatrix<f64, 8, 8>;
pub type Measurement = SVector<f64, 4>;
pub type ProjectedCovariance = SMatrix<f64, 4, 4>;

/// Seuil du gating chi² à 4 degrés de liberté, quantile 0.95.
pub const CHI2_95_4DOF: f64 = 9.4877;

const STD_WEIGHT_POSITION: f64 = 1.0 / 20.0;
const STD_WEIGHT_VELOCITY: f64 = 1.0 / 160.0;

pub struct KalmanFilter {
    motion_mat: SMatrix<f64, 8, 8>,
    update_mat: SMatrix<f64, 4, 8>,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilter {
    pub fn new() -> Self {
        let mut motion_mat = SMatrix::<f64, 8, 8>::identity();
        for i in 0..4 {
            motion_mat[(i, i + 4)] = 1.0;
        }

        let mut update_mat = SMatrix::<f64, 4, 8>::zeros();
        for i in 0..4 {
            update_mat[(i, i)] = 1.0;
        }

        Self {
            motion_mat,
            update_mat,
        }
    }

    /// Crée une piste à partir d'une détection non associée : vitesse nulle,
    /// incertitude proportionnelle à la hauteur de la boîte.
    pub fn initiate(&self, measurement: &Measurement) -> (State, Covariance) {
        let h = measurement[3];

        let mut mean = State::zeros();
        mean.fixed_rows_mut::<4>(0).copy_from(measurement);

        let std = [
            2.0 * STD_WEIGHT_POSITION * h,
            2.0 * STD_WEIGHT_POSITION * h,
            1e-2,
            2.0 * STD_WEIGHT_POSITION * h,
            10.0 * STD_WEIGHT_VELOCITY * h,
            10.0 * STD_WEIGHT_VELOCITY * h,
            1e-5,
            10.0 * STD_WEIGHT_VELOCITY * h,
        ];
        let covariance = Covariance::from_diagonal(&SVector::<f64, 8>::from_iterator(
            std.iter().map(|s| s * s),
        ));

        (mean, covariance)
    }

    /// Prédit l'état à l'instant suivant (dt = 1 frame).
    pub fn predict(&self, mean: &State, covariance: &Covariance) -> (State, Covariance) {
        let h = mean[3];

        let std = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-2,
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * h,
            1e-5,
            STD_WEIGHT_VELOCITY * h,
        ];
        let motion_cov = Covariance::from_diagonal(&SVector::<f64, 8>::from_iterator(
            std.iter().map(|s| s * s),
        ));

        let mean = self.motion_mat * mean;
        let covariance = self.motion_mat * covariance * self.motion_mat.transpose() + motion_cov;

        (mean, covariance)
    }

    /// Projette l'état dans l'espace des mesures (cx, cy, a, h), en ajoutant
    /// le bruit de mesure.
    pub fn project(&self, mean: &State, covariance: &Covariance) -> (Measurement, ProjectedCovariance) {
        let h = mean[3];

        let std = [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-1,
            STD_WEIGHT_POSITION * h,
        ];
        let innovation_cov = ProjectedCovariance::from_diagonal(&SVector::<f64, 4>::from_iterator(
            std.iter().map(|s| s * s),
        ));

        let projected_mean = self.update_mat * mean;
        let projected_cov =
            self.update_mat * covariance * self.update_mat.transpose() + innovation_cov;

        (projected_mean, projected_cov)
    }

    /// Corrige l'état prédit avec une détection associée.
    pub fn update(
        &self,
        mean: &State,
        covariance: &Covariance,
        measurement: &Measurement,
    ) -> (State, Covariance) {
        let (projected_mean, projected_cov) = self.project(mean, covariance);

        let chol = projected_cov
            .cholesky()
            .expect("la covariance projetée doit être définie positive");

        // K^T résout S * K^T = H * P (S symétrique), soit K = P H^T S^{-1}.
        let hp = self.update_mat * covariance;
        let kalman_gain = chol.solve(&hp).transpose();

        let innovation = measurement - projected_mean;
        let new_mean = mean + kalman_gain * innovation;
        let new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();

        (new_mean, new_covariance)
    }

    /// Distance de Mahalanobis au carré entre l'état projeté et chaque mesure.
    /// `only_position` restreint le calcul à (cx, cy) — utilisé pour le
    /// gating lorsqu'on veut ignorer l'aspect/la hauteur.
    pub fn gating_distance(
        &self,
        mean: &State,
        covariance: &Covariance,
        measurements: &[Measurement],
        only_position: bool,
    ) -> Vec<f64> {
        let (projected_mean, projected_cov) = self.project(mean, covariance);

        if only_position {
            let mean2 = projected_mean.fixed_rows::<2>(0).into_owned();
            let cov2 = projected_cov.fixed_view::<2, 2>(0, 0).into_owned();
            let chol = cov2
                .cholesky()
                .expect("la covariance projetée (position) doit être définie positive");

            measurements
                .iter()
                .map(|m| {
                    let d = m.fixed_rows::<2>(0).into_owned() - mean2;
                    d.dot(&chol.solve(&d))
                })
                .collect()
        } else {
            let chol = projected_cov
                .cholesky()
                .expect("la covariance projetée doit être définie positive");

            measurements
                .iter()
                .map(|m| {
                    let d = m - projected_mean;
                    d.dot(&chol.solve(&d))
                })
                .collect()
        }
    }
}
