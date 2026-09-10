//! Cycle de vie d'une piste : tentative → confirmé → supprimé, tel que décrit
//! dans Wojke et al. 2017 (`track.py` de la référence).

use crate::kalman::{Covariance, KalmanFilter, Measurement, State};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Deleted,
}

pub struct Track {
    pub id: u64,
    pub mean: State,
    pub covariance: Covariance,
    pub state: TrackState,
    pub hits: u32,
    pub age: u32,
    pub time_since_update: u32,
    /// Features accumulées depuis le dernier passage dans la banque du
    /// tracker (cf. `Tracker::update`) : non vidées tant que la piste est
    /// tentative, pour reproduire exactement l'ordre d'éviction du budget
    /// de la référence.
    pub features: Vec<Vec<f64>>,
    n_init: u32,
    max_age: u32,
}

impl Track {
    pub fn new(
        id: u64,
        mean: State,
        covariance: Covariance,
        feature: Vec<f64>,
        n_init: u32,
        max_age: u32,
    ) -> Self {
        Self {
            id,
            mean,
            covariance,
            state: TrackState::Tentative,
            hits: 1,
            age: 1,
            time_since_update: 0,
            features: vec![feature],
            n_init,
            max_age,
        }
    }

    pub fn predict(&mut self, kf: &KalmanFilter) {
        let (mean, covariance) = kf.predict(&self.mean, &self.covariance);
        self.mean = mean;
        self.covariance = covariance;
        self.age += 1;
        self.time_since_update += 1;
    }

    pub fn update(&mut self, kf: &KalmanFilter, measurement: &Measurement, feature: Vec<f64>) {
        let (mean, covariance) = kf.update(&self.mean, &self.covariance, measurement);
        self.mean = mean;
        self.covariance = covariance;
        self.features.push(feature);
        self.hits += 1;
        self.time_since_update = 0;
        if self.state == TrackState::Tentative && self.hits >= self.n_init {
            self.state = TrackState::Confirmed;
        }
    }

    pub fn mark_missed(&mut self) {
        if self.state == TrackState::Tentative || self.time_since_update > self.max_age {
            self.state = TrackState::Deleted;
        }
    }

    pub fn is_confirmed(&self) -> bool {
        self.state == TrackState::Confirmed
    }

    pub fn is_deleted(&self) -> bool {
        self.state == TrackState::Deleted
    }

    /// Boîte courante en (left, top, width, height), dérivée de l'état filtré.
    pub fn to_ltwh(&self) -> [f64; 4] {
        let (cx, cy, a, h) = (self.mean[0], self.mean[1], self.mean[2], self.mean[3]);
        let w = a * h;
        [cx - w / 2.0, cy - h / 2.0, w, h]
    }
}
