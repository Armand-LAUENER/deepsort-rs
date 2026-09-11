//! Orchestration d'une frame : prédiction, association, mise à jour du cycle
//! de vie des pistes et de la banque de features (`nn_budget`).

use std::collections::{HashMap, HashSet};

use crate::cascade::associate;
use crate::kalman::{KalmanFilter, Measurement};
use crate::track::Track;

pub struct TrackerParams {
    pub max_age: u32,
    pub n_init: u32,
    pub max_cosine_distance: f64,
    pub nn_budget: Option<usize>,
    pub max_iou_distance: f64,
}

pub struct TrackOutput {
    pub id: u64,
    /// (x1, y1, x2, y2)
    pub ltrb: [f64; 4],
    pub age: u32,
}

pub struct Tracker {
    kf: KalmanFilter,
    tracks: Vec<Track>,
    /// Historique d'apparence par piste confirmée, borné par `nn_budget` —
    /// séparé de `Track::features` pour reproduire exactement la référence :
    /// une piste tentative accumule localement sans jamais alimenter la
    /// banque tant qu'elle n'est pas confirmée.
    feature_bank: HashMap<u64, Vec<Vec<f64>>>,
    next_id: u64,
    params: TrackerParams,
}

impl Tracker {
    pub fn new(params: TrackerParams) -> Self {
        Self {
            kf: KalmanFilter::new(),
            tracks: Vec::new(),
            feature_bank: HashMap::new(),
            next_id: 1,
            params,
        }
    }

    pub fn update(&mut self, boxes_xyxy: &[[f64; 4]], embeddings: &[Vec<f64>]) -> Vec<TrackOutput> {
        for track in &mut self.tracks {
            track.predict(&self.kf);
        }

        // Normalisées une seule fois ici : la distance cosinus dans la
        // cascade (cascade.rs) suppose des vecteurs à norme 1 et fait un
        // simple produit scalaire, au lieu de renormaliser à chaque paire.
        let embeddings: Vec<Vec<f64>> = embeddings.iter().map(|e| crate::metrics::normalize(e)).collect();
        let embeddings = &embeddings[..];

        let ltwh: Vec<[f64; 4]> = boxes_xyxy
            .iter()
            .map(|b| [b[0], b[1], b[2] - b[0], b[3] - b[1]])
            .collect();
        let xyah: Vec<Measurement> = ltwh
            .iter()
            .map(|b| {
                let aspect = if b[3] > 0.0 { b[2] / b[3] } else { 0.0 };
                Measurement::from_column_slice(&[b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, aspect, b[3]])
            })
            .collect();

        let result = associate(
            &self.kf,
            &self.tracks,
            &xyah,
            &ltwh,
            embeddings,
            &self.feature_bank,
            self.params.max_cosine_distance,
            self.params.max_age,
            self.params.max_iou_distance,
        );

        for (track_idx, det_idx) in result.matches {
            self.tracks[track_idx].update(&self.kf, &xyah[det_idx], embeddings[det_idx].clone());
        }
        for track_idx in result.unmatched_tracks {
            self.tracks[track_idx].mark_missed();
        }
        for det_idx in result.unmatched_detections {
            self.initiate_track(&xyah[det_idx], embeddings[det_idx].clone());
        }

        self.tracks.retain(|t| !t.is_deleted());
        self.flush_features();

        self.tracks
            .iter()
            .map(|t| TrackOutput {
                id: t.id,
                ltrb: to_ltrb(t.to_ltwh()),
                age: t.age,
            })
            .collect()
    }

    fn initiate_track(&mut self, measurement: &Measurement, feature: Vec<f64>) {
        let (mean, covariance) = self.kf.initiate(measurement);
        self.tracks.push(Track::new(
            self.next_id,
            mean,
            covariance,
            feature,
            self.params.n_init,
            self.params.max_age,
        ));
        self.next_id += 1;
    }

    /// Déverse les features accumulées des pistes confirmées dans la banque,
    /// applique le budget `nn_budget`, puis ne garde localement que la
    /// dernière feature — miroir de `Tracker.update`/`partial_fit` dans la
    /// référence.
    fn flush_features(&mut self) {
        for track in &mut self.tracks {
            if !track.is_confirmed() {
                continue;
            }
            let entry = self.feature_bank.entry(track.id).or_default();
            entry.append(&mut track.features);
            if let Some(budget) = self.params.nn_budget
                && entry.len() > budget
            {
                let excess = entry.len() - budget;
                entry.drain(0..excess);
            }
            track.features.push(entry.last().expect("au moins une feature").clone());
        }

        let active_ids: HashSet<u64> = self
            .tracks
            .iter()
            .filter(|t| t.is_confirmed())
            .map(|t| t.id)
            .collect();
        self.feature_bank.retain(|id, _| active_ids.contains(id));
    }
}

fn to_ltrb(ltwh: [f64; 4]) -> [f64; 4] {
    [ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]]
}
