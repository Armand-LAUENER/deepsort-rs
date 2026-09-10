//! Matching cascade par âge de piste + second passage IoU pour les
//! non-appariés, tel que décrit dans Wojke et al. 2017 (`tracker.py::_match`).

use std::collections::{HashMap, HashSet};

use crate::assignment::{gate_cost_matrix, min_cost_matching};
use crate::kalman::{KalmanFilter, Measurement};
use crate::metrics::{iou, nn_cosine_distance, INFTY_COST};
use crate::track::Track;

pub struct MatchOutput {
    pub matches: Vec<(usize, usize)>,
    pub unmatched_tracks: Vec<usize>,
    pub unmatched_detections: Vec<usize>,
}

/// Matching cascade : à chaque niveau (âge croissant depuis la dernière mise
/// à jour), on associe par coût cosinus gaté les pistes de cet âge exact aux
/// détections encore libres, en donnant la priorité aux pistes vues le plus
/// récemment.
#[allow(clippy::too_many_arguments)]
fn matching_cascade(
    kf: &KalmanFilter,
    tracks: &[Track],
    detections_xyah: &[Measurement],
    embeddings: &[Vec<f64>],
    feature_bank: &HashMap<u64, Vec<Vec<f64>>>,
    max_cosine_distance: f64,
    cascade_depth: u32,
    track_indices: &[usize],
    detection_indices: &[usize],
) -> MatchOutput {
    let mut unmatched_detections = detection_indices.to_vec();
    let mut matches = Vec::new();

    for level in 0..cascade_depth {
        if unmatched_detections.is_empty() {
            break;
        }

        let track_indices_l: Vec<usize> = track_indices
            .iter()
            .copied()
            .filter(|&k| tracks[k].time_since_update == 1 + level)
            .collect();
        if track_indices_l.is_empty() {
            continue;
        }

        let mut cost_matrix: Vec<Vec<f64>> = track_indices_l
            .iter()
            .map(|&t| match feature_bank.get(&tracks[t].id) {
                Some(samples) => unmatched_detections
                    .iter()
                    .map(|&d| nn_cosine_distance(samples, &embeddings[d]))
                    .collect(),
                None => vec![INFTY_COST; unmatched_detections.len()],
            })
            .collect();

        let track_states: Vec<_> = track_indices_l
            .iter()
            .map(|&t| (&tracks[t].mean, &tracks[t].covariance))
            .collect();
        let measurements: Vec<Measurement> = unmatched_detections
            .iter()
            .map(|&d| detections_xyah[d])
            .collect();
        gate_cost_matrix(kf, &mut cost_matrix, &track_states, &measurements);

        let result = min_cost_matching(
            &cost_matrix,
            max_cosine_distance,
            &track_indices_l,
            &unmatched_detections,
        );
        matches.extend(result.matches);
        unmatched_detections = result.unmatched_detections;
    }

    let matched: HashSet<usize> = matches.iter().map(|&(t, _)| t).collect();
    let unmatched_tracks = track_indices
        .iter()
        .copied()
        .filter(|k| !matched.contains(k))
        .collect();

    MatchOutput {
        matches,
        unmatched_tracks,
        unmatched_detections,
    }
}

/// Association complète d'une frame : cascade par apparence sur les pistes
/// confirmées, puis passage IoU pour les pistes non confirmées et celles
/// manquées une seule fois ce tour-ci (elles restent de bonnes candidates
/// positionnelles).
#[allow(clippy::too_many_arguments)]
pub fn associate(
    kf: &KalmanFilter,
    tracks: &[Track],
    detections_xyah: &[Measurement],
    detections_ltwh: &[[f64; 4]],
    embeddings: &[Vec<f64>],
    feature_bank: &HashMap<u64, Vec<Vec<f64>>>,
    max_cosine_distance: f64,
    max_age: u32,
    max_iou_distance: f64,
) -> MatchOutput {
    let confirmed: Vec<usize> = tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.is_confirmed())
        .map(|(i, _)| i)
        .collect();
    let unconfirmed: Vec<usize> = tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.is_confirmed())
        .map(|(i, _)| i)
        .collect();
    let all_detections: Vec<usize> = (0..detections_xyah.len()).collect();

    let cascade_result = matching_cascade(
        kf,
        tracks,
        detections_xyah,
        embeddings,
        feature_bank,
        max_cosine_distance,
        max_age,
        &confirmed,
        &all_detections,
    );

    let mut iou_candidates = unconfirmed;
    let mut unmatched_tracks_a = Vec::new();
    for k in cascade_result.unmatched_tracks {
        if tracks[k].time_since_update == 1 {
            iou_candidates.push(k);
        } else {
            unmatched_tracks_a.push(k);
        }
    }

    let candidates_ltwh: Vec<[f64; 4]> = cascade_result
        .unmatched_detections
        .iter()
        .map(|&d| detections_ltwh[d])
        .collect();
    let iou_cost_matrix: Vec<Vec<f64>> = iou_candidates
        .iter()
        .map(|&t| {
            if tracks[t].time_since_update > 1 {
                vec![INFTY_COST; candidates_ltwh.len()]
            } else {
                iou(tracks[t].to_ltwh(), &candidates_ltwh)
                    .into_iter()
                    .map(|v| 1.0 - v)
                    .collect()
            }
        })
        .collect();

    let iou_result = min_cost_matching(
        &iou_cost_matrix,
        max_iou_distance,
        &iou_candidates,
        &cascade_result.unmatched_detections,
    );

    let mut matches = cascade_result.matches;
    matches.extend(iou_result.matches);

    let mut unmatched_tracks: HashSet<usize> = unmatched_tracks_a.into_iter().collect();
    unmatched_tracks.extend(iou_result.unmatched_tracks);

    MatchOutput {
        matches,
        unmatched_tracks: unmatched_tracks.into_iter().collect(),
        unmatched_detections: iou_result.unmatched_detections,
    }
}
