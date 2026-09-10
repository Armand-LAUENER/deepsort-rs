//! Assignation optimale (Hungarian / Kuhn-Munkres) et gating par distance de
//! Mahalanobis, tels que décrits dans Wojke et al. 2017 (`linear_assignment.py`
//! de la référence : résolution globale, puis rejet des paires au-delà du
//! seuil de matching).

use crate::kalman::{Covariance, KalmanFilter, Measurement, State, CHI2_95_4DOF};
use crate::metrics::INFTY_COST;
use ordered_float::OrderedFloat;
use pathfinding::kuhn_munkres::kuhn_munkres_min;
use pathfinding::matrix::Matrix;

pub struct AssignmentResult {
    pub matches: Vec<(usize, usize)>,
    pub unmatched_tracks: Vec<usize>,
    pub unmatched_detections: Vec<usize>,
}

/// Résout l'assignation de coût minimal entre `track_indices` et
/// `detection_indices` à partir de `cost_matrix` (dimensions
/// `track_indices.len() x detection_indices.len()`), puis rejette les paires
/// dont le coût dépasse `max_distance`.
///
/// `kuhn_munkres_min` exige rows <= cols : quand il y a plus de pistes que de
/// détections, on complète la matrice avec des colonnes factices à coût
/// prohibitif, filtrées après résolution.
pub fn min_cost_matching(
    cost_matrix: &[Vec<f64>],
    max_distance: f64,
    track_indices: &[usize],
    detection_indices: &[usize],
) -> AssignmentResult {
    if track_indices.is_empty() || detection_indices.is_empty() {
        return AssignmentResult {
            matches: Vec::new(),
            unmatched_tracks: track_indices.to_vec(),
            unmatched_detections: detection_indices.to_vec(),
        };
    }

    let n_rows = track_indices.len();
    let n_cols = detection_indices.len();
    let padded_cols = n_rows.max(n_cols);

    let clamped_cost = |r: usize, c: usize| -> f64 {
        if c >= n_cols {
            INFTY_COST
        } else {
            let v = cost_matrix[r][c];
            if v > max_distance {
                max_distance + 1e-5
            } else {
                v
            }
        }
    };

    let matrix: Matrix<OrderedFloat<f64>> =
        Matrix::from_fn(n_rows, padded_cols, |(r, c)| OrderedFloat(clamped_cost(r, c)));
    let (_, assignment) = kuhn_munkres_min(&matrix);

    let mut matched_row = vec![false; n_rows];
    let mut matched_col = vec![false; n_cols];
    let mut matches = Vec::new();

    for (row, &col) in assignment.iter().enumerate() {
        if col >= n_cols || cost_matrix[row][col] > max_distance {
            continue;
        }
        matched_row[row] = true;
        matched_col[col] = true;
        matches.push((track_indices[row], detection_indices[col]));
    }

    let unmatched_tracks = (0..n_rows)
        .filter(|&r| !matched_row[r])
        .map(|r| track_indices[r])
        .collect();
    let unmatched_detections = (0..n_cols)
        .filter(|&c| !matched_col[c])
        .map(|c| detection_indices[c])
        .collect();

    AssignmentResult {
        matches,
        unmatched_tracks,
        unmatched_detections,
    }
}

/// Invalide (coût prohibitif) les paires piste/détection dont la distance de
/// Mahalanobis dépasse le seuil chi² à 0.95 — appliqué en place sur la
/// matrice de coût d'apparence avant assignation.
pub fn gate_cost_matrix(
    kf: &KalmanFilter,
    cost_matrix: &mut [Vec<f64>],
    track_states: &[(&State, &Covariance)],
    measurements: &[Measurement],
) {
    for (row, (mean, covariance)) in track_states.iter().enumerate() {
        let distances = kf.gating_distance(mean, covariance, measurements, false);
        for (col, distance) in distances.into_iter().enumerate() {
            if distance > CHI2_95_4DOF {
                cost_matrix[row][col] = INFTY_COST;
            }
        }
    }
}
