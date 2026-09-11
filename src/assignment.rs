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

    // La référence construit ses listes de non-appariés en deux temps : d'abord
    // ceux que l'assignation n'a jamais retenus, puis ceux qu'elle a retenus
    // mais dont le coût dépasse le seuil. Cet ordre-là est observable : le
    // tracker crée les nouvelles pistes dans l'ordre des détections non
    // appariées, donc un ordre différent attribue les IDs différemment.
    let mut assigned_row = vec![false; n_rows];
    let mut assigned_col = vec![false; n_cols];
    let mut matches = Vec::new();
    let mut rejected_tracks = Vec::new();
    let mut rejected_detections = Vec::new();

    for (row, &col) in assignment.iter().enumerate() {
        // Colonne factice ajoutée par le padding : la ligne n'a pas été
        // assignée du tout, comme une ligne en trop chez scipy.
        if col >= n_cols {
            continue;
        }
        assigned_row[row] = true;
        assigned_col[col] = true;
        if cost_matrix[row][col] > max_distance {
            rejected_tracks.push(track_indices[row]);
            rejected_detections.push(detection_indices[col]);
        } else {
            matches.push((track_indices[row], detection_indices[col]));
        }
    }

    let mut unmatched_tracks: Vec<usize> = (0..n_rows)
        .filter(|&r| !assigned_row[r])
        .map(|r| track_indices[r])
        .collect();
    unmatched_tracks.extend(rejected_tracks);

    let mut unmatched_detections: Vec<usize> = (0..n_cols)
        .filter(|&c| !assigned_col[c])
        .map(|c| detection_indices[c])
        .collect();
    unmatched_detections.extend(rejected_detections);

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

#[cfg(test)]
mod tests {
    use super::*;

    /// La référence renvoie les détections jamais retenues par l'assignation
    /// AVANT celles qu'elle a retenues puis rejetées sur le seuil de coût.
    /// Le tracker crée les nouvelles pistes dans cet ordre : l'inverser
    /// suffit à intervertir deux track_id sur une séquence réelle, sans que
    /// rien d'autre ne bouge.
    #[test]
    fn unmatched_detections_put_never_assigned_before_cost_rejected() {
        // Une piste, deux détections : l'assignation retient la colonne 0
        // (moins chère) mais son coût dépasse le seuil ; la colonne 1 n'est
        // jamais retenue, faute d'une seconde ligne.
        let cost_matrix = vec![vec![0.90, 0.95]];
        let result = min_cost_matching(&cost_matrix, 0.5, &[7], &[10, 11]);

        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_detections, vec![11, 10]);
        assert_eq!(result.unmatched_tracks, vec![7]);
    }

    /// Symétrique côté pistes, avec la matrice transposée.
    #[test]
    fn unmatched_tracks_put_never_assigned_before_cost_rejected() {
        let cost_matrix = vec![vec![0.90], vec![0.95]];
        let result = min_cost_matching(&cost_matrix, 0.5, &[7, 8], &[10]);

        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_tracks, vec![8, 7]);
        assert_eq!(result.unmatched_detections, vec![10]);
    }

    /// Le cas nominal ne doit pas bouger : sous le seuil, la paire est appariée.
    #[test]
    fn pairs_under_the_threshold_are_matched() {
        let cost_matrix = vec![vec![0.1, 0.8], vec![0.8, 0.2]];
        let result = min_cost_matching(&cost_matrix, 0.5, &[3, 4], &[20, 21]);

        assert_eq!(result.matches, vec![(3, 20), (4, 21)]);
        assert!(result.unmatched_tracks.is_empty());
        assert!(result.unmatched_detections.is_empty());
    }
}
