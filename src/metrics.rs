//! Distances utilisées pour construire les matrices de coût : IoU (boîtes en
//! format ltwh) et cosinus (embeddings d'apparence).

/// Coût élevé assigné aux paires interdites (gating, IoU hors fenêtre) —
/// reste fini pour que l'algorithme d'assignation demeure bien défini, mais
/// largement supérieur à n'importe quel seuil de matching.
pub const INFTY_COST: f64 = 1e5;

/// Intersection-over-union entre une boîte (ltwh) et un ensemble de candidates (ltwh).
pub fn iou(bbox: [f64; 4], candidates: &[[f64; 4]]) -> Vec<f64> {
    let (bx1, by1) = (bbox[0], bbox[1]);
    let (bx2, by2) = (bbox[0] + bbox[2], bbox[1] + bbox[3]);
    let area_bbox = bbox[2] * bbox[3];

    candidates
        .iter()
        .map(|c| {
            let (cx1, cy1) = (c[0], c[1]);
            let (cx2, cy2) = (c[0] + c[2], c[1] + c[3]);

            let ix1 = bx1.max(cx1);
            let iy1 = by1.max(cy1);
            let ix2 = bx2.min(cx2);
            let iy2 = by2.min(cy2);

            let iw = (ix2 - ix1).max(0.0);
            let ih = (iy2 - iy1).max(0.0);
            let intersection = iw * ih;

            let area_candidate = c[2] * c[3];
            let union = area_bbox + area_candidate - intersection;

            if union <= 0.0 {
                0.0
            } else {
                intersection / union
            }
        })
        .collect()
}

/// Normalise un vecteur à la norme 1 (vecteur nul inchangé).
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Distance cosinus (1 - similarité) entre deux vecteurs déjà normalisés
/// (norme 1) : simple produit scalaire. Toute feature d'apparence passe par
/// `normalize()` une seule fois à l'entrée du tracker (`Tracker::update`)
/// plutôt que d'être renormalisée à chaque paire piste/détection comparée —
/// un coût quadratique en O(pistes × détections × budget) sinon, visible au
/// banc du Jalon 4 à haute densité.
pub fn cosine_distance_normalized(a: &[f64], b: &[f64]) -> f64 {
    1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>()
}

/// Plus petite distance cosinus entre `query` et un ensemble d'échantillons
/// observés pour une piste (historique borné par `nn_budget`). `query` et
/// `samples` doivent déjà être normalisés.
pub fn nn_cosine_distance(samples: &[Vec<f64>], query: &[f64]) -> f64 {
    samples
        .iter()
        .map(|s| cosine_distance_normalized(s, query))
        .fold(f64::INFINITY, f64::min)
}
