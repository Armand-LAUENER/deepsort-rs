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

/// Lignes de features contiguës en `f32` : la ligne `i` occupe
/// `data[i * dim..(i + 1) * dim]`. Sert à la fois pour l'historique
/// d'apparence d'une piste et pour la matrice des embeddings d'une frame —
/// une seule allocation, et un format directement consommable par `sgemm`.
#[derive(Clone, Debug)]
pub struct FeatureMatrix {
    dim: usize,
    rows: usize,
    data: Vec<f32>,
}

impl FeatureMatrix {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            rows: 0,
            data: Vec::new(),
        }
    }

    /// `data` doit contenir `rows * dim` valeurs, ligne par ligne.
    pub fn from_rows(dim: usize, rows: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * dim, "data must hold rows * dim values");
        Self { dim, rows, data }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn last(&self) -> &[f32] {
        self.row(self.rows - 1)
    }

    pub fn push(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.dim, "feature dimension mismatch");
        self.data.extend_from_slice(v);
        self.rows += 1;
    }

    /// Ajoute les lignes de `other` à la suite et vide `other`.
    pub fn append(&mut self, other: &mut FeatureMatrix) {
        assert_eq!(other.dim, self.dim, "feature dimension mismatch");
        self.data.append(&mut other.data);
        self.rows += other.rows;
        other.rows = 0;
    }

    /// Ne garde que les `budget` dernières lignes (éviction des plus
    /// anciennes, comme `samples[-budget:]` dans la référence).
    pub fn keep_last(&mut self, budget: usize) {
        if self.rows > budget {
            self.data.drain(0..(self.rows - budget) * self.dim);
            self.rows = budget;
        }
    }

    /// Copie normalisée ligne par ligne (norme 1 ; une ligne nulle reste
    /// nulle). La norme s'accumule en `f64` : gratuit, et plus précis.
    pub fn normalized(&self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());
        for i in 0..self.rows {
            let row = self.row(i);
            let norm = row
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt();
            if norm == 0.0 {
                data.extend_from_slice(row);
            } else {
                data.extend(row.iter().map(|&x| (x as f64 / norm) as f32));
            }
        }
        Self::from_rows(self.dim, self.rows, data)
    }

    /// Sous-matrice formée des lignes `indices`, dans cet ordre.
    pub fn select_rows(&self, indices: &[usize]) -> Self {
        let mut data = Vec::with_capacity(indices.len() * self.dim);
        for &i in indices {
            data.extend_from_slice(self.row(i));
        }
        Self::from_rows(self.dim, indices.len(), data)
    }
}

/// Plus petite distance cosinus entre chaque requête `queries[j]` et les
/// échantillons d'une piste : `cost[j] = min_i (1 - <samples_i, queries_j>)`.
/// Lignes supposées déjà normalisées (cf. `FeatureMatrix::normalized`).
///
/// Les similarités sont calculées d'un bloc, `C = S · Qᵀ`, par `sgemm`
/// (vectorisé, en tuiles), comme le produit matriciel numpy de la référence
/// (`nn_matching._cosine_distance`) ; le minimum de la distance est le
/// maximum de la similarité. `scratch` est réutilisé d'un appel à l'autre.
pub fn nn_cosine_costs(
    samples: &FeatureMatrix,
    queries: &FeatureMatrix,
    scratch: &mut Vec<f32>,
) -> Vec<f64> {
    assert_eq!(samples.dim, queries.dim, "feature dimension mismatch");
    let (n, d, m) = (samples.rows, samples.dim, queries.rows);
    if n == 0 {
        return vec![INFTY_COST; m];
    }

    scratch.clear();
    scratch.resize(n * m, 0.0);
    // SAFETY : `samples.data` contient n×d valeurs et `queries.data` m×d
    // (invariant de `FeatureMatrix`), `scratch` n×m. Qᵀ (d×m) est lu sans
    // copie via les strides : ligne-stride 1, colonne-stride d.
    unsafe {
        matrixmultiply::sgemm(
            n,
            d,
            m,
            1.0,
            samples.data.as_ptr(),
            d as isize,
            1,
            queries.data.as_ptr(),
            1,
            d as isize,
            0.0,
            scratch.as_mut_ptr(),
            m as isize,
            1,
        );
    }

    let mut best = vec![f32::NEG_INFINITY; m];
    for row in scratch.chunks_exact(m) {
        for (b, &c) in best.iter_mut().zip(row) {
            *b = b.max(c);
        }
    }
    best.into_iter().map(|b| 1.0 - b as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Version naïve (un produit scalaire par paire), en `f64`.
    fn naive_costs(samples: &FeatureMatrix, queries: &FeatureMatrix) -> Vec<f64> {
        (0..queries.len())
            .map(|j| {
                (0..samples.len())
                    .map(|i| {
                        let dot: f64 = samples
                            .row(i)
                            .iter()
                            .zip(queries.row(j))
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .sum();
                        1.0 - dot
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect()
    }

    fn pseudo_random(dim: usize, rows: usize, seed: u32) -> FeatureMatrix {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let data = (0..dim * rows)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        FeatureMatrix::from_rows(dim, rows, data).normalized()
    }

    #[test]
    fn sgemm_costs_match_naive_dot_products() {
        let samples = pseudo_random(512, 37, 1);
        let queries = pseudo_random(512, 11, 2);
        let mut scratch = Vec::new();
        let fast = nn_cosine_costs(&samples, &queries, &mut scratch);
        let slow = naive_costs(&samples, &queries);
        for (a, b) in fast.iter().zip(&slow) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn identical_query_has_zero_cost() {
        let samples = pseudo_random(16, 5, 3);
        let queries = samples.select_rows(&[2]);
        let costs = nn_cosine_costs(&samples, &queries, &mut Vec::new());
        assert!(costs[0].abs() < 1e-6);
    }

    #[test]
    fn keep_last_evicts_oldest_rows() {
        let mut m = FeatureMatrix::new(2);
        for i in 0..5 {
            m.push(&[i as f32, -(i as f32)]);
        }
        m.keep_last(3);
        assert_eq!(m.len(), 3);
        assert_eq!(m.row(0), &[2.0, -2.0]);
        assert_eq!(m.last(), &[4.0, -4.0]);
    }

    #[test]
    fn zero_row_stays_zero_after_normalization() {
        let m = FeatureMatrix::from_rows(3, 2, vec![0.0, 0.0, 0.0, 3.0, 0.0, 4.0]).normalized();
        assert_eq!(m.row(0), &[0.0, 0.0, 0.0]);
        assert_eq!(m.row(1), &[0.6, 0.0, 0.8]);
    }
}
