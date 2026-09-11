mod assignment;
mod cascade;
mod kalman;
mod metrics;
mod track;
mod tracker;

use kalman::{Covariance, KalmanFilter as RustKalmanFilter, Measurement, ProjectedCovariance, State};
use nalgebra::SVector;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tracker::{Tracker as RustTracker, TrackerParams};

fn vector4_from_numpy(arr: PyReadonlyArray1<'_, f64>) -> PyResult<Measurement> {
    let view = arr.as_array();
    if view.len() != 4 {
        return Err(PyValueError::new_err("measurement must have length 4"));
    }
    Ok(Measurement::from_iterator(view.iter().copied()))
}

fn vector8_from_numpy(arr: PyReadonlyArray1<'_, f64>) -> PyResult<State> {
    let view = arr.as_array();
    if view.len() != 8 {
        return Err(PyValueError::new_err("mean must have length 8"));
    }
    Ok(State::from_iterator(view.iter().copied()))
}

fn matrix8_from_numpy(arr: PyReadonlyArray2<'_, f64>) -> PyResult<Covariance> {
    let view = arr.as_array();
    if view.shape() != [8, 8] {
        return Err(PyValueError::new_err("covariance must have shape (8, 8)"));
    }
    Ok(Covariance::from_fn(|i, j| view[[i, j]]))
}

fn matrix_nx4_from_numpy(arr: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Measurement>> {
    let view = arr.as_array();
    if view.shape().len() != 2 || view.shape()[1] != 4 {
        return Err(PyValueError::new_err("measurements must have shape (N, 4)"));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|row| Measurement::from_iterator(row.iter().copied()))
        .collect())
}

fn vector_to_numpy<'py, const N: usize>(
    py: Python<'py>,
    v: &SVector<f64, N>,
) -> Bound<'py, PyArray1<f64>> {
    v.as_slice().to_vec().into_pyarray(py)
}

fn matrix_to_numpy<'py, const N: usize>(
    py: Python<'py>,
    m: &nalgebra::SMatrix<f64, N, N>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows: Vec<Vec<f64>> = (0..N).map(|i| (0..N).map(|j| m[(i, j)]).collect()).collect();
    PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
}

type MeanCovPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);

/// Filtre de Kalman DeepSORT (état 8D, mesure 4D) exposé à Python.
#[pyclass(name = "KalmanFilter")]
struct PyKalmanFilter {
    inner: RustKalmanFilter,
}

#[pymethods]
impl PyKalmanFilter {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustKalmanFilter::new(),
        }
    }

    fn initiate<'py>(
        &self,
        py: Python<'py>,
        measurement: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<MeanCovPair<'py>> {
        let measurement = vector4_from_numpy(measurement)?;
        let (mean, covariance) = self.inner.initiate(&measurement);
        Ok((
            vector_to_numpy(py, &mean),
            matrix_to_numpy(py, &covariance)?,
        ))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        mean: PyReadonlyArray1<'py, f64>,
        covariance: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<MeanCovPair<'py>> {
        let mean = vector8_from_numpy(mean)?;
        let covariance = matrix8_from_numpy(covariance)?;
        let (mean, covariance) = self.inner.predict(&mean, &covariance);
        Ok((
            vector_to_numpy(py, &mean),
            matrix_to_numpy(py, &covariance)?,
        ))
    }

    fn project<'py>(
        &self,
        py: Python<'py>,
        mean: PyReadonlyArray1<'py, f64>,
        covariance: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<MeanCovPair<'py>> {
        let mean = vector8_from_numpy(mean)?;
        let covariance = matrix8_from_numpy(covariance)?;
        let (projected_mean, projected_cov): (Measurement, ProjectedCovariance) =
            self.inner.project(&mean, &covariance);
        Ok((
            vector_to_numpy(py, &projected_mean),
            matrix_to_numpy(py, &projected_cov)?,
        ))
    }

    fn update<'py>(
        &self,
        py: Python<'py>,
        mean: PyReadonlyArray1<'py, f64>,
        covariance: PyReadonlyArray2<'py, f64>,
        measurement: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<MeanCovPair<'py>> {
        let mean = vector8_from_numpy(mean)?;
        let covariance = matrix8_from_numpy(covariance)?;
        let measurement = vector4_from_numpy(measurement)?;
        let (mean, covariance) = self.inner.update(&mean, &covariance, &measurement);
        Ok((
            vector_to_numpy(py, &mean),
            matrix_to_numpy(py, &covariance)?,
        ))
    }

    #[pyo3(signature = (mean, covariance, measurements, only_position=false))]
    fn gating_distance<'py>(
        &self,
        py: Python<'py>,
        mean: PyReadonlyArray1<'py, f64>,
        covariance: PyReadonlyArray2<'py, f64>,
        measurements: PyReadonlyArray2<'py, f64>,
        only_position: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mean = vector8_from_numpy(mean)?;
        let covariance = matrix8_from_numpy(covariance)?;
        let measurements = matrix_nx4_from_numpy(measurements)?;
        let distances = self
            .inner
            .gating_distance(&mean, &covariance, &measurements, only_position);
        Ok(distances.into_pyarray(py))
    }
}

fn boxes_from_numpy(arr: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<[f64; 4]>> {
    let view = arr.as_array();
    if view.shape().len() != 2 || view.shape()[1] != 4 {
        return Err(PyValueError::new_err("boxes must have shape (N, 4)"));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|r| [r[0] as f64, r[1] as f64, r[2] as f64, r[3] as f64])
        .collect())
}

fn embeddings_from_numpy(arr: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<Vec<f64>>> {
    let view = arr.as_array();
    Ok(view
        .rows()
        .into_iter()
        .map(|r| r.iter().map(|&v| v as f64).collect())
        .collect())
}

/// Tracker DeepSORT complet (Kalman + cascade d'apparence + IoU) exposé à Python.
#[pyclass(name = "Tracker")]
struct PyTracker {
    inner: RustTracker,
}

#[pymethods]
impl PyTracker {
    #[new]
    #[pyo3(signature = (max_age=30, n_init=3, max_cosine_distance=0.2, nn_budget=100, max_iou_distance=0.7))]
    fn new(
        max_age: u32,
        n_init: u32,
        max_cosine_distance: f64,
        nn_budget: Option<usize>,
        max_iou_distance: f64,
    ) -> Self {
        Self {
            inner: RustTracker::new(TrackerParams {
                max_age,
                n_init,
                max_cosine_distance,
                nn_budget,
                max_iou_distance,
            }),
        }
    }

    fn update<'py>(
        &mut self,
        py: Python<'py>,
        boxes: PyReadonlyArray2<'py, f32>,
        embeddings: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let boxes = boxes_from_numpy(boxes)?;
        let embeddings = embeddings_from_numpy(embeddings)?;
        if boxes.len() != embeddings.len() {
            return Err(PyValueError::new_err(
                "boxes and embeddings must have the same length",
            ));
        }

        let tracks = self.inner.update(&boxes, &embeddings);
        let rows: Vec<Vec<f64>> = tracks
            .iter()
            .map(|t| {
                vec![
                    t.id as f64,
                    t.ltrb[0],
                    t.ltrb[1],
                    t.ltrb[2],
                    t.ltrb[3],
                    t.age as f64,
                    if t.confirmed { 1.0 } else { 0.0 },
                ]
            })
            .collect();

        if rows.is_empty() {
            Ok(PyArray2::zeros(py, [0, 7], false))
        } else {
            PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
}

#[pymodule]
fn _deepsort_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKalmanFilter>()?;
    m.add_class::<PyTracker>()?;
    m.add("CHI2_95_4DOF", kalman::CHI2_95_4DOF)?;
    Ok(())
}
