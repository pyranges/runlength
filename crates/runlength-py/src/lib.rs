//! Python bindings for `runlength`, exposed as the `runlength` package.

mod batch;
mod convert;
mod rle;

use pyo3::prelude::*;
use runlength_arrow::ArrowRle;

use crate::batch::PyRleBatch;
use crate::convert::{OrRaise, array_from_py, ints_from_py, table_from_py};
use crate::rle::PyRle;

/// Coverage of half-open intervals [start, end) as an Rle.
///
/// With `values`, each position holds the sum of the values of the intervals
/// covering it, in the values' numeric type. Without, it holds the number of
/// covering intervals. Uncovered positions are 0. The result covers
/// 0..max(end), or 0..length if given.
#[pyfunction]
#[pyo3(signature = (starts, ends, values=None, length=None))]
fn coverage(
    py: Python<'_>,
    starts: &Bound<'_, PyAny>,
    ends: &Bound<'_, PyAny>,
    values: Option<&Bound<'_, PyAny>>,
    length: Option<i64>,
) -> PyResult<PyRle> {
    let starts = ints_from_py(starts, "starts")?;
    let ends = ints_from_py(ends, "ends")?;
    let values = values.map(array_from_py).transpose()?;
    let rle: ArrowRle = py
        .detach(|| runlength_arrow::coverage(&starts, &ends, values.as_deref(), length))
        .or_raise()?;
    Ok(rle.into())
}

/// Coverage per key, computed in parallel, as an RleBatch.
///
/// `keys` has one row per interval: a polars or pandas DataFrame, a pyarrow
/// Table, or a dict of columns (for example Chromosome and Strand). Keys
/// appear in order of first appearance; each Rle covers 0..max(end) of its
/// own intervals.
#[pyfunction]
#[pyo3(signature = (keys, starts, ends, values=None))]
fn coverage_by_key(
    py: Python<'_>,
    keys: &Bound<'_, PyAny>,
    starts: &Bound<'_, PyAny>,
    ends: &Bound<'_, PyAny>,
    values: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyRleBatch> {
    let keys = table_from_py(keys)?;
    let starts = ints_from_py(starts, "starts")?;
    let ends = ints_from_py(ends, "ends")?;
    let values = values.map(array_from_py).transpose()?;
    let batch = py
        .detach(|| runlength_arrow::coverage_by_key(&keys, &starts, &ends, values.as_deref()))
        .or_raise()?;
    Ok(batch.into())
}

#[pymodule]
fn _runlength(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRle>()?;
    m.add_class::<PyRleBatch>()?;
    m.add_function(wrap_pyfunction!(coverage, m)?)?;
    m.add_function(wrap_pyfunction!(coverage_by_key, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
