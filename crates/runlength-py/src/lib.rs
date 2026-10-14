//! Python bindings for `runlength`, exposed as the `runlength` package.

mod batch;
mod convert;
mod rle;

use std::sync::Arc;

use arrow_array::Int64Array;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use runlength_arrow::{Agg, ArrowRle, ArrowRleBatch};

use crate::batch::PyRleBatch;
use crate::convert::{
    FillArg, OrRaise, array_from_py, array_to_py, indices_to_py, ints_from_py, scalar_of_type,
    table_from_py,
};
use crate::rle::{PyRle, call_python, parse_backend};

fn parse_agg(agg: &str, has_values: bool) -> PyResult<Agg> {
    match agg {
        "sum" if has_values => Ok(Agg::Sum),
        "sum" | "count" => Ok(Agg::Count),
        "max" => Ok(Agg::Max),
        "min" => Ok(Agg::Min),
        other => Err(PyValueError::new_err(format!(
            "agg must be 'sum', 'count', 'max' or 'min', got {other:?}"
        ))),
    }
}

/// Whether coverage can use the fast path, which fills uncovered positions
/// with 0.
fn zero_filled(agg: Agg, fill: &FillArg, py: Python<'_>) -> bool {
    let zero = match fill {
        FillArg::Default => true,
        FillArg::Null => false,
        FillArg::Value(value) => value.bind(py).extract::<i64>().ok() == Some(0),
    };
    matches!(agg, Agg::Sum | Agg::Count) && zero
}

/// Fills the nulls left by `coverage_agg` at uncovered positions.
fn apply_fill(py: Python<'_>, rle: ArrowRle, agg: Agg, fill: &FillArg) -> PyResult<ArrowRle> {
    let value = match fill {
        _ if agg == Agg::Count => return Ok(rle),
        FillArg::Null => return Ok(rle),
        FillArg::Default => 0i64.into_pyobject(py)?.into_any().unbind(),
        FillArg::Value(value) => value.clone_ref(py),
    };
    let fill = scalar_of_type(value.bind(py), rle.data_type())?;
    rle.fill_null(fill.as_ref()).or_raise()
}

/// Coverage of half-open intervals [start, end) as an Rle.
///
/// `agg` combines the intervals covering each position: "sum" of `values`
/// (the number of intervals if no values are given), "count", "max" or
/// "min" (numbers or strings). Uncovered positions hold `fill`, 0 by
/// default; `fill=None` leaves them null. The result covers 0..max(end), or
/// 0..length if given.
#[pyfunction]
#[pyo3(signature = (starts, ends, values=None, length=None, agg="sum", fill=FillArg::Default))]
fn coverage(
    py: Python<'_>,
    starts: &Bound<'_, PyAny>,
    ends: &Bound<'_, PyAny>,
    values: Option<&Bound<'_, PyAny>>,
    length: Option<i64>,
    agg: &str,
    fill: FillArg,
) -> PyResult<PyRle> {
    let agg = parse_agg(agg, values.is_some())?;
    let starts = ints_from_py(starts, "starts")?;
    let ends = ints_from_py(ends, "ends")?;
    let values = values.map(array_from_py).transpose()?;
    let rle = if zero_filled(agg, &fill, py) {
        py.detach(|| runlength_arrow::coverage(&starts, &ends, values.as_deref(), length))
            .or_raise()?
    } else {
        let rle = py
            .detach(|| {
                runlength_arrow::coverage_agg(&starts, &ends, values.as_deref(), length, agg)
            })
            .or_raise()?;
        apply_fill(py, rle, agg, &fill)?
    };
    Ok(rle.into())
}

/// Coverage per key, computed in parallel, as an RleBatch.
///
/// `keys` has one row per interval: a polars or pandas DataFrame, a pyarrow
/// Table, or a dict of columns (for example Chromosome and Strand). Keys
/// appear in order of first appearance; each Rle covers 0..max(end) of its
/// own intervals. `agg` and `fill` are as in `coverage`.
#[pyfunction]
#[pyo3(signature = (keys, starts, ends, values=None, agg="sum", fill=FillArg::Default))]
fn coverage_by_key(
    py: Python<'_>,
    keys: &Bound<'_, PyAny>,
    starts: &Bound<'_, PyAny>,
    ends: &Bound<'_, PyAny>,
    values: Option<&Bound<'_, PyAny>>,
    agg: &str,
    fill: FillArg,
) -> PyResult<PyRleBatch> {
    let agg = parse_agg(agg, values.is_some())?;
    let keys = table_from_py(keys)?;
    let starts = ints_from_py(starts, "starts")?;
    let ends = ints_from_py(ends, "ends")?;
    let values = values.map(array_from_py).transpose()?;
    if zero_filled(agg, &fill, py) {
        let batch = py
            .detach(|| runlength_arrow::coverage_by_key(&keys, &starts, &ends, values.as_deref()))
            .or_raise()?;
        return Ok(batch.into());
    }
    let batch = py
        .detach(|| {
            runlength_arrow::coverage_agg_by_key(&keys, &starts, &ends, values.as_deref(), agg)
        })
        .or_raise()?;
    let rles = batch
        .iter()
        .map(|r| apply_fill(py, r, agg, &fill))
        .collect::<PyResult<Vec<_>>>()?;
    let data_type = rles
        .first()
        .map_or(batch.values().data_type().clone(), |r| {
            r.data_type().clone()
        });
    Ok(
        ArrowRleBatch::from_rles(batch.keys().clone(), rles, &data_type)
            .or_raise()?
            .into(),
    )
}

/// Joins Rles end to end, merging identical neighbours.
#[pyfunction]
fn concat(rles: Vec<PyRef<'_, PyRle>>) -> PyResult<PyRle> {
    let rles: Vec<ArrowRle> = rles.iter().map(|r| r.inner.clone()).collect();
    Ok(runlength_arrow::concat(&rles).or_raise()?.into())
}

/// The plan for combining several Rles: (ends, [idx per Rle]). An index
/// equal to that Rle's number of runs means the segment is past its end.
#[pyfunction]
fn plan_many(rles: Vec<PyRef<'_, PyRle>>) -> (PyArray, Vec<PyArray>) {
    let ends: Vec<&[i64]> = rles.iter().map(|r| r.inner.ends()).collect();
    let plan = runlength::plan_many(&ends);
    let idx = plan.idx.into_iter().map(indices_to_py).collect();
    (array_to_py(Arc::new(Int64Array::from(plan.ends))), idx)
}

/// Combines several Rles position by position: `func(*arrays)` receives one
/// array per Rle (their values on each segment) and returns one value per
/// segment. Shorter Rles are padded with `fill` (one value for all; by
/// default unequal lengths are an error). `backend` is as in `Rle.zip_with`.
#[pyfunction]
#[pyo3(signature = (rles, func, fill=None, backend="arrow"))]
fn zip_many(
    rles: Vec<PyRef<'_, PyRle>>,
    func: &Bound<'_, PyAny>,
    fill: Option<&Bound<'_, PyAny>>,
    backend: &str,
) -> PyResult<PyRle> {
    let backend = parse_backend(backend)?;
    let rles: Vec<ArrowRle> = rles.iter().map(|r| r.inner.clone()).collect();
    let fills = rles
        .iter()
        .map(|r| fill.map(|f| scalar_of_type(f, r.data_type())).transpose())
        .collect::<PyResult<Vec<_>>>()?;
    let fill_refs: Vec<Option<&dyn arrow_array::Array>> =
        fills.iter().map(|f| f.as_deref()).collect();
    let out = runlength_arrow::zip_many(&rles, &fill_refs, |columns| {
        let refs: Vec<&dyn arrow_array::Array> = columns.iter().map(|c| c.as_ref()).collect();
        call_python(func, backend, &refs)
    });
    Ok(out.or_raise()?.into())
}

/// Which intervals cover each segment: (ends, offsets, ids), where segment s
/// ends at ends[s] and is covered by intervals ids[offsets[s]:offsets[s+1]].
/// Aggregate interval values per segment in any way, for example with a
/// polars group_by.
#[pyfunction]
#[pyo3(signature = (starts, ends, length=None))]
fn overlap_plan(
    py: Python<'_>,
    starts: &Bound<'_, PyAny>,
    ends: &Bound<'_, PyAny>,
    length: Option<i64>,
) -> PyResult<(PyArray, PyArray, PyArray)> {
    let starts = ints_from_py(starts, "starts")?;
    let ends = ints_from_py(ends, "ends")?;
    let plan = py
        .detach(|| runlength::overlap_plan(&starts, &ends, length))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        array_to_py(Arc::new(Int64Array::from(plan.ends))),
        indices_to_py(plan.offsets),
        indices_to_py(plan.interval_ids),
    ))
}

#[pymodule]
fn _runlength(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRle>()?;
    m.add_class::<PyRleBatch>()?;
    m.add_function(wrap_pyfunction!(coverage, m)?)?;
    m.add_function(wrap_pyfunction!(coverage_by_key, m)?)?;
    m.add_function(wrap_pyfunction!(concat, m)?)?;
    m.add_function(wrap_pyfunction!(plan_many, m)?)?;
    m.add_function(wrap_pyfunction!(zip_many, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_plan, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
