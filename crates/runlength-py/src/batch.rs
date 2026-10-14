use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::concat::concat;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};
use pyo3_arrow::PyTable;
use runlength::How;
use runlength_arrow::kernels::{self, Kernel};
use runlength_arrow::{ArrowRle, ArrowRleBatch, common_numeric_type};

use crate::convert::{
    OrRaise, scalar_of_type, table_capsule, table_from_py, table_to_py, to_arrow, to_pandas,
    to_polars, value_to_py,
};
use crate::rle::{PyRle, call_python, fill_of, format_values, kernel_for, parse_backend};

/// Many Rles, each identified by a row of key columns of any types (for
/// example Chromosome and Strand).
///
/// Build one with `coverage_by_key`, `Rle.getitems` or `RleBatch.from_rles`.
/// Operators combine two batches key by key (an outer join, padding missing
/// keys and shorter sequences with zero), in parallel.
#[pyclass(module = "runlength", name = "RleBatch", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRleBatch {
    pub(crate) inner: ArrowRleBatch,
}

impl From<ArrowRleBatch> for PyRleBatch {
    fn from(inner: ArrowRleBatch) -> Self {
        PyRleBatch { inner }
    }
}

/// A column name or a list of them.
#[derive(FromPyObject)]
pub(crate) enum StrOrList {
    One(String),
    Many(Vec<String>),
}

impl StrOrList {
    fn into_vec(self) -> Vec<String> {
        match self {
            StrOrList::One(s) => vec![s],
            StrOrList::Many(v) => v,
        }
    }
}

pub(crate) fn binary_kernel(op: &str) -> PyResult<Kernel> {
    match op {
        "add" => Ok(kernels::add),
        "sub" => Ok(kernels::sub),
        "mul" => Ok(kernels::mul),
        "div" => Ok(kernels::div),
        "and" => Ok(kernels::and),
        "or" => Ok(kernels::or),
        other => Err(PyValueError::new_err(format!(
            "op must be 'add', 'sub', 'mul', 'div', 'and' or 'or', got {other:?}"
        ))),
    }
}

fn parse_how(how: &str) -> PyResult<How> {
    match how {
        "outer" => Ok(How::Outer),
        "inner" => Ok(How::Inner),
        "left" => Ok(How::Left),
        "right" => Ok(How::Right),
        other => Err(PyValueError::new_err(format!(
            "how must be 'outer', 'inner', 'left' or 'right', got {other:?}"
        ))),
    }
}

impl PyRleBatch {
    fn binary(&self, other: &Bound<'_, PyAny>, kernel: Kernel) -> PyResult<PyRleBatch> {
        let other = other
            .extract::<PyRef<PyRleBatch>>()
            .map_err(|_| PyTypeError::new_err("RleBatch operators need another RleBatch"))?;
        let (a, b) = (&self.inner, &other.inner);
        let common = common_numeric_type(a.values().data_type(), b.values().data_type());
        let (a, b) = match common {
            Some(t) if a.values().data_type() != b.values().data_type() => {
                (cast_batch(a, &t)?, cast_batch(b, &t)?)
            }
            _ => (a.clone(), b.clone()),
        };
        Ok(a.zip_with_default(&b, kernel).or_raise()?.into())
    }

    /// One row per key: the key columns plus `name`, computed by `f`.
    fn aggregate(
        &self,
        name: &str,
        f: impl Fn(&ArrowRle) -> runlength_arrow::Result<ArrayRef>,
    ) -> PyResult<PyTable> {
        let columns = self
            .inner
            .iter()
            .map(|r| f(&r))
            .collect::<runlength_arrow::Result<Vec<_>>>()
            .or_raise()?;
        let data_type = columns
            .first()
            .map_or(DataType::Float64, |c| c.data_type().clone());
        let refs: Vec<&dyn Array> = columns.iter().map(|c| c.as_ref()).collect();
        let column = if refs.is_empty() {
            arrow_array::new_empty_array(&data_type)
        } else {
            concat(&refs).or_raise()?
        };
        let keys = self.inner.keys();
        let mut fields: Vec<Field> = keys
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new(name, data_type, true));
        let mut arrays = keys.columns().to_vec();
        arrays.push(column);
        table_to_py(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).or_raise()?)
    }

    fn float_aggregate(
        &self,
        name: &str,
        f: impl Fn(&ArrowRle) -> runlength_arrow::Result<Option<f64>>,
    ) -> PyResult<PyTable> {
        self.aggregate(name, |r| {
            Ok(Arc::new(Float64Array::from(vec![f(r)?])) as ArrayRef)
        })
    }

    /// The key at `position` as a tuple of Python values.
    fn key_tuple<'py>(&self, py: Python<'py>, position: usize) -> PyResult<Bound<'py, PyTuple>> {
        let values = self
            .inner
            .keys()
            .columns()
            .iter()
            .map(|c| value_to_py(py, c, position))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, values)
    }
}

fn cast_batch(batch: &ArrowRleBatch, data_type: &DataType) -> PyResult<ArrowRleBatch> {
    if batch.values().data_type() == data_type {
        return Ok(batch.clone());
    }
    let rles = batch
        .iter()
        .map(|r| r.cast(data_type))
        .collect::<runlength_arrow::Result<Vec<_>>>()
        .or_raise()?;
    ArrowRleBatch::from_rles(batch.keys().clone(), rles, data_type).or_raise()
}

#[pymethods]
impl PyRleBatch {
    /// Builds a batch from a key table (one row per Rle: a polars or pandas
    /// DataFrame, pyarrow Table, or dict of columns) and a list of Rles with
    /// the same value type.
    #[staticmethod]
    fn from_rles(keys: &Bound<'_, PyAny>, rles: Vec<PyRef<'_, PyRle>>) -> PyResult<Self> {
        let keys = table_from_py(keys)?;
        let data_type = rles
            .first()
            .map_or(DataType::Null, |r| r.inner.data_type().clone());
        let rles = rles.iter().map(|r| r.inner.clone()).collect();
        Ok(ArrowRleBatch::from_rles(keys, rles, &data_type)
            .or_raise()?
            .into())
    }

    /// The key table, one row per Rle.
    #[getter]
    fn keys(&self) -> PyResult<PyTable> {
        table_to_py(self.inner.keys().clone())
    }

    /// The Arrow type of the values.
    #[getter]
    fn dtype(&self) -> String {
        self.inner.values().data_type().to_string()
    }

    /// The number of Rles.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The Rle at a position (negative positions count from the end).
    fn __getitem__(&self, position: isize) -> PyResult<PyRle> {
        let n = self.inner.len() as isize;
        let index = if position < 0 { position + n } else { position };
        if !(0..n).contains(&index) {
            return Err(PyIndexError::new_err(format!(
                "position {position} out of range for {n} keys"
            )));
        }
        Ok(self
            .inner
            .get_index(index as usize)
            .expect("checked above")
            .into())
    }

    /// The Rle for a key, or None. A key is a tuple with one value per key
    /// column, or a single value if there is one key column.
    fn get(&self, key: &Bound<'_, PyAny>) -> PyResult<Option<PyRle>> {
        let keys = self.inner.keys();
        let parts: Vec<Bound<'_, PyAny>> = match key.cast::<PyTuple>() {
            Ok(tuple) => tuple.iter().collect(),
            Err(_) => vec![key.clone()],
        };
        if parts.len() != keys.num_columns() {
            return Err(PyValueError::new_err(format!(
                "key has {} parts but the batch has {} key columns",
                parts.len(),
                keys.num_columns()
            )));
        }
        let columns = parts
            .iter()
            .zip(keys.schema().fields())
            .map(|(value, field)| scalar_of_type(value, field.data_type()))
            .collect::<PyResult<Vec<_>>>()?;
        let lookup = RecordBatch::try_new(keys.schema(), columns).or_raise()?;
        let position = self.inner.positions(&lookup).or_raise()?[0];
        Ok(position
            .and_then(|p| self.inner.get_index(p))
            .map(Into::into))
    }

    /// (key tuple, Rle) pairs in key order.
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Vec<(Bound<'py, PyTuple>, PyRle)>> {
        (0..self.inner.len())
            .map(|i| {
                Ok((
                    self.key_tuple(py, i)?,
                    self.inner.get_index(i).expect("in range").into(),
                ))
            })
            .collect()
    }

    /// Combines with `other` key by key using `func`; see `Rle.zip_with`.
    /// `how` is "outer", "inner", "left" or "right". A key missing on one
    /// side is an empty Rle there, padded with that side's fill value.
    #[pyo3(signature = (other, func, how="outer", fill_self=None, fill_other=None, backend="arrow"))]
    #[allow(clippy::too_many_arguments)]
    fn zip_with(
        &self,
        py: Python<'_>,
        other: PyRef<'_, PyRleBatch>,
        func: &Bound<'_, PyAny>,
        how: &str,
        fill_self: Option<&Bound<'_, PyAny>>,
        fill_other: Option<&Bound<'_, PyAny>>,
        backend: &str,
    ) -> PyResult<Self> {
        let how = parse_how(how)?;
        let backend = parse_backend(backend)?;
        let fill_a = fill_of(fill_self, self.inner.values().data_type())?;
        let fill_b = fill_of(fill_other, other.inner.values().data_type())?;
        let (fa, fb) = (fill_a.as_deref(), fill_b.as_deref());
        let func: Py<PyAny> = func.clone().unbind();
        let (mine, theirs) = (&self.inner, &other.inner);
        // Keys are processed on rayon threads. The GIL is released here and
        // taken per callback; holding it while waiting would deadlock.
        let joined = py.detach(|| {
            mine.zip_with(theirs, how, fa, fb, |a, b| {
                Python::attach(|py| call_python(func.bind(py), backend, &[a, b]))
            })
        });
        Ok(joined.or_raise()?.into())
    }

    /// Combines the Rles whose keys agree on the remaining key columns, with
    /// `op` ("add", "sub", "mul", "div", "and" or "or"), padding with zero.
    /// Give the columns to `keep` or to `drop`, for example
    /// `batch.collapse(drop="Strand")` to add up the strands.
    #[pyo3(signature = (keep=None, drop=None, op="add"))]
    fn collapse(
        &self,
        keep: Option<Vec<String>>,
        drop: Option<StrOrList>,
        op: &str,
    ) -> PyResult<Self> {
        let names: Vec<String> = self
            .inner
            .keys()
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let keep = match (keep, drop) {
            (Some(keep), None) => keep,
            (None, Some(drop)) => {
                let drop = drop.into_vec();
                names.into_iter().filter(|n| !drop.contains(n)).collect()
            }
            _ => return Err(PyValueError::new_err("give exactly one of keep and drop")),
        };
        let keep: Vec<&str> = keep.iter().map(String::as_str).collect();
        let kernel = binary_kernel(op)?;
        Ok(self.inner.collapse(&keep, kernel).or_raise()?.into())
    }

    /// Sum per key, as a table of the key columns plus `sum`.
    fn sum(&self) -> PyResult<PyTable> {
        self.aggregate("sum", |r| r.sum())
    }

    /// Mean per key, as a table of the key columns plus `mean`.
    fn mean(&self) -> PyResult<PyTable> {
        self.float_aggregate("mean", |r| r.mean())
    }

    /// Variance per key, as a table of the key columns plus `var`.
    #[pyo3(signature = (ddof=1))]
    fn var(&self, ddof: i64) -> PyResult<PyTable> {
        self.float_aggregate("var", |r| r.variance(ddof))
    }

    /// Standard deviation per key, as a table of the key columns plus `std`.
    #[pyo3(signature = (ddof=1))]
    fn std(&self, ddof: i64) -> PyResult<PyTable> {
        self.float_aggregate("std", |r| r.std(ddof))
    }

    /// Minimum per key, as a table of the key columns plus `min`.
    fn min(&self) -> PyResult<PyTable> {
        self.aggregate("min", |r| r.min())
    }

    /// Maximum per key, as a table of the key columns plus `max`.
    fn max(&self) -> PyResult<PyTable> {
        self.aggregate("max", |r| r.max())
    }

    /// The runs of all keys as a pyarrow Table: the key columns, then start,
    /// end and value.
    fn to_arrow<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_arrow(slf.as_any())
    }

    /// The runs of all keys as a polars DataFrame: the key columns, then
    /// start, end and value.
    fn to_polars<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_polars(slf.as_any())
    }

    /// The runs of all keys as a pandas DataFrame: the key columns, then
    /// start, end and value.
    fn to_pandas<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_pandas(slf.as_any())
    }

    /// Exports the run table as an Arrow stream.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        table_capsule(
            py,
            self.inner.to_record_batch().or_raise()?,
            requested_schema,
        )
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::add)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::sub)
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::mul)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other = other
            .extract::<PyRef<PyRleBatch>>()
            .map_err(|_| PyTypeError::new_err("RleBatch operators need another RleBatch"))?;
        let a = cast_batch(&self.inner, &DataType::Float64)?;
        let b = cast_batch(&other.inner, &DataType::Float64)?;
        Ok(a.zip_with_default(&b, kernels::div).or_raise()?.into())
    }
    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::and)
    }
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::or)
    }

    /// Element-wise comparison per key, returning a batch of boolean Rles.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Self> {
        self.binary(other, kernel_for(op))
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "the truth value of an RleBatch is ambiguous; use len(batch)",
        ))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let n = self.inner.len();
        let mut lines = vec![format!(
            "RleBatch(keys={}, runs={}, dtype={})",
            n,
            self.inner.ends().len(),
            self.inner.values().data_type()
        )];
        for i in (0..n).take(5) {
            let rle = self.inner.get_index(i).expect("in range");
            lines.push(format!(
                "  {}: length={}, values={}",
                self.key_tuple(py, i)?.repr()?,
                rle.len(),
                format_values(rle.values().as_ref(), 6)
            ));
        }
        if n > 5 {
            lines.push(format!("  ... and {} more", n - 5));
        }
        Ok(lines.join("\n"))
    }
}
