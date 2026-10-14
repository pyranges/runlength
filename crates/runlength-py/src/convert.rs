//! Converting between Python objects and Arrow data.
//!
//! Input is accepted from anything that speaks the Arrow PyCapsule interface
//! (polars, pyarrow, pandas, arro3, nanoarrow), from numpy arrays, and from
//! plain Python lists. Output is pyo3-arrow's `Array` and `Table` classes,
//! which implement the same interface, so any of those libraries can read
//! them without copying.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, NullArray, RecordBatch, StringArray,
    StructArray, new_empty_array,
};
use arrow_cast::cast;
use arrow_schema::{ArrowError, DataType, Field};
use arrow_select::concat::{concat, concat_batches};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};
use pyo3_arrow::ffi::{ArrayIterator, to_stream_pycapsule};
use pyo3_arrow::input::{AnyArray, AnyRecordBatch};
use pyo3_arrow::{PyArray, PyScalar, PyTable};

/// Converts a `runlength_arrow` error, re-raising Python exceptions that
/// came from a Python callback unchanged.
pub fn to_pyerr(error: runlength_arrow::Error) -> PyErr {
    use runlength_arrow::Error;
    match error {
        Error::Arrow(ArrowError::ExternalError(inner)) => match inner.downcast::<PyErr>() {
            Ok(err) => *err,
            Err(other) => PyValueError::new_err(other.to_string()),
        },
        Error::InKey { position, source } => {
            let err = to_pyerr(*source);
            Python::attach(|py| {
                let message = format!("key at output position {position}: {}", err.value(py));
                PyErr::from_type(err.get_type(py), message)
            })
        }
        e @ (Error::UnsupportedType(_) | Error::InvalidFill { .. } | Error::KeySchemaMismatch) => {
            PyTypeError::new_err(e.to_string())
        }
        e => PyValueError::new_err(e.to_string()),
    }
}

/// `?`-friendly conversion of `runlength_arrow` results.
pub trait OrRaise<T> {
    fn or_raise(self) -> PyResult<T>;
}

impl<T> OrRaise<T> for runlength_arrow::Result<T> {
    fn or_raise(self) -> PyResult<T> {
        self.map_err(to_pyerr)
    }
}

impl<T> OrRaise<T> for Result<T, ArrowError> {
    fn or_raise(self) -> PyResult<T> {
        self.map_err(|e| to_pyerr(e.into()))
    }
}

fn concat_chunks(chunks: Vec<ArrayRef>, data_type: &DataType) -> PyResult<ArrayRef> {
    match chunks.len() {
        0 => Ok(new_empty_array(data_type)),
        1 => Ok(chunks.into_iter().next().expect("one chunk")),
        _ => {
            let refs: Vec<&dyn Array> = chunks.iter().map(|c| c.as_ref()).collect();
            concat(&refs).or_raise()
        }
    }
}

/// Reads a one-dimensional array from Python.
pub fn array_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    let py = obj.py();
    if obj.hasattr(intern!(py, "__arrow_c_array__"))?
        || obj.hasattr(intern!(py, "__arrow_c_stream__"))?
    {
        let any: AnyArray = obj.extract()?;
        let (chunks, field) = any.into_chunked_array()?.into_inner();
        return concat_chunks(chunks, field.data_type());
    }
    // numpy arrays and other buffer-protocol objects.
    if let Ok(array) = obj.extract::<PyArray>() {
        return Ok(array.into_inner().0);
    }
    // pandas Series and Index without Arrow export.
    if obj.hasattr(intern!(py, "to_numpy"))? && !obj.is_instance_of::<pyo3::types::PyList>() {
        let converted = obj.call_method0(intern!(py, "to_numpy"))?;
        return array_from_py(&converted);
    }
    if let Ok(values) = obj.extract::<Vec<Option<bool>>>() {
        return Ok(Arc::new(BooleanArray::from(values)));
    }
    if let Ok(values) = obj.extract::<Vec<Option<i64>>>() {
        return Ok(Arc::new(Int64Array::from(values)));
    }
    if let Ok(values) = obj.extract::<Vec<Option<f64>>>() {
        return Ok(Arc::new(Float64Array::from(values)));
    }
    if let Ok(values) = obj.extract::<Vec<Option<String>>>() {
        return Ok(Arc::new(StringArray::from(values)));
    }
    Err(PyTypeError::new_err(format!(
        "expected an Arrow array, polars or pandas Series, numpy array or list, got {}",
        obj.get_type().name()?
    )))
}

/// Reads positions or lengths: integers without nulls.
pub fn ints_from_py(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<i64>> {
    let array = array_from_py(obj)?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "{what} must not contain nulls"
        )));
    }
    if !array.data_type().is_integer() && !array.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "{what} must be integers, got {}",
            array.data_type()
        )));
    }
    let array = cast(array.as_ref(), &DataType::Int64).or_raise()?;
    Ok(array.as_primitive::<Int64Type>().values().to_vec())
}

/// Reads a boolean mask without nulls.
pub fn bools_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    let array = array_from_py(obj)?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err("mask must not contain nulls"));
    }
    let array = cast(array.as_ref(), &DataType::Boolean).or_raise()?;
    Ok(array.as_boolean().values().iter().collect())
}

/// Reads a single value as a length-1 array.
pub fn scalar_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    if obj.is_none() {
        return Ok(Arc::new(NullArray::new(1)));
    }
    if let Ok(v) = obj.extract::<bool>() {
        return Ok(Arc::new(BooleanArray::from(vec![v])));
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Arc::new(Int64Array::from(vec![v])));
    }
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(Arc::new(Float64Array::from(vec![v])));
    }
    if let Ok(v) = obj.extract::<String>() {
        return Ok(Arc::new(StringArray::from(vec![v])));
    }
    let array = array_from_py(obj)?;
    if array.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "expected a single value, got {} values",
            array.len()
        )));
    }
    Ok(array)
}

/// A `fill` argument that tells an omitted value (`Default`) from an
/// explicit `None` (`Null`).
pub enum FillArg {
    Default,
    Null,
    Value(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for FillArg {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_none() {
            Ok(FillArg::Null)
        } else {
            Ok(FillArg::Value(obj.to_owned().unbind()))
        }
    }
}

/// Reads a single value and casts it to `data_type`.
pub fn scalar_of_type(obj: &Bound<'_, PyAny>, data_type: &DataType) -> PyResult<ArrayRef> {
    let value = scalar_from_py(obj)?;
    if value.data_type() == data_type {
        return Ok(value);
    }
    cast(value.as_ref(), data_type).or_raise()
}

/// Reads a table of key columns: anything exporting an Arrow stream (polars
/// or pandas DataFrame, pyarrow Table) or a dict of column name to array.
pub fn table_from_py(obj: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        let columns = dict
            .iter()
            .map(|(name, values)| Ok((name.extract::<String>()?, array_from_py(&values)?)))
            .collect::<PyResult<Vec<_>>>()?;
        return RecordBatch::try_from_iter(columns).or_raise();
    }
    let any: AnyRecordBatch = obj.extract()?;
    let (batches, schema) = any.into_table()?.into_inner();
    concat_batches(&schema, &batches).or_raise()
}

/// Whether `obj` is a table (a DataFrame, pyarrow or arro3 Table, or a dict
/// of columns) rather than a single array or value.
pub fn is_table_like(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(obj.cast::<PyDict>().is_ok()
        || obj.hasattr(intern!(obj.py(), "columns"))?
        || obj.hasattr(intern!(obj.py(), "column_names"))?)
}

/// Reads key columns: a table (see [`table_from_py`]), or a single array
/// (a polars or pandas Series, pyarrow or numpy array, or list) as one
/// column named after the Series, or `key`.
pub fn keys_from_py(obj: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    if is_table_like(obj)? {
        return table_from_py(obj);
    }
    let name = obj
        .getattr(intern!(obj.py(), "name"))
        .ok()
        .and_then(|n| n.extract::<String>().ok())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "key".to_string());
    RecordBatch::try_from_iter([(name, array_from_py(obj)?)]).or_raise()
}

/// Exports indices to Python as `int64`, the index type numpy, polars and
/// pandas all accept.
pub fn indices_to_py(indices: impl IntoIterator<Item = usize>) -> PyArray {
    array_to_py(Arc::new(Int64Array::from_iter_values(
        indices.into_iter().map(|i| i as i64),
    )))
}

/// Exports an array to Python.
pub fn array_to_py(array: ArrayRef) -> PyArray {
    PyArray::from_array_ref(array)
}

/// Exports a table to Python.
pub fn table_to_py(batch: RecordBatch) -> PyResult<PyTable> {
    let schema = batch.schema();
    PyTable::try_new(vec![batch], schema)
}

/// Converts value `index` of `array` to a Python object.
pub fn value_to_py(py: Python<'_>, array: &ArrayRef, index: usize) -> PyResult<Py<PyAny>> {
    let scalar = PyScalar::try_from_array_ref(array.slice(index, 1))?;
    Ok(scalar
        .into_pyobject(py)?
        .call_method0(intern!(py, "as_py"))?
        .unbind())
}

/// Implements `__arrow_c_stream__` for a table.
pub fn table_capsule<'py>(
    py: Python<'py>,
    batch: RecordBatch,
    requested_schema: Option<Bound<'py, PyCapsule>>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let schema = batch.schema();
    let field = Field::new_struct("", schema.fields().clone(), false)
        .with_metadata(schema.metadata().clone());
    let array: ArrayRef = Arc::new(StructArray::from(batch));
    let reader = Box::new(ArrayIterator::new(vec![Ok(array)], Arc::new(field)));
    Ok(to_stream_pycapsule(py, reader, requested_schema)?)
}

/// `pyarrow.table(obj)`, `polars.DataFrame(obj)` and pandas via pyarrow, for
/// objects with `__arrow_c_stream__`.
pub fn to_arrow<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    py.import(intern!(py, "pyarrow"))?
        .call_method1(intern!(py, "table"), (obj,))
}

pub fn to_polars<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    py.import(intern!(py, "polars"))?
        .call_method1(intern!(py, "DataFrame"), (obj,))
}

pub fn to_pandas<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    to_arrow(obj)?.call_method0(intern!(obj.py(), "to_pandas"))
}
