use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, RecordBatchOptions};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};
use arrow_select::concat::concat;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};
use pyo3_arrow::{PyArray, PyTable};
use runlength::How;
use runlength_arrow::kernels::Kernel;
use runlength_arrow::{ArrowRle, ArrowRleBatch, RunExtents, common_numeric_type};

use crate::convert::{
    FillArg, OrRaise, array_to_py, ints_from_array, ints_from_py, is_table_like, keys_from_py,
    scalar_of_type, table_capsule, table_from_py, table_to_py, to_arrow, to_pandas, to_polars,
    value_to_py,
};
use crate::rle::{
    PyRle, call_python, combine_scalar, fill_array, fill_of, format_values, kernel_for,
    parse_backend, parse_slice_agg, scalar_operand,
};

/// Many Rles, each identified by a row of key columns of any types (for
/// example Chromosome and Strand).
///
/// Build one with `coverage_by_key`, `Rle.getitems`, `RleBatch.from_table` or
/// `RleBatch.from_rles`.
/// Operators combine two batches key by key (an outer join, padding missing
/// keys and shorter sequences with zero), in parallel.
///
/// `len(batch)` is the number of Rles and `batch[i]` the Rle at position `i`;
/// look one up by key with `get`.
///
/// Examples:
///     >>> keys = pl.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "strand": ["+", "-", "+"]})
///     >>> batch = rl.coverage_by_key(keys, [0, 2, 5], [3, 4, 7])
///     >>> batch
///     RleBatch(keys=3, runs=5, dtype=Int64)
///       ('chr1', '+'): length=3, runs=[3], values=[1]
///       ('chr1', '-'): length=4, runs=[2, 2], values=[0, 1]
///       ('chr2', '+'): length=7, runs=[5, 2], values=[0, 1]
///     >>> len(batch)
///     3
///     >>> batch[-1]
///     Rle(length=7, runs=2, dtype=Int64)
///       runs:   [5, 2]
///       values: [0, 1]
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
        "add" => Ok(Kernel::Add),
        "sub" => Ok(Kernel::Sub),
        "mul" => Ok(Kernel::Mul),
        "div" => Ok(Kernel::Div),
        "and" => Ok(Kernel::And),
        "or" => Ok(Kernel::Or),
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
    /// `self <kernel> other` for another batch (keys outer-joined) or a single
    /// value (applied to every Rle); `reflected` swaps the sides for a value.
    /// With `to_float` (for `/`), integers become floats first.
    fn binary(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        kernel: Kernel,
        reflected: bool,
        to_float: bool,
    ) -> PyResult<PyRleBatch> {
        let Ok(other) = other.extract::<PyRef<PyRleBatch>>() else {
            let value = scalar_operand(other, "RleBatch")?;
            let inner = &self.inner;
            let out = py.detach(|| {
                inner.map_rles(|r| combine_scalar(r, &value, kernel, reflected, to_float))
            });
            return Ok(out.or_raise()?.into());
        };
        let (a, b) = (&self.inner, &other.inner);
        let (ta, tb) = (a.values().data_type(), b.values().data_type());
        let is_string =
            |t: &DataType| matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View);
        let common = if to_float {
            Some(DataType::Float64)
        } else if ta != tb && is_string(ta) && is_string(tb) {
            // Different string encodings, e.g. from polars and from a list.
            Some(ta.clone())
        } else {
            common_numeric_type(ta, tb)
        };
        let (a, b) = match common {
            Some(t) if a.values().data_type() != &t || b.values().data_type() != &t => {
                (cast_batch(a, &t)?, cast_batch(b, &t)?)
            }
            _ => (a.clone(), b.clone()),
        };
        Ok(py
            .detach(|| a.zip_with_default(&b, kernel))
            .or_raise()?
            .into())
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

/// Reads lookup keys for a batch: a table whose columns have the batch's key
/// names (cast to the batch's key types), or a single array if the batch has
/// one key column.
fn keys_like(batch: &ArrowRleBatch, obj: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    let schema = batch.keys().schema();
    // A single array (not a table) matches a single key column whatever its
    // name; table columns are matched by name.
    let single = !is_table_like(obj)? && schema.fields().len() == 1;
    let table = keys_from_py(obj)?;
    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            let column = if single {
                Some(table.column(0))
            } else {
                table.column_by_name(field.name())
            };
            let column = column.ok_or_else(|| {
                PyValueError::new_err(format!("keys have no column {:?}", field.name()))
            })?;
            cast(column.as_ref(), field.data_type()).or_raise()
        })
        .collect::<PyResult<Vec<_>>>()?;
    RecordBatch::try_new_with_options(
        schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(table.num_rows())),
    )
    .or_raise()
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
    ///
    /// Examples:
    ///     >>> rl.RleBatch.from_rles({"name": ["a", "b"]}, [rl.Rle([2], [1]), rl.Rle([1, 1], [3, 4])])
    ///     RleBatch(keys=2, runs=3, dtype=Int64)
    ///       ('a',): length=2, runs=[2], values=[1]
    ///       ('b',): length=2, runs=[1, 1], values=[3, 4]
    #[staticmethod]
    fn from_rles(keys: &Bound<'_, PyAny>, rles: Vec<PyRef<'_, PyRle>>) -> PyResult<Self> {
        let keys = keys_from_py(keys)?;
        let data_type = rles
            .first()
            .map_or(DataType::Null, |r| r.inner.data_type().clone());
        let rles = rles.iter().map(|r| r.inner.clone()).collect();
        Ok(ArrowRleBatch::from_rles(keys, rles, &data_type)
            .or_raise()?
            .into())
    }

    /// Builds a batch from a table with one row per run: a polars or pandas
    /// DataFrame, pyarrow Table, or dict of columns. This is the inverse of
    /// `to_polars`, `to_pandas` and `to_arrow`.
    ///
    /// `keys` names the key column(s). Each run lies at `starts`..`ends`;
    /// each key's runs must start at 0 and follow each other without gaps
    /// (for intervals with gaps, use `coverage_by_key`). With `starts=None`,
    /// each run ends at `ends` and starts where the previous one ended, and
    /// with `runs`, each run has that length. Keys appear in order of first
    /// appearance, and a key's rows need not be adjacent, but they must be in
    /// position order.
    ///
    /// Examples:
    ///     >>> table = pl.DataFrame({
    ///     ...     "chrom": ["chr1", "chr1", "chr2"],
    ///     ...     "start": [0, 3, 0],
    ///     ...     "end": [3, 5, 4],
    ///     ...     "value": [1, 2, 7],
    ///     ... })
    ///     >>> rl.RleBatch.from_table(table, "chrom")
    ///     RleBatch(keys=2, runs=3, dtype=Int64)
    ///       ('chr1',): length=5, runs=[3, 2], values=[1, 2]
    ///       ('chr2',): length=4, runs=[4], values=[7]
    ///     >>> lengths = {"chrom": ["chr1", "chr1"], "width": [3, 2], "score": [0.5, 1.5]}
    ///     >>> rl.RleBatch.from_table(lengths, "chrom", values="score", runs="width")
    ///     RleBatch(keys=1, runs=2, dtype=Float64)
    ///       ('chr1',): length=5, runs=[3, 2], values=[0.5, 1.5]
    #[staticmethod]
    #[pyo3(signature = (table, keys, values="value", starts=Some("start"), ends="end", runs=None))]
    fn from_table(
        py: Python<'_>,
        table: &Bound<'_, PyAny>,
        keys: StrOrList,
        values: &str,
        starts: Option<&str>,
        ends: &str,
        runs: Option<&str>,
    ) -> PyResult<Self> {
        let table = table_from_py(table)?;
        let column = |name: &str| {
            table.column_by_name(name).cloned().ok_or_else(|| {
                let hint = if Some(name) == starts {
                    "; pass starts=None if each run starts where the previous one ended"
                } else {
                    ""
                };
                PyValueError::new_err(format!("table has no column {name:?}{hint}"))
            })
        };
        let keys = keys.into_vec();
        if keys.is_empty() {
            return Err(PyValueError::new_err("keys must name at least one column"));
        }
        let key_columns = keys
            .iter()
            .map(|k| Ok((k.as_str(), column(k)?)))
            .collect::<PyResult<Vec<_>>>()?;
        let key_table = RecordBatch::try_from_iter(key_columns).or_raise()?;
        let values = column(values)?;
        let (lengths, start_values, end_values);
        let extents = if let Some(runs) = runs {
            lengths = ints_from_array(&column(runs)?, "runs")?;
            RunExtents::Lengths(&lengths)
        } else {
            end_values = ints_from_array(&column(ends)?, "ends")?;
            match starts {
                Some(starts) => {
                    start_values = ints_from_array(&column(starts)?, "starts")?;
                    RunExtents::StartsEnds(&start_values, &end_values)
                }
                None => RunExtents::Ends(&end_values),
            }
        };
        let batch = py
            .detach(|| ArrowRleBatch::from_run_table(&key_table, extents, values))
            .or_raise()?;
        Ok(batch.into())
    }

    /// The key table, one row per Rle.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
    ///     >>> pl.DataFrame(batch.keys)
    ///     shape: (2, 1)
    ///     ┌───────┐
    ///     │ chrom │
    ///     │ ---   │
    ///     │ str   │
    ///     ╞═══════╡
    ///     │ chr1  │
    ///     │ chr2  │
    ///     └───────┘
    #[getter]
    fn keys(&self) -> PyResult<PyTable> {
        table_to_py(self.inner.keys().clone())
    }

    /// The Arrow type of the values.
    ///
    /// Examples:
    ///     >>> rl.coverage_by_key({"chrom": ["chr1"]}, [0], [3], values=[0.5]).dtype
    ///     'Float64'
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
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1"], "strand": ["+", "-"]}, [0, 2], [3, 4])
    ///     >>> batch.get(("chr1", "-"))
    ///     Rle(length=4, runs=2, dtype=Int64)
    ///       runs:   [2, 2]
    ///       values: [0, 1]
    ///     >>> batch.get(("chr9", "+")) is None
    ///     True
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
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
    ///     >>> [(key, rle.num_runs) for key, rle in batch.items()]
    ///     [(('chr1',), 1), (('chr2',), 2)]
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
    ///
    /// Examples:
    ///     >>> a = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
    ///     >>> b = rl.coverage_by_key({"chrom": ["chr1", "chr3"]}, [1, 0], [2, 2])
    ///     >>> a.zip_with(b, lambda x, y: x * y, how="inner", fill_other=0, backend="numpy")
    ///     RleBatch(keys=1, runs=3, dtype=Int64)
    ///       ('chr1',): length=3, runs=[1, 1, 1], values=[0, 1, 0]
    ///     >>> a.zip_with(b, lambda x, y: x + y, how="left", fill_self=0, fill_other=0, backend="numpy")
    ///     RleBatch(keys=2, runs=5, dtype=Int64)
    ///       ('chr1',): length=3, runs=[1, 1, 1], values=[1, 2, 1]
    ///       ('chr2',): length=7, runs=[5, 2], values=[0, 1]
    ///     >>> a + b
    ///     RleBatch(keys=3, runs=6, dtype=Int64)
    ///       ('chr1',): length=3, runs=[1, 1, 1], values=[1, 2, 1]
    ///       ('chr2',): length=7, runs=[5, 2], values=[0, 1]
    ///       ('chr3',): length=2, runs=[2], values=[1]
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
    ///
    /// Examples:
    ///     >>> keys = {"chrom": ["chr1", "chr1", "chr2"], "strand": ["+", "-", "+"]}
    ///     >>> batch = rl.coverage_by_key(keys, [0, 2, 5], [3, 4, 7])
    ///     >>> batch.collapse(drop="strand")
    ///     RleBatch(keys=2, runs=5, dtype=Int64)
    ///       ('chr1',): length=4, runs=[2, 1, 1], values=[1, 2, 1]
    ///       ('chr2',): length=7, runs=[5, 2], values=[0, 1]
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

    /// Reduces many slices without copying them, in parallel: entry k reduces
    /// positions starts[k]:ends[k] of the Rle for row k of `keys` (a table
    /// with the batch's key columns, or one array if there is one key
    /// column). A key not in the batch gives an empty slice. `agg` and `ddof`
    /// are as in `Rle.slice_agg`.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
    ///     >>> batch.slice_agg(["chr1", "chr2"], [0, 4], [4, 7], agg="max").to_pylist()
    ///     [2, 1]
    #[pyo3(signature = (keys, starts, ends, agg="mean", ddof=1))]
    fn slice_agg(
        &self,
        py: Python<'_>,
        keys: &Bound<'_, PyAny>,
        starts: &Bound<'_, PyAny>,
        ends: &Bound<'_, PyAny>,
        agg: &str,
        ddof: i64,
    ) -> PyResult<PyArray> {
        let agg = parse_slice_agg(agg, ddof)?;
        let keys = keys_like(&self.inner, keys)?;
        let starts = ints_from_py(starts, "starts")?;
        let ends = ints_from_py(ends, "ends")?;
        let inner = &self.inner;
        let out = py
            .detach(|| inner.slice_agg(&keys, &starts, &ends, agg))
            .or_raise()?;
        Ok(array_to_py(out))
    }

    /// Shifts every Rle; see `Rle.shift`.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
    ///     >>> batch.shift(1)
    ///     RleBatch(keys=2, runs=4, dtype=Int64)
    ///       ('chr1',): length=3, runs=[1, 2], values=[0, 1]
    ///       ('chr2',): length=7, runs=[6, 1], values=[0, 1]
    #[pyo3(signature = (distance, fill=FillArg::Default))]
    fn shift(&self, py: Python<'_>, distance: i64, fill: FillArg) -> PyResult<Self> {
        let fill = fill_array(py, fill, self.inner.values().data_type())?;
        let inner = &self.inner;
        Ok(py
            .detach(|| inner.shift(distance, fill.as_ref()))
            .or_raise()?
            .into())
    }

    /// Reverses every Rle.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
    ///     >>> batch.reverse()
    ///     RleBatch(keys=2, runs=3, dtype=Int64)
    ///       ('chr1',): length=3, runs=[3], values=[1]
    ///       ('chr2',): length=7, runs=[2, 5], values=[1, 0]
    fn reverse(&self) -> PyResult<Self> {
        Ok(self.inner.reversed().or_raise()?.into())
    }

    /// Sum per key, as a table of the key columns plus `sum`.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
    ///     >>> pl.DataFrame(batch.sum())
    ///     shape: (2, 2)
    ///     ┌───────┬─────┐
    ///     │ chrom ┆ sum │
    ///     │ ---   ┆ --- │
    ///     │ str   ┆ i64 │
    ///     ╞═══════╪═════╡
    ///     │ chr1  ┆ 5   │
    ///     │ chr2  ┆ 2   │
    ///     └───────┴─────┘
    fn sum(&self) -> PyResult<PyTable> {
        self.aggregate("sum", |r| r.sum())
    }

    /// Mean per key, as a table of the key columns plus `mean`.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
    ///     >>> pl.DataFrame(batch.mean())
    ///     shape: (2, 2)
    ///     ┌───────┬──────────┐
    ///     │ chrom ┆ mean     │
    ///     │ ---   ┆ ---      │
    ///     │ str   ┆ f64      │
    ///     ╞═══════╪══════════╡
    ///     │ chr1  ┆ 1.25     │
    ///     │ chr2  ┆ 0.285714 │
    ///     └───────┴──────────┘
    fn mean(&self) -> PyResult<PyTable> {
        self.float_aggregate("mean", |r| r.mean())
    }

    /// Variance per key, as a table of the key columns plus `var`.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
    ///     >>> pl.DataFrame(batch.var())
    ///     shape: (2, 2)
    ///     ┌───────┬──────────┐
    ///     │ chrom ┆ var      │
    ///     │ ---   ┆ ---      │
    ///     │ str   ┆ f64      │
    ///     ╞═══════╪══════════╡
    ///     │ chr1  ┆ 0.25     │
    ///     │ chr2  ┆ 0.238095 │
    ///     └───────┴──────────┘
    #[pyo3(signature = (ddof=1))]
    fn var(&self, ddof: i64) -> PyResult<PyTable> {
        self.float_aggregate("var", |r| r.variance(ddof))
    }

    /// Standard deviation per key, as a table of the key columns plus `std`.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
    ///     >>> pl.DataFrame(batch.std(ddof=0))
    ///     shape: (2, 2)
    ///     ┌───────┬──────────┐
    ///     │ chrom ┆ std      │
    ///     │ ---   ┆ ---      │
    ///     │ str   ┆ f64      │
    ///     ╞═══════╪══════════╡
    ///     │ chr1  ┆ 0.433013 │
    ///     │ chr2  ┆ 0.451754 │
    ///     └───────┴──────────┘
    #[pyo3(signature = (ddof=1))]
    fn std(&self, ddof: i64) -> PyResult<PyTable> {
        self.float_aggregate("std", |r| r.std(ddof))
    }

    /// Minimum per key, as a table of the key columns plus `min`.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
    ///     >>> pl.DataFrame(batch.min())
    ///     shape: (2, 2)
    ///     ┌───────┬─────┐
    ///     │ chrom ┆ min │
    ///     │ ---   ┆ --- │
    ///     │ str   ┆ i64 │
    ///     ╞═══════╪═════╡
    ///     │ chr1  ┆ 1   │
    ///     │ chr2  ┆ 0   │
    ///     └───────┴─────┘
    fn min(&self) -> PyResult<PyTable> {
        self.aggregate("min", |r| r.min())
    }

    /// Maximum per key, as a table of the key columns plus `max`.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
    ///     >>> pl.DataFrame(batch.max())
    ///     shape: (2, 2)
    ///     ┌───────┬─────┐
    ///     │ chrom ┆ max │
    ///     │ ---   ┆ --- │
    ///     │ str   ┆ i64 │
    ///     ╞═══════╪═════╡
    ///     │ chr1  ┆ 2   │
    ///     │ chr2  ┆ 1   │
    ///     └───────┴─────┘
    fn max(&self) -> PyResult<PyTable> {
        self.aggregate("max", |r| r.max())
    }

    /// The runs of all keys as a pyarrow Table: the key columns, then start,
    /// end and value.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
    ///     >>> batch.to_arrow()
    ///     pyarrow.Table
    ///     chrom: string not null
    ///     start: int64 not null
    ///     end: int64 not null
    ///     value: int64
    ///     ----
    ///     chrom: [["chr1","chr2","chr2"]]
    ///     start: [[0,0,5]]
    ///     end: [[3,5,7]]
    ///     value: [[1,0,1]]
    fn to_arrow<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_arrow(slf.as_any())
    }

    /// The runs of all keys as a polars DataFrame: the key columns, then
    /// start, end and value.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
    ///     >>> batch.to_polars()
    ///     shape: (3, 4)
    ///     ┌───────┬───────┬─────┬───────┐
    ///     │ chrom ┆ start ┆ end ┆ value │
    ///     │ ---   ┆ ---   ┆ --- ┆ ---   │
    ///     │ str   ┆ i64   ┆ i64 ┆ i64   │
    ///     ╞═══════╪═══════╪═════╪═══════╡
    ///     │ chr1  ┆ 0     ┆ 3   ┆ 1     │
    ///     │ chr2  ┆ 0     ┆ 5   ┆ 0     │
    ///     │ chr2  ┆ 5     ┆ 7   ┆ 1     │
    ///     └───────┴───────┴─────┴───────┘
    fn to_polars<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_polars(slf.as_any())
    }

    /// The runs of all keys as a pandas DataFrame: the key columns, then
    /// start, end and value.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
    ///     >>> batch.to_pandas()
    ///       chrom  start  end  value
    ///     0  chr1      0    3      1
    ///     1  chr2      0    5      0
    ///     2  chr2      5    7      1
    fn to_pandas<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_pandas(slf.as_any(), || {
            slf.get().inner.to_record_batch().or_raise()
        })
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

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Add, false, false)
    }
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Add, true, false)
    }
    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Sub, false, false)
    }
    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Sub, true, false)
    }
    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Mul, false, false)
    }
    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Mul, true, false)
    }
    /// True division, as in Python: integers are divided as floats.
    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Div, false, true)
    }
    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Div, true, true)
    }
    fn __and__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::And, false, false)
    }
    fn __rand__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::And, true, false)
    }
    fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Or, false, false)
    }
    fn __ror__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(py, other, Kernel::Or, true, false)
    }

    /// Element-wise comparison with another batch (per key) or a single
    /// value, returning a batch of boolean Rles.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Self> {
        self.binary(py, other, kernel_for(op), false, false)
    }

    /// Applies `func` to the run values of every Rle and merges each result;
    /// see `Rle.map`. Keys are processed in parallel.
    ///
    /// Examples:
    ///     >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
    ///     >>> batch.map(lambda v: v > 0, backend="numpy")
    ///     RleBatch(keys=2, runs=3, dtype=Boolean)
    ///       ('chr1',): length=3, runs=[3], values=[true]
    ///       ('chr2',): length=7, runs=[5, 2], values=[false, true]
    #[pyo3(signature = (func, backend="arrow"))]
    fn map(&self, py: Python<'_>, func: &Bound<'_, PyAny>, backend: &str) -> PyResult<Self> {
        let backend = parse_backend(backend)?;
        let func: Py<PyAny> = func.clone().unbind();
        let inner = &self.inner;
        // As in zip_with: release the GIL, take it per callback.
        let out = py.detach(|| {
            inner.map_rles(|r| {
                r.map(|values| Python::attach(|py| call_python(func.bind(py), backend, &[values])))
            })
        });
        Ok(out.or_raise()?.into())
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
            let runs = Int64Array::from(rle.runs());
            lines.push(format!(
                "  {}: length={}, runs={}, values={}",
                self.key_tuple(py, i)?.repr()?,
                rle.len(),
                format_values(&runs, 6),
                format_values(rle.values().as_ref(), 6)
            ));
        }
        if n > 5 {
            lines.push(format!("  ... and {} more", n - 5));
        }
        Ok(lines.join("\n"))
    }
}
