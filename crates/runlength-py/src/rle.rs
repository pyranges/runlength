use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, make_array};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PySlice, PyTuple};
use pyo3_arrow::PyArray;
use runlength::Keep;
use runlength_arrow::kernels::{self, Kernel};
use runlength_arrow::{ArrowRle, SliceAgg, common_numeric_type};

use crate::batch::PyRleBatch;
use crate::convert::{
    FillArg, OrRaise, array_from_py, array_to_py, bools_from_py, indices_to_py, ints_from_py,
    scalar_from_py, scalar_of_type, table_capsule, to_arrow, to_pandas, to_polars, value_to_py,
};

/// A run-length encoded sequence of values of any Arrow type.
///
/// Rle(runs, values)
///
/// `runs` are the run lengths and `values` one value per run. Both can be
/// polars or pandas Series, pyarrow or numpy arrays, or lists. Zero-length
/// runs are dropped; neighbouring identical values are kept as given (see
/// `merge`).
#[pyclass(module = "runlength", name = "Rle", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRle {
    pub(crate) inner: ArrowRle,
}

impl From<ArrowRle> for PyRle {
    fn from(inner: ArrowRle) -> Self {
        PyRle { inner }
    }
}

pub(crate) fn parse_keep(keep: &str) -> PyResult<Keep> {
    match keep {
        "first" => Ok(Keep::First),
        "last" => Ok(Keep::Last),
        other => Err(PyValueError::new_err(format!(
            "keep must be 'first' or 'last', got {other:?}"
        ))),
    }
}

/// Converts both sides to a common numeric type if they differ.
pub(crate) fn coerce(a: &ArrowRle, b: &ArrowRle) -> PyResult<(ArrowRle, ArrowRle)> {
    match common_numeric_type(a.data_type(), b.data_type()) {
        Some(t) if a.data_type() != b.data_type() => {
            Ok((a.cast(&t).or_raise()?, b.cast(&t).or_raise()?))
        }
        _ => Ok((a.clone(), b.clone())),
    }
}

pub(crate) fn kernel_for(op: CompareOp) -> Kernel {
    match op {
        CompareOp::Lt => kernels::lt,
        CompareOp::Le => kernels::le,
        CompareOp::Eq => kernels::eq,
        CompareOp::Ne => kernels::ne,
        CompareOp::Gt => kernels::gt,
        CompareOp::Ge => kernels::ge,
    }
}

/// The library a Python callback receives its arrays from.
#[derive(Clone, Copy)]
pub(crate) enum Backend {
    Arrow,
    PyArrow,
    Polars,
    Numpy,
}

pub(crate) fn parse_backend(backend: &str) -> PyResult<Backend> {
    match backend {
        "arrow" => Ok(Backend::Arrow),
        "pyarrow" => Ok(Backend::PyArrow),
        "polars" => Ok(Backend::Polars),
        "numpy" => Ok(Backend::Numpy),
        other => Err(PyValueError::new_err(format!(
            "backend must be 'arrow', 'pyarrow', 'polars' or 'numpy', got {other:?}"
        ))),
    }
}

fn to_backend<'py>(
    py: Python<'py>,
    array: PyArray,
    backend: Backend,
) -> PyResult<Bound<'py, PyAny>> {
    let array = array.into_pyobject(py)?.into_any();
    match backend {
        Backend::Arrow => Ok(array),
        Backend::PyArrow => py.import("pyarrow")?.call_method1("array", (array,)),
        Backend::Polars => py.import("polars")?.call_method1("Series", (array,)),
        Backend::Numpy => py.import("numpy")?.call_method1("asarray", (array,)),
    }
}

/// Calls a Python function with Arrow arrays, for `zip_with` and `map`.
pub(crate) fn call_python(
    func: &Bound<'_, PyAny>,
    backend: Backend,
    args: &[&dyn Array],
) -> Result<ArrayRef, ArrowError> {
    let py = func.py();
    let call = || -> PyResult<ArrayRef> {
        let converted = args
            .iter()
            .map(|a| to_backend(py, array_to_py(make_array(a.to_data())), backend))
            .collect::<PyResult<Vec<_>>>()?;
        let result = func.call1(PyTuple::new(py, converted)?)?;
        array_from_py(&result)
    };
    call().map_err(|e| ArrowError::ExternalError(Box::new(e)))
}

pub(crate) fn parse_slice_agg(agg: &str, ddof: i64) -> PyResult<SliceAgg> {
    match agg {
        "sum" => Ok(SliceAgg::Sum),
        "count" => Ok(SliceAgg::Count),
        "mean" => Ok(SliceAgg::Mean),
        "var" => Ok(SliceAgg::Var { ddof }),
        "std" => Ok(SliceAgg::Std { ddof }),
        "min" => Ok(SliceAgg::Min),
        "max" => Ok(SliceAgg::Max),
        other => Err(PyValueError::new_err(format!(
            "agg must be 'sum', 'count', 'mean', 'var', 'std', 'min' or 'max', got {other:?}"
        ))),
    }
}

/// A single fill value of `data_type`: the given value, null, or the type's
/// default fill (null if it has none).
pub(crate) fn fill_array(
    py: Python<'_>,
    fill: FillArg,
    data_type: &DataType,
) -> PyResult<ArrayRef> {
    let null = || arrow_array::new_null_array(data_type, 1);
    match fill {
        FillArg::Value(value) => scalar_of_type(value.bind(py), data_type),
        FillArg::Null => Ok(null()),
        FillArg::Default => Ok(runlength_arrow::default_fill(data_type).unwrap_or_else(null)),
    }
}

pub(crate) fn fill_of(
    fill: Option<&Bound<'_, PyAny>>,
    data_type: &DataType,
) -> PyResult<Option<ArrayRef>> {
    fill.map(|f| scalar_of_type(f, data_type)).transpose()
}

pub(crate) fn format_values(values: &dyn Array, limit: usize) -> String {
    let Ok(formatter) =
        ArrayFormatter::try_new(values, &FormatOptions::default().with_null("null"))
    else {
        return "[...]".to_string();
    };
    let n = values.len();
    let shown: Vec<String> = if n <= limit {
        (0..n).map(|i| formatter.value(i).to_string()).collect()
    } else {
        let head = (0..limit / 2).map(|i| formatter.value(i).to_string());
        let tail = (n - limit / 2..n).map(|i| formatter.value(i).to_string());
        head.chain(std::iter::once("...".to_string()))
            .chain(tail)
            .collect()
    };
    format!("[{}]", shown.join(", "))
}

impl PyRle {
    /// Turns the right operand of an operator into a sequence: another `Rle`,
    /// or a single value repeated over this sequence's length.
    fn operand(&self, other: &Bound<'_, PyAny>) -> PyResult<ArrowRle> {
        if let Ok(rle) = other.extract::<PyRef<PyRle>>() {
            return Ok(rle.inner.clone());
        }
        let value = scalar_from_py(other).map_err(|_| {
            PyTypeError::new_err(format!(
                "unsupported operand type: Rle and {}",
                other
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            ))
        })?;
        ArrowRle::from_runs(&[self.inner.len()], value).or_raise()
    }

    fn binary(&self, other: &Bound<'_, PyAny>, kernel: Kernel, reflected: bool) -> PyResult<PyRle> {
        self.binary_as(other, kernel, reflected, false)
    }

    /// With `to_float`, integer operands become `Float64` first (for `/`).
    fn binary_as(
        &self,
        other: &Bound<'_, PyAny>,
        kernel: Kernel,
        reflected: bool,
        to_float: bool,
    ) -> PyResult<PyRle> {
        let other = self.operand(other)?;
        let (a, b) = if reflected {
            (other, self.inner.clone())
        } else {
            (self.inner.clone(), other)
        };
        let (a, b) = coerce(&a, &b)?;
        let (a, b) = if to_float && a.data_type().is_integer() {
            (
                a.cast(&DataType::Float64).or_raise()?,
                b.cast(&DataType::Float64).or_raise()?,
            )
        } else {
            (a, b)
        };
        Ok(a.zip_with_default(&b, kernel).or_raise()?.into())
    }

    pub(crate) fn run_table(&self) -> PyResult<RecordBatch> {
        let rle = &self.inner;
        let mut starts = Vec::with_capacity(rle.num_runs());
        let mut prev = 0;
        for &end in rle.ends() {
            starts.push(prev);
            prev = end;
        }
        let schema = Schema::new(vec![
            Field::new("start", DataType::Int64, false),
            Field::new("end", DataType::Int64, false),
            Field::new("value", rle.data_type().clone(), true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(starts)),
                Arc::new(Int64Array::from(rle.ends().to_vec())),
                rle.values().clone(),
            ],
        )
        .or_raise()
    }
}

#[pymethods]
impl PyRle {
    #[new]
    fn new(runs: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<Self> {
        let runs = ints_from_py(runs, "runs")?;
        let values = array_from_py(values)?;
        Ok(ArrowRle::from_runs(&runs, values).or_raise()?.into())
    }

    /// Encodes one value per position, merging identical neighbours.
    #[staticmethod]
    fn from_dense(values: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(ArrowRle::from_dense(array_from_py(values)?)
            .or_raise()?
            .into())
    }

    /// Builds a sequence from cumulative run ends and one value per run.
    #[staticmethod]
    fn from_ends(ends: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ends = ints_from_py(ends, "ends")?;
        Ok(ArrowRle::try_new(ends, array_from_py(values)?)
            .or_raise()?
            .into())
    }

    /// The length of each run.
    #[getter]
    fn runs(&self) -> PyArray {
        array_to_py(Arc::new(Int64Array::from(self.inner.runs())))
    }

    /// The cumulative end of each run.
    #[getter]
    fn ends(&self) -> PyArray {
        array_to_py(Arc::new(Int64Array::from(self.inner.ends().to_vec())))
    }

    /// The start of each run.
    #[getter]
    fn starts(&self) -> PyArray {
        let mut starts = Vec::with_capacity(self.inner.num_runs());
        let mut prev = 0;
        for &end in self.inner.ends() {
            starts.push(prev);
            prev = end;
        }
        array_to_py(Arc::new(Int64Array::from(starts)))
    }

    /// One value per run.
    #[getter]
    fn values(&self) -> PyArray {
        array_to_py(self.inner.values().clone())
    }

    /// The Arrow type of the values.
    #[getter]
    fn dtype(&self) -> String {
        self.inner.data_type().to_string()
    }

    #[getter]
    fn num_runs(&self) -> usize {
        self.inner.num_runs()
    }

    /// The number of positions.
    fn __len__(&self) -> usize {
        self.inner.len() as usize
    }

    /// `rle[i]` is the value at position `i`; `rle[a:b]` is a new sequence
    /// starting at 0. Negative indices count from the end.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let len = self.inner.len();
        if let Ok(slice) = key.cast::<PySlice>() {
            let idx = slice.indices(len as isize)?;
            if idx.step != 1 {
                return Err(PyValueError::new_err(
                    "slices with a step are not supported",
                ));
            }
            let sliced: PyRle = self.inner.slice(idx.start as i64, idx.stop as i64).into();
            return Ok(sliced.into_pyobject(py)?.into_any().unbind());
        }
        let mut pos: i64 = key.extract()?;
        if pos < 0 {
            pos += len;
        }
        let run = self.inner.run_index(pos).ok_or_else(|| {
            PyIndexError::new_err(format!("position out of range for length {len}"))
        })?;
        value_to_py(py, self.inner.values(), run)
    }

    /// Many slices at once, as an RleBatch keyed by slice number. Ends past the
    /// sequence are clamped.
    fn getitems(
        &self,
        py: Python<'_>,
        starts: &Bound<'_, PyAny>,
        ends: &Bound<'_, PyAny>,
    ) -> PyResult<PyRleBatch> {
        let starts = ints_from_py(starts, "starts")?;
        let ends = ints_from_py(ends, "ends")?;
        let inner = &self.inner;
        Ok(py
            .detach(|| inner.getitems(&starts, &ends))
            .or_raise()?
            .into())
    }

    /// Reduces many slices without copying them, in parallel: entry k reduces
    /// positions starts[k]:ends[k] (ends past the sequence are clamped).
    /// `agg` is "sum", "count", "mean", "var", "std", "min" or "max"; nulls
    /// are skipped and `ddof` applies to "var" and "std". Values must be
    /// numeric or boolean.
    #[pyo3(signature = (starts, ends, agg="mean", ddof=1))]
    fn slice_agg(
        &self,
        py: Python<'_>,
        starts: &Bound<'_, PyAny>,
        ends: &Bound<'_, PyAny>,
        agg: &str,
        ddof: i64,
    ) -> PyResult<PyArray> {
        let agg = parse_slice_agg(agg, ddof)?;
        let starts = ints_from_py(starts, "starts")?;
        let ends = ints_from_py(ends, "ends")?;
        let inner = &self.inner;
        let out = py
            .detach(|| inner.slice_agg(&starts, &ends, agg))
            .or_raise()?;
        Ok(array_to_py(out))
    }

    /// Merges neighbouring runs. Without a mask, runs merge when their values
    /// are identical (NaN with NaN, null with null). With a mask (one boolean
    /// per pair of neighbouring runs), run i + 1 merges into run i where
    /// mask[i] is true, keeping the "first" or "last" value of each group.
    #[pyo3(signature = (mask=None, keep="first"))]
    fn merge(&self, mask: Option<&Bound<'_, PyAny>>, keep: &str) -> PyResult<Self> {
        let keep = parse_keep(keep)?;
        let rle = self.inner.clone();
        let merged = match mask {
            None => rle.merge(),
            Some(mask) => rle.merge_mask(&bools_from_py(mask)?, keep),
        };
        Ok(merged.or_raise()?.into())
    }

    /// Groups runs by a mask without touching values: returns (ends, offsets),
    /// where group g is runs offsets[g]:offsets[g + 1] and ends at ends[g].
    fn merge_plan(&self, mask: &Bound<'_, PyAny>) -> PyResult<(PyArray, PyArray)> {
        let plan = runlength::merge_plan(self.inner.ends(), &bools_from_py(mask)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            array_to_py(Arc::new(Int64Array::from(plan.ends))),
            indices_to_py(plan.offsets),
        ))
    }

    /// Whether no two neighbouring runs hold identical values.
    fn is_merged(&self) -> PyResult<bool> {
        self.inner.is_merged().or_raise()
    }

    /// The plan for combining with `other`: (ends, idx_self, idx_other).
    ///
    /// Segment k ends at ends[k] and is covered by run idx_self[k] of this
    /// sequence and run idx_other[k] of `other`. An index equal to that side's
    /// number of runs means the segment lies past its end.
    fn plan(&self, other: PyRef<'_, PyRle>) -> (PyArray, PyArray, PyArray) {
        let p = self.inner.plan_with(&other.inner);
        (
            array_to_py(Arc::new(Int64Array::from(p.ends))),
            indices_to_py(p.idx_a),
            indices_to_py(p.idx_b),
        )
    }

    /// Combines with `other` position by position using `func`.
    ///
    /// `func(a, b)` receives two Arrow arrays of equal length, the values of
    /// both sides on each segment of the plan, and must return one value per
    /// segment (any Arrow-compatible array). The shorter side is padded with
    /// its fill value; without one, unequal lengths are an error.
    ///
    /// `backend` picks what `func` receives: "arrow" (objects with the Arrow
    /// PyCapsule interface), "pyarrow", "polars" or "numpy".
    #[pyo3(signature = (other, func, fill_self=None, fill_other=None, merge=true, backend="arrow"))]
    fn zip_with(
        &self,
        other: PyRef<'_, PyRle>,
        func: &Bound<'_, PyAny>,
        fill_self: Option<&Bound<'_, PyAny>>,
        fill_other: Option<&Bound<'_, PyAny>>,
        merge: bool,
        backend: &str,
    ) -> PyResult<Self> {
        let backend = parse_backend(backend)?;
        let fill_a = fill_of(fill_self, self.inner.data_type())?;
        let fill_b = fill_of(fill_other, other.inner.data_type())?;
        let f = |a: &dyn Array, b: &dyn Array| call_python(func, backend, &[a, b]);
        let (fa, fb) = (fill_a.as_deref(), fill_b.as_deref());
        let out = if merge {
            self.inner.zip_with(&other.inner, fa, fb, f)
        } else {
            self.inner.zip_with_raw(&other.inner, fa, fb, f)
        };
        Ok(out.or_raise()?.into())
    }

    /// Applies `func` to the run values and merges the result. `func` must
    /// return one value per run; `backend` is as in `zip_with`.
    #[pyo3(signature = (func, backend="arrow"))]
    fn map(&self, func: &Bound<'_, PyAny>, backend: &str) -> PyResult<Self> {
        let backend = parse_backend(backend)?;
        Ok(self
            .inner
            .map(|values| call_python(func, backend, &[values]))
            .or_raise()?
            .into())
    }

    /// Casts the values to another Arrow type, given as a string such as
    /// "float64" or "int32".
    fn cast(&self, dtype: &str) -> PyResult<Self> {
        let data_type: DataType = dtype
            .parse()
            .map_err(|e: ArrowError| PyValueError::new_err(e.to_string()))?;
        Ok(self.inner.cast(&data_type).or_raise()?.into())
    }

    /// Shifts values `distance` positions right (left if negative), keeping
    /// the length. Positions shifted in hold `fill`: by default 0 for numbers,
    /// false for booleans and null otherwise; `fill=None` means null.
    #[pyo3(signature = (distance, fill=FillArg::Default))]
    fn shift(&self, py: Python<'_>, distance: i64, fill: FillArg) -> PyResult<Self> {
        let fill = fill_array(py, fill, self.inner.data_type())?;
        Ok(self.inner.shift(distance, fill.as_ref()).or_raise()?.into())
    }

    /// The sequence back to front.
    fn reverse(&self) -> PyResult<Self> {
        Ok(self.inner.reversed().or_raise()?.into())
    }

    /// For a boolean Rle, the ranges where it is true, as (starts, ends);
    /// nulls count as false. For example `(coverage > 0).ranges()`.
    fn ranges(&self) -> PyResult<(PyArray, PyArray)> {
        let (starts, ends) = self.inner.true_ranges().or_raise()?;
        Ok((
            array_to_py(Arc::new(Int64Array::from(starts))),
            array_to_py(Arc::new(Int64Array::from(ends))),
        ))
    }

    /// Replaces nulls with `value`.
    fn fill_null(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let fill = scalar_of_type(value, self.inner.data_type())?;
        Ok(self.inner.fill_null(fill.as_ref()).or_raise()?.into())
    }

    /// Whether both sequences hold identical values at every position,
    /// however their runs are split. (`==` compares element-wise.)
    fn equals(&self, other: PyRef<'_, PyRle>) -> bool {
        self.inner == other.inner
    }

    /// The sum over all positions.
    fn sum(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_py(py, &self.inner.sum().or_raise()?, 0)
    }

    /// The mean over all positions, or None if empty.
    fn mean(&self) -> PyResult<Option<f64>> {
        self.inner.mean().or_raise()
    }

    /// The variance over all positions, with `ddof` delta degrees of freedom.
    #[pyo3(signature = (ddof=1))]
    fn var(&self, ddof: i64) -> PyResult<Option<f64>> {
        self.inner.variance(ddof).or_raise()
    }

    /// The standard deviation over all positions, with `ddof` delta degrees of
    /// freedom.
    #[pyo3(signature = (ddof=1))]
    fn std(&self, ddof: i64) -> PyResult<Option<f64>> {
        self.inner.std(ddof).or_raise()
    }

    /// The smallest value, ignoring NaN.
    fn min(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_py(py, &self.inner.min().or_raise()?, 0)
    }

    /// The largest value, ignoring NaN.
    fn max(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_py(py, &self.inner.max().or_raise()?, 0)
    }

    /// One value per position, as an Arrow array.
    fn to_dense(&self) -> PyResult<PyArray> {
        Ok(array_to_py(self.inner.to_dense().or_raise()?))
    }

    /// One value per position, as a numpy array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("numpy")?
            .call_method1("asarray", (self.to_dense()?,))
    }

    /// The runs as a pyarrow Table with columns start, end and value.
    fn to_arrow<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_arrow(slf.as_any())
    }

    /// The runs as a polars DataFrame with columns start, end and value.
    fn to_polars<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_polars(slf.as_any())
    }

    /// The runs as a pandas DataFrame with columns start, end and value.
    fn to_pandas<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_pandas(slf.as_any())
    }

    /// Exports the run table (start, end, value) as an Arrow stream.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        table_capsule(py, self.run_table()?, requested_schema)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::add, false)
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::add, true)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::sub, false)
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::sub, true)
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::mul, false)
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::mul, true)
    }
    /// True division, as in Python: integers are divided as floats.
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary_as(other, kernels::div, false, true)
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary_as(other, kernels::div, true, true)
    }
    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::and, false)
    }
    fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::and, true)
    }
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::or, false)
    }
    fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, kernels::or, true)
    }

    /// Element-wise comparison, returning an Rle of booleans.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Self> {
        self.binary(other, kernel_for(op), false)
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.inner
            .map(|values| arrow_arith::numeric::neg(values))
            .or_raise()
            .map(Into::into)
    }

    fn __invert__(&self) -> PyResult<Self> {
        self.inner
            .map(
                |values| match values.as_any().downcast_ref::<arrow_array::BooleanArray>() {
                    Some(b) => Ok(Arc::new(arrow_arith::boolean::not(b)?) as ArrayRef),
                    None => Err(ArrowError::InvalidArgumentError(format!(
                        "~ needs boolean values, got {}",
                        values.data_type()
                    ))),
                },
            )
            .or_raise()
            .map(Into::into)
    }

    /// An Rle is not true or false; use `len(rle)`, `.equals()`, or reduce a
    /// boolean Rle with `.min()` / `.max()`.
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyTypeError::new_err(
            "the truth value of an Rle is ambiguous; use len(rle), rle.equals(other), or .min()/.max() of a boolean Rle",
        ))
    }

    fn __repr__(&self) -> String {
        let rle = &self.inner;
        let runs = Int64Array::from(rle.runs());
        format!(
            "Rle(length={}, runs={}, dtype={})\n  runs:   {}\n  values: {}",
            rle.len(),
            rle.num_runs(),
            rle.data_type(),
            format_values(&runs, 10),
            format_values(rle.values().as_ref(), 10),
        )
    }
}
