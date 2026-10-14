//! Aggregating many slices of sequences without copying them.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, Float64Array, Int64Array, PrimitiveArray, RecordBatch,
};
use arrow_buffer::ScalarBuffer;
use arrow_cast::cast;
use arrow_schema::DataType;
use rayon::prelude::*;
use runlength::Num;

use crate::batch::ArrowRleBatch;
use crate::error::Result;
use crate::numeric::with_numeric_type;
use crate::rle::ArrowRle;

/// How [`ArrowRle::slice_agg`] reduces each slice. Null values are skipped,
/// as in polars and pandas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceAgg {
    /// The sum over all positions, in the values' type; 0 for an empty slice.
    Sum,
    /// The number of positions with a non-null value (`Int64`).
    Count,
    /// The mean over non-null positions (`Float64`), null if there are none.
    Mean,
    /// The variance with `ddof` delta degrees of freedom (`Float64`), null if
    /// there are `ddof` or fewer non-null positions.
    Var { ddof: i64 },
    /// The square root of [`SliceAgg::Var`].
    Std { ddof: i64 },
    /// The smallest value ignoring `NaN`, in the values' type; null if none.
    Min,
    /// The largest value ignoring `NaN`, in the values' type; null if none.
    Max,
}

/// Minimum slices per rayon task; reducing one slice is cheap.
const SLICES_PER_TASK: usize = 1024;

fn check_slices(starts: &[i64], ends: &[i64]) -> Result<()> {
    if starts.len() != ends.len() {
        return Err(runlength::Error::LengthMismatch {
            what: "starts and ends",
            left: starts.len(),
            right: ends.len(),
        }
        .into());
    }
    for (index, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        if start < 0 || end < start {
            return Err(runlength::Error::InvalidInterval { index, start, end }.into());
        }
    }
    Ok(())
}

/// The runs of many sequences stored back to back, with the slices to
/// reduce: slice `k` covers `starts[k]..stops[k]` of the sequence at runs
/// `range(k)`, or nothing if `range(k)` is `None`.
struct Slices<'a, T: ArrowPrimitiveType, R> {
    ends: &'a [i64],
    values: &'a PrimitiveArray<T>,
    starts: &'a [i64],
    stops: &'a [i64],
    range: R,
}

impl<T, R> Slices<'_, T, R>
where
    T: ArrowPrimitiveType,
    T::Native: Num,
    R: Fn(usize) -> Option<(usize, usize)> + Sync,
{
    /// `(length, value)` for each run slice `k` touches; `None` for nulls.
    fn runs(&self, k: usize) -> impl Iterator<Item = (i64, Option<T::Native>)> + Clone + '_ {
        let (lo, hi) = (self.range)(k).unwrap_or((0, 0));
        let ends = &self.ends[lo..hi];
        let len = ends.last().copied().unwrap_or(0);
        let (start, stop) = (self.starts[k].min(len), self.stops[k].min(len));
        let (first, last) = if start < stop {
            (
                ends.partition_point(|&e| e <= start),
                ends.partition_point(|&e| e < stop) + 1,
            )
        } else {
            (0, 0)
        };
        (first..last).map(move |i| {
            let run_start = if i == 0 { 0 } else { ends[i - 1] }.max(start);
            let run_end = ends[i].min(stop);
            let value = self
                .values
                .is_valid(lo + i)
                .then(|| self.values.value(lo + i));
            (run_end - run_start, value)
        })
    }

    fn map<X: Send>(&self, f: impl Fn(usize) -> X + Send + Sync) -> Vec<X>
    where
        Self: Sync,
    {
        (0..self.starts.len())
            .into_par_iter()
            .with_min_len(SLICES_PER_TASK)
            .map(f)
            .collect()
    }

    fn count(&self, k: usize) -> i64 {
        self.runs(k).filter_map(|(w, v)| v.map(|_| w)).sum::<i64>()
    }

    fn mean(&self, k: usize) -> Option<f64> {
        let (n, total) = self.runs(k).fold((0, 0.0), |(n, total), (w, v)| match v {
            Some(v) => (n + w, total + w as f64 * v.to_f64()),
            None => (n, total),
        });
        (n > 0).then(|| total / n as f64)
    }

    fn variance(&self, k: usize, ddof: i64) -> Option<f64> {
        let n = self.count(k);
        if n - ddof <= 0 {
            return None;
        }
        let mean = self.mean(k)?;
        let squares: f64 = self
            .runs(k)
            .filter_map(|(w, v)| v.map(|v| w as f64 * (v.to_f64() - mean).powi(2)))
            .sum();
        Some(squares / (n - ddof) as f64)
    }

    fn extreme(&self, k: usize, max: bool) -> Option<T::Native> {
        self.runs(k)
            .filter_map(|(_, v)| v.filter(|v| v.partial_cmp(v).is_some()))
            .fold(None, |best, v| match best {
                Some(b) if (max && v <= b) || (!max && v >= b) => Some(b),
                _ => Some(v),
            })
    }

    fn aggregate(&self, agg: SliceAgg) -> ArrayRef
    where
        Self: Sync,
    {
        match agg {
            SliceAgg::Sum => {
                let sums = self.map(|k| {
                    self.runs(k)
                        .filter_map(|(w, v)| v.map(|v| (w, v)))
                        .fold(T::Native::default(), |acc, (w, v)| {
                            acc + v * T::Native::from_i64(w)
                        })
                });
                Arc::new(PrimitiveArray::<T>::new(ScalarBuffer::from(sums), None))
            }
            SliceAgg::Count => Arc::new(Int64Array::from(self.map(|k| self.count(k)))),
            SliceAgg::Mean => Arc::new(Float64Array::from(self.map(|k| self.mean(k)))),
            SliceAgg::Var { ddof } => {
                Arc::new(Float64Array::from(self.map(|k| self.variance(k, ddof))))
            }
            SliceAgg::Std { ddof } => Arc::new(Float64Array::from(
                self.map(|k| self.variance(k, ddof).map(f64::sqrt)),
            )),
            SliceAgg::Min => Arc::new(
                self.map(|k| self.extreme(k, false))
                    .into_iter()
                    .collect::<PrimitiveArray<T>>(),
            ),
            SliceAgg::Max => Arc::new(
                self.map(|k| self.extreme(k, true))
                    .into_iter()
                    .collect::<PrimitiveArray<T>>(),
            ),
        }
    }
}

/// Reduces the slices over `values`, which must be numeric or boolean
/// (booleans count as 0 and 1).
fn aggregate(
    ends: &[i64],
    values: &ArrayRef,
    starts: &[i64],
    stops: &[i64],
    range: impl Fn(usize) -> Option<(usize, usize)> + Sync,
    agg: SliceAgg,
) -> Result<ArrayRef> {
    check_slices(starts, stops)?;
    if values.data_type() == &DataType::Boolean {
        let ints = cast(values, &DataType::Int64)?;
        let out = aggregate(ends, &ints, starts, stops, range, agg)?;
        return match agg {
            SliceAgg::Min | SliceAgg::Max => Ok(cast(&out, &DataType::Boolean)?),
            _ => Ok(out),
        };
    }
    with_numeric_type!(values, |T| {
        let slices = Slices {
            ends,
            values: values
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .expect("the type was matched"),
            starts,
            stops,
            range,
        };
        Ok(slices.aggregate(agg))
    })
}

impl ArrowRle {
    /// Reduces many slices without copying them, in parallel: entry `k` of
    /// the result reduces positions `starts[k]..ends[k]` (clamped to the
    /// sequence) with `agg`. Values must be numeric or boolean.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_array::{Array, Float64Array, Int64Array};
    /// use runlength_arrow::{ArrowRle, SliceAgg};
    ///
    /// let rle = ArrowRle::from_runs(&[3, 2], Arc::new(Int64Array::from(vec![1, 4]))).unwrap();
    /// let means = rle.slice_agg(&[0, 2], &[2, 5], SliceAgg::Mean).unwrap();
    /// let means = means.as_any().downcast_ref::<Float64Array>().unwrap();
    /// assert_eq!(means.values().to_vec(), vec![1.0, 3.0]);
    /// ```
    pub fn slice_agg(&self, starts: &[i64], ends: &[i64], agg: SliceAgg) -> Result<ArrayRef> {
        let n = self.num_runs();
        aggregate(
            self.ends(),
            self.values(),
            starts,
            ends,
            |_| Some((0, n)),
            agg,
        )
    }
}

impl ArrowRleBatch {
    /// Like [`ArrowRle::slice_agg`], where slice `k` is in the sequence of row
    /// `k` of `keys`. A key not in the batch gives an empty slice. `keys` must
    /// have the same column names and types as the batch's keys.
    pub fn slice_agg(
        &self,
        keys: &RecordBatch,
        starts: &[i64],
        ends: &[i64],
        agg: SliceAgg,
    ) -> Result<ArrayRef> {
        if keys.num_rows() != starts.len() {
            return Err(runlength::Error::LengthMismatch {
                what: "keys and starts",
                left: keys.num_rows(),
                right: starts.len(),
            }
            .into());
        }
        let positions = self.positions(keys)?;
        let offsets = self.offsets();
        aggregate(
            self.ends(),
            self.values(),
            starts,
            ends,
            |k| positions[k].map(|p| (offsets[p], offsets[p + 1])),
            agg,
        )
    }
}
