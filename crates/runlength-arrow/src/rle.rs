use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BooleanArray, Int64Array, PrimitiveArray, UInt64Array,
    new_empty_array,
};
use arrow_cast::cast;
use arrow_schema::{ArrowError, DataType};
use arrow_select::concat::concat;
use arrow_select::filter::filter;
use arrow_select::take::take;
use rayon::prelude::*;
use runlength::{Keep, Plan, Rle, RleLike, merge_plan, plan};

use crate::batch::ArrowRleBatch;
use crate::error::{Error, Result};
use crate::identical::identical_neighbours;
use crate::kernels::Kernel;

/// A run-length encoded sequence whose values are an Arrow array of any type.
///
/// Like [`runlength::Rle`], it stores cumulative run ends and one value per
/// run, starts at position 0, and has no zero-length runs. Values may contain
/// nulls; a null is a value like any other and merges with neighbouring nulls.
#[derive(Clone, Debug)]
pub struct ArrowRle {
    pub(crate) ends: Vec<i64>,
    pub(crate) values: ArrayRef,
}

fn validate_ends(ends: &[i64]) -> Result<()> {
    let mut prev = 0;
    for (index, &end) in ends.iter().enumerate() {
        if end <= prev {
            return Err(runlength::Error::InvalidEnds { index }.into());
        }
        prev = end;
    }
    Ok(())
}

fn check_same_len(what: &'static str, left: usize, right: usize) -> Result<()> {
    if left != right {
        return Err(runlength::Error::LengthMismatch { what, left, right }.into());
    }
    Ok(())
}

/// The fill value operators use for a value type: zero for numbers, `false`
/// for booleans, and `None` for other types (which then require equal lengths).
pub fn default_fill(data_type: &DataType) -> Option<ArrayRef> {
    if data_type == &DataType::Boolean {
        return Some(Arc::new(BooleanArray::from(vec![false])));
    }
    if data_type.is_numeric() {
        return cast(&Int64Array::from(vec![0]), data_type).ok();
    }
    None
}

/// Gathers `values[idx]`, where an index equal to `values.len()` means `fill`.
pub(crate) fn gather(
    values: &ArrayRef,
    idx: &[usize],
    fill: Option<&dyn Array>,
) -> Result<ArrayRef> {
    let n = values.len();
    if idx.len() == n && idx.iter().enumerate().all(|(k, &i)| k == i) {
        return Ok(values.clone());
    }
    // A plan only needs the fill value at its tail.
    let source = if idx.last() == Some(&n) {
        let fill = fill.expect("the length check guarantees a fill value past the end");
        concat(&[values.as_ref(), fill])?
    } else {
        values.clone()
    };
    let indices = UInt64Array::from_iter_values(idx.iter().map(|&i| i as u64));
    Ok(take(source.as_ref(), &indices, None)?)
}

/// An owned reference to a borrowed array.
pub(crate) fn array_ref(array: &dyn Array) -> ArrayRef {
    arrow_array::make_array(array.to_data())
}

pub(crate) fn check_fill(fill: Option<&dyn Array>, data_type: &DataType) -> Result<()> {
    if let Some(fill) = fill {
        if fill.len() != 1 || fill.data_type() != data_type {
            return Err(Error::InvalidFill {
                expected: data_type.clone(),
                actual: fill.data_type().clone(),
                len: fill.len(),
            });
        }
    }
    Ok(())
}

impl ArrowRle {
    /// Builds a sequence from cumulative run ends and one value per run.
    pub fn try_new(ends: Vec<i64>, values: ArrayRef) -> Result<Self> {
        check_same_len("ends and values", ends.len(), values.len())?;
        validate_ends(&ends)?;
        Ok(ArrowRle { ends, values })
    }

    /// Builds a sequence from run lengths and one value per run. Zero-length
    /// runs are dropped; neighbouring identical values are kept as they are.
    pub fn from_runs(runs: &[i64], values: ArrayRef) -> Result<Self> {
        check_same_len("runs and values", runs.len(), values.len())?;
        let mut ends = Vec::with_capacity(runs.len());
        let mut total: i64 = 0;
        for (index, &run) in runs.iter().enumerate() {
            if run < 0 {
                return Err(runlength::Error::NegativeRun { index, run }.into());
            }
            if run > 0 {
                total = total.checked_add(run).ok_or(runlength::Error::Overflow)?;
                ends.push(total);
            }
        }
        let values = if ends.len() < runs.len() {
            let keep = BooleanArray::from_iter(runs.iter().map(|&r| Some(r > 0)));
            filter(values.as_ref(), &keep)?
        } else {
            values
        };
        Ok(ArrowRle { ends, values })
    }

    /// Encodes a dense array, merging identical neighbours.
    pub fn from_dense(values: ArrayRef) -> Result<Self> {
        let ends = (1..=values.len() as i64).collect();
        ArrowRle { ends, values }.merge()
    }

    /// An empty sequence of the given value type.
    pub fn empty(data_type: &DataType) -> Self {
        ArrowRle {
            ends: Vec::new(),
            values: new_empty_array(data_type),
        }
    }

    /// Converts a core sequence of primitive values.
    pub fn from_rle<T: ArrowPrimitiveType>(rle: &Rle<T::Native>) -> Self {
        ArrowRle {
            ends: rle.ends().to_vec(),
            values: Arc::new(PrimitiveArray::<T>::from_iter_values(
                rle.values().iter().copied(),
            )),
        }
    }

    /// Converts to a core sequence of primitive values. The values must be of
    /// type `T` and contain no nulls.
    pub fn to_rle<T: ArrowPrimitiveType>(&self) -> Result<Rle<T::Native>> {
        let values = self
            .values
            .as_primitive_opt::<T>()
            .ok_or_else(|| Error::UnsupportedType(self.values.data_type().clone()))?;
        if values.null_count() > 0 {
            return Err(Error::UnexpectedNulls);
        }
        Ok(Rle::from_ends(self.ends.clone(), values.values().to_vec())?)
    }

    /// The cumulative end of each run.
    pub fn ends(&self) -> &[i64] {
        &self.ends
    }

    /// One value per run.
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    /// The Arrow type of the values.
    pub fn data_type(&self) -> &DataType {
        self.values.data_type()
    }

    /// The number of positions the sequence covers.
    pub fn len(&self) -> i64 {
        self.ends.last().copied().unwrap_or(0)
    }

    /// Whether the sequence has no positions.
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// The number of runs.
    pub fn num_runs(&self) -> usize {
        self.ends.len()
    }

    /// The length of each run.
    pub fn runs(&self) -> Vec<i64> {
        let mut prev = 0;
        self.ends
            .iter()
            .map(|&end| {
                let run = end - prev;
                prev = end;
                run
            })
            .collect()
    }

    /// The run ends and values, without copying.
    pub fn into_parts(self) -> (Vec<i64>, ArrayRef) {
        (self.ends, self.values)
    }

    /// The index of the run containing `pos`, if `pos` is in range.
    pub fn run_index(&self, pos: i64) -> Option<usize> {
        (0..self.len())
            .contains(&pos)
            .then(|| self.ends.partition_point(|&end| end <= pos))
    }

    /// The positions `start..end` as a new sequence starting at 0, clamped to
    /// the sequence. The values are a zero-copy slice.
    pub fn slice(&self, start: i64, end: i64) -> ArrowRle {
        let (start, end) = (start.max(0), end.min(self.len()));
        if start >= end {
            return ArrowRle::empty(self.data_type());
        }
        let first = self.ends.partition_point(|&e| e <= start);
        let last = self.ends.partition_point(|&e| e < end);
        let mut ends: Vec<i64> = self.ends[first..last].iter().map(|&e| e - start).collect();
        ends.push(end - start);
        ArrowRle {
            ends,
            values: self.values.slice(first, last - first + 1),
        }
    }

    /// Many slices at once, as a batch with one key column, `slice`, holding
    /// the slice number. Each interval needs `0 <= start <= end`.
    pub fn getitems(&self, starts: &[i64], ends: &[i64]) -> Result<ArrowRleBatch> {
        check_same_len("starts and ends", starts.len(), ends.len())?;
        for (index, (&start, &end)) in starts.iter().zip(ends).enumerate() {
            if start < 0 || end < start {
                return Err(runlength::Error::InvalidInterval { index, start, end }.into());
            }
        }
        // The binary searches dominate, so find each slice's runs in parallel
        // first, then copy in order.
        let len = self.len();
        let runs: Vec<Option<(usize, usize)>> = (0..starts.len())
            .into_par_iter()
            .with_min_len(1024)
            .map(|k| {
                let end = ends[k].min(len);
                (starts[k] < end).then(|| {
                    let first = self.ends.partition_point(|&e| e <= starts[k]);
                    let last = self.ends.partition_point(|&e| e < end);
                    (first, last)
                })
            })
            .collect();
        let total: usize = runs
            .iter()
            .flatten()
            .map(|(first, last)| last - first + 1)
            .sum();
        let mut offsets = Vec::with_capacity(starts.len() + 1);
        offsets.push(0);
        let mut out_ends = Vec::with_capacity(total);
        let mut run_ids = Vec::with_capacity(total);
        for (k, slice_runs) in runs.into_iter().enumerate() {
            if let Some((first, last)) = slice_runs {
                let (start, end) = (starts[k], ends[k].min(len));
                out_ends.extend(self.ends[first..last].iter().map(|&e| e - start));
                out_ends.push(end - start);
                run_ids.extend(first as u64..=last as u64);
            }
            offsets.push(out_ends.len());
        }
        let values = take(self.values.as_ref(), &UInt64Array::from(run_ids), None)?;
        let keys = UInt64Array::from_iter_values(0..starts.len() as u64);
        ArrowRleBatch::from_parts_unchecked(
            crate::batch::single_key_column("slice", Arc::new(keys)),
            offsets,
            out_ends,
            values,
        )
    }

    /// Expands the sequence into one value per position.
    pub fn to_dense(&self) -> Result<ArrayRef> {
        let mut indices = Vec::with_capacity(self.len() as usize);
        for (i, run) in self.runs().into_iter().enumerate() {
            indices.extend(std::iter::repeat_n(i as u64, run as usize));
        }
        Ok(take(
            self.values.as_ref(),
            &UInt64Array::from(indices),
            None,
        )?)
    }

    /// The plan for combining this sequence with `other`. See [`Plan`].
    pub fn plan_with(&self, other: &ArrowRle) -> Plan {
        plan(&self.ends, &other.ends)
    }

    /// For each pair of neighbouring runs, whether their values are identical.
    pub fn identical_neighbours(&self) -> Result<Vec<bool>> {
        identical_neighbours(self.values.as_ref())
    }

    /// Whether no two neighbouring runs hold identical values.
    pub fn is_merged(&self) -> Result<bool> {
        Ok(!self.identical_neighbours()?.into_iter().any(|x| x))
    }

    /// Merges neighbouring runs with identical values.
    pub fn merge(self) -> Result<Self> {
        let mask = self.identical_neighbours()?;
        self.merge_mask(&mask, Keep::First)
    }

    /// Merges run `i + 1` into run `i` wherever `mask[i]` is true, keeping the
    /// first or last value of each merged group.
    pub fn merge_mask(self, mask: &[bool], keep: Keep) -> Result<Self> {
        let groups = merge_plan(&self.ends, mask)?;
        if groups.num_groups() == self.ends.len() {
            return Ok(self);
        }
        let kept = groups.offsets.windows(2).map(|w| match keep {
            Keep::First => w[0] as u64,
            Keep::Last => w[1] as u64 - 1,
        });
        let values = take(
            self.values.as_ref(),
            &UInt64Array::from_iter_values(kept),
            None,
        )?;
        Ok(ArrowRle {
            ends: groups.ends,
            values,
        })
    }

    /// Combines two sequences position by position and merges the result.
    ///
    /// `f` receives the values of both sides gathered onto the plan's
    /// segments (two arrays of equal length) and must return one value per
    /// segment; any Arrow kernel fits. The shorter side is padded with its
    /// fill value, a single value of the same type; with no fill value,
    /// unequal lengths are an error.
    pub fn zip_with<F>(
        &self,
        other: &ArrowRle,
        fill_a: Option<&dyn Array>,
        fill_b: Option<&dyn Array>,
        f: F,
    ) -> Result<ArrowRle>
    where
        F: FnOnce(&dyn Array, &dyn Array) -> std::result::Result<ArrayRef, ArrowError>,
    {
        self.zip_with_raw(other, fill_a, fill_b, f)?.merge()
    }

    /// Like [`zip_with`](ArrowRle::zip_with) but without merging the result.
    pub fn zip_with_raw<F>(
        &self,
        other: &ArrowRle,
        fill_a: Option<&dyn Array>,
        fill_b: Option<&dyn Array>,
        f: F,
    ) -> Result<ArrowRle>
    where
        F: FnOnce(&dyn Array, &dyn Array) -> std::result::Result<ArrayRef, ArrowError>,
    {
        check_fill(fill_a, self.data_type())?;
        check_fill(fill_b, other.data_type())?;
        let (len_a, len_b) = (self.len(), other.len());
        if (len_a < len_b && fill_a.is_none()) || (len_b < len_a && fill_b.is_none()) {
            return Err(runlength::Error::UnequalLength {
                left: len_a,
                right: len_b,
            }
            .into());
        }
        let p = self.plan_with(other);
        let a = gather(&self.values, &p.idx_a, fill_a)?;
        let b = gather(&other.values, &p.idx_b, fill_b)?;
        let values = f(a.as_ref(), b.as_ref())?;
        if values.len() != p.ends.len() {
            return Err(Error::WrongLength {
                expected: p.ends.len(),
                actual: values.len(),
            });
        }
        Ok(ArrowRle {
            ends: p.ends,
            values,
        })
    }

    /// Applies an Arrow kernel to the run values and merges the result.
    pub fn map<F>(&self, f: F) -> Result<ArrowRle>
    where
        F: FnOnce(&dyn Array) -> std::result::Result<ArrayRef, ArrowError>,
    {
        let values = f(self.values.as_ref())?;
        if values.len() != self.values.len() {
            return Err(Error::WrongLength {
                expected: self.values.len(),
                actual: values.len(),
            });
        }
        ArrowRle {
            ends: self.ends.clone(),
            values,
        }
        .merge()
    }

    /// Casts the values to `data_type` and merges any neighbours that become
    /// identical (for example when floats are truncated to integers).
    pub fn cast(&self, data_type: &DataType) -> Result<ArrowRle> {
        if self.data_type() == data_type {
            return Ok(self.clone());
        }
        let values = cast(self.values.as_ref(), data_type)?;
        ArrowRle {
            ends: self.ends.clone(),
            values,
        }
        .merge()
    }

    /// Combines with `other` using [`default_fill`] on each side, and merges.
    pub fn zip_with_default(&self, other: &ArrowRle, kernel: Kernel) -> Result<ArrowRle> {
        self.op(other, kernel)
    }

    pub(crate) fn op(&self, other: &ArrowRle, kernel: Kernel) -> Result<ArrowRle> {
        if let Some(out) = crate::fused::fused(self, other, kernel) {
            return Ok(out);
        }
        let fill_a = default_fill(self.data_type());
        let fill_b = default_fill(other.data_type());
        self.zip_with(other, fill_a.as_deref(), fill_b.as_deref(), |a, b| {
            kernel.apply(a, b)
        })
    }
}

crate::kernels::kernel_methods!(
    ArrowRle,
    " The shorter side is padded with [`default_fill`]."
);

impl PartialEq for ArrowRle {
    /// Whether both sequences represent the same values at every position
    /// (by [`identical_neighbours`]'s rule), however the runs are split.
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() || self.data_type() != other.data_type() {
            return false;
        }
        let p = self.plan_with(other);
        let (Ok(a), Ok(b)) = (
            gather(&self.values, &p.idx_a, None),
            gather(&other.values, &p.idx_b, None),
        ) else {
            return false;
        };
        // Interleave a and b so each pair becomes a neighbouring pair.
        let n = a.len();
        let indices: Vec<(usize, usize)> = (0..n).flat_map(|k| [(0, k), (1, k)]).collect();
        let Ok(pairs) = arrow_select::interleave::interleave(&[a.as_ref(), b.as_ref()], &indices)
        else {
            return false;
        };
        identical_neighbours(pairs.as_ref())
            .map(|mask| mask.into_iter().step_by(2).all(|x| x))
            .unwrap_or(false)
    }
}
