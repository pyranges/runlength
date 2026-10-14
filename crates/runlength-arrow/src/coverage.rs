use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, PrimitiveArray, RecordBatch, StringArray, UInt64Array,
};
use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;
use arrow_select::take::take;
use rayon::prelude::*;
use runlength::{Rle, RleBatch};

use crate::batch::{ArrowRleBatch, key_converter, key_rows};
use crate::error::{Error, Result};
use crate::numeric::with_numeric_type;
use crate::rle::ArrowRle;

fn to_array<T: ArrowPrimitiveType>(values: Vec<T::Native>) -> ArrayRef {
    Arc::new(PrimitiveArray::<T>::new(ScalarBuffer::from(values), None))
}

fn native<T: ArrowPrimitiveType>(values: &dyn Array) -> Result<&[T::Native]> {
    if values.null_count() > 0 {
        return Err(Error::UnexpectedNulls);
    }
    Ok(values.as_primitive::<T>().values())
}

/// Coverage of half-open intervals `[start, end)`.
///
/// With `values`, each position holds the sum of the values of the intervals
/// covering it, in the values' own numeric type (which must have no nulls).
/// Without, it holds the number of covering intervals as `Int64`. Uncovered
/// positions are 0. The result covers `0..max(end)`, or `0..length`.
pub fn coverage(
    starts: &[i64],
    ends: &[i64],
    values: Option<&dyn Array>,
    length: Option<i64>,
) -> Result<ArrowRle> {
    let Some(values) = values else {
        let rle = runlength::coverage_count(starts, ends, length)?;
        return Ok(ArrowRle::from_rle::<Int64Type>(&rle));
    };
    with_numeric_type!(values, |T| {
        let rle = runlength::coverage_sum(starts, ends, native::<T>(values)?, length)?;
        Ok(ArrowRle::from_rle::<T>(&rle))
    })
}

/// How [`coverage_agg`] combines the intervals covering a position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Agg {
    /// The sum of the values (the values' numeric type).
    Sum,
    /// The number of intervals (`Int64`); values are ignored.
    Count,
    /// The largest value (numbers or strings), ignoring `NaN` where possible.
    Max,
    /// The smallest value (numbers or strings), ignoring `NaN` where possible.
    Min,
}

fn from_options<T: ArrowPrimitiveType>(rle: Rle<Option<T::Native>>) -> ArrowRle {
    let (ends, values) = rle.into_parts();
    ArrowRle::try_new(
        ends,
        Arc::new(values.into_iter().collect::<PrimitiveArray<T>>()),
    )
    .expect("a core sequence has valid ends")
}

fn strings_from_options(rle: Rle<Option<&str>>) -> ArrowRle {
    let (ends, values) = rle.into_parts();
    ArrowRle::try_new(ends, Arc::new(values.into_iter().collect::<StringArray>()))
        .expect("a core sequence has valid ends")
}

/// The core max/min coverage, for any ordered value type.
fn core_extreme<V>(
    starts: &[i64],
    ends: &[i64],
    values: &[V],
    length: Option<i64>,
    max: bool,
) -> runlength::Result<Rle<Option<V>>>
where
    V: PartialOrd + Clone + runlength::Identical,
{
    if max {
        runlength::coverage_max(starts, ends, values, length)
    } else {
        runlength::coverage_min(starts, ends, values, length)
    }
}

/// Max/min coverage over numbers or strings.
fn extreme(
    starts: &[i64],
    ends: &[i64],
    values: &dyn Array,
    length: Option<i64>,
    max: bool,
) -> Result<ArrowRle> {
    if let Some(strings) = values.as_string_opt::<i32>() {
        if strings.null_count() > 0 {
            return Err(Error::UnexpectedNulls);
        }
        let strings: Vec<&str> = strings.iter().map(|s| s.expect("no nulls")).collect();
        return Ok(strings_from_options(core_extreme(
            starts, ends, &strings, length, max,
        )?));
    }
    with_numeric_type!(values, |T| {
        Ok(from_options::<T>(core_extreme(
            starts,
            ends,
            native::<T>(values)?,
            length,
            max,
        )?))
    })
}

/// Coverage of half-open intervals `[start, end)` with any aggregation.
///
/// Unlike [`coverage`], positions no interval covers are null (for
/// [`Agg::Count`] they are 0); use [`ArrowRle::fill_null`] to fill them.
/// `values` are required except for [`Agg::Count`].
pub fn coverage_agg(
    starts: &[i64],
    ends: &[i64],
    values: Option<&dyn Array>,
    length: Option<i64>,
    agg: Agg,
) -> Result<ArrowRle> {
    let values = match (agg, values) {
        (Agg::Count, _) => return coverage(starts, ends, None, length),
        (_, Some(values)) => values,
        (_, None) => return Err(Error::MissingValues),
    };
    match agg {
        Agg::Sum => with_numeric_type!(values, |T| {
            let rle = runlength::coverage_sum_opt(starts, ends, native::<T>(values)?, length)?;
            Ok(from_options::<T>(rle))
        }),
        Agg::Max => extreme(starts, ends, values, length, true),
        Agg::Min => extreme(starts, ends, values, length, false),
        Agg::Count => unreachable!("handled above"),
    }
}

/// [`coverage_agg`] per key, computed in parallel; see [`coverage_by_key`]
/// for the keys.
pub fn coverage_agg_by_key(
    keys: &RecordBatch,
    starts: &[i64],
    ends: &[i64],
    values: Option<&dyn Array>,
    agg: Agg,
) -> Result<ArrowRleBatch> {
    if agg == Agg::Count {
        return coverage_by_key(keys, starts, ends, None);
    }
    let values = values.ok_or(Error::MissingValues)?;
    let (ids, first_rows) = group_rows(keys)?;
    let groups = group_indices(&ids, first_rows.len());
    let rles = groups
        .par_iter()
        .map(|idx| {
            let starts: Vec<i64> = idx.iter().map(|&i| starts[i]).collect();
            let ends: Vec<i64> = idx.iter().map(|&i| ends[i]).collect();
            let indices = UInt64Array::from_iter_values(idx.iter().map(|&i| i as u64));
            let values = take(values, &indices, None)?;
            coverage_agg(&starts, &ends, Some(values.as_ref()), None, agg)
        })
        .collect::<Result<Vec<_>>>()?;
    let first_rows = UInt64Array::from(first_rows);
    let columns = keys
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &first_rows, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let keys = RecordBatch::try_new(keys.schema(), columns)?;
    let data_type = rles
        .first()
        .map_or(DataType::Null, |r| r.data_type().clone());
    ArrowRleBatch::from_rles(keys, rles, &data_type)
}

fn group_indices(ids: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); n];
    for (i, &g) in ids.iter().enumerate() {
        groups[g].push(i);
    }
    groups
}

/// Numbers each key row by first appearance; returns the group of every row
/// and the first row of every group.
fn group_rows(keys: &RecordBatch) -> Result<(Vec<usize>, Vec<u64>)> {
    let converter = key_converter(keys)?;
    let rows = key_rows(&converter, keys)?;
    let mut groups = HashMap::new();
    let mut first_rows = Vec::new();
    let ids = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            *groups.entry(row).or_insert_with(|| {
                first_rows.push(i as u64);
                first_rows.len() - 1
            })
        })
        .collect();
    Ok((ids, first_rows))
}

fn into_arrow_batch<T: ArrowPrimitiveType>(
    keys: &RecordBatch,
    first_rows: Vec<u64>,
    batch: RleBatch<usize, T::Native>,
) -> Result<ArrowRleBatch> {
    let first_rows = UInt64Array::from(first_rows);
    let columns = keys
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &first_rows, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let keys = RecordBatch::try_new(keys.schema(), columns)?;
    let (_, offsets, ends, values) = batch.into_parts();
    ArrowRleBatch::from_parts_unchecked(keys, offsets, ends, to_array::<T>(values))
}

/// [`coverage`] per key, computed in parallel.
///
/// Row `i` of `keys` (any number of columns, of any types) is the key of
/// interval `i`. Keys appear in order of first appearance, and each sequence
/// covers `0..max(end)` of its own intervals.
pub fn coverage_by_key(
    keys: &RecordBatch,
    starts: &[i64],
    ends: &[i64],
    values: Option<&dyn Array>,
) -> Result<ArrowRleBatch> {
    let (ids, first_rows) = group_rows(keys)?;
    let Some(values) = values else {
        let batch = runlength::coverage_count_by_key(&ids, starts, ends)?;
        return into_arrow_batch::<Int64Type>(keys, first_rows, batch);
    };
    with_numeric_type!(values, |T| {
        let batch = runlength::coverage_sum_by_key(&ids, starts, ends, native::<T>(values)?)?;
        into_arrow_batch::<T>(keys, first_rows, batch)
    })
}
