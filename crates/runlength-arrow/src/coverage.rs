use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, Int64Array, LargeListArray, PrimitiveArray, RecordBatch,
    RecordBatchOptions, StringArray, UInt64Array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};
use arrow_select::take::take;
use rayon::prelude::*;
use runlength::{Rle, RleBatch};

use crate::batch::{ArrowRleBatch, group_rows};
use crate::error::{Error, Result};
use crate::numeric::with_numeric_type;
use crate::rle::{ArrowRle, check_same_len};

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
    // Other string encodings (polars uses Utf8View) go through Utf8 and back.
    if matches!(values.data_type(), DataType::LargeUtf8 | DataType::Utf8View) {
        let utf8 = cast(values, &DataType::Utf8)?;
        return extreme(starts, ends, utf8.as_ref(), length, max)?.cast(values.data_type());
    }
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

/// Checks that keys, starts, ends and values have one entry per interval.
fn check_intervals(
    keys: &RecordBatch,
    starts: &[i64],
    ends: &[i64],
    values: Option<&dyn Array>,
) -> Result<()> {
    check_same_len("keys and starts", keys.num_rows(), starts.len())?;
    check_same_len("starts and ends", starts.len(), ends.len())?;
    if let Some(values) = values {
        check_same_len("starts and values", starts.len(), values.len())?;
    }
    Ok(())
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
    check_intervals(keys, starts, ends, Some(values))?;
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
    check_intervals(keys, starts, ends, values)?;
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

/// [`runlength::overlap_plan`] per key, as one long table: the key columns,
/// then `start` and `end` of each segment and `ids`, a list of the intervals
/// covering it (positions in the input, sorted). Keys appear in order of
/// first appearance; each key's segments cover `0..max(end)` of its own
/// intervals.
///
/// Exploding `ids` and joining interval values onto them gives any
/// aggregation per segment, for example with a polars `group_by`.
pub fn overlap_plan_by_key(
    keys: &RecordBatch,
    starts: &[i64],
    ends: &[i64],
) -> Result<RecordBatch> {
    check_intervals(keys, starts, ends, None)?;
    let (ids, first_rows) = group_rows(keys)?;
    let (groups, plans) = runlength::overlap_plan_by_key(&ids, starts, ends)?;
    debug_assert!(groups.iter().enumerate().all(|(i, &g)| i == g));
    let segments: usize = plans.iter().map(|p| p.num_segments()).sum();
    let mut key_rows = Vec::with_capacity(segments);
    let mut seg_starts = Vec::with_capacity(segments);
    let mut seg_ends = Vec::with_capacity(segments);
    let mut list_offsets = Vec::with_capacity(segments + 1);
    list_offsets.push(0i64);
    let mut items = Vec::new();
    for (plan, &row) in plans.iter().zip(&first_rows) {
        let mut prev = 0;
        for s in 0..plan.num_segments() {
            key_rows.push(row);
            seg_starts.push(prev);
            seg_ends.push(plan.ends[s]);
            prev = plan.ends[s];
            items.extend(plan.ids(s).iter().map(|&i| i as i64));
            list_offsets.push(items.len() as i64);
        }
    }
    let key_rows = UInt64Array::from(key_rows);
    let mut columns = keys
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &key_rows, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut fields: Vec<Field> = keys
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let item = Arc::new(Field::new("item", DataType::Int64, false));
    fields.push(Field::new("start", DataType::Int64, false));
    fields.push(Field::new("end", DataType::Int64, false));
    fields.push(Field::new("ids", DataType::LargeList(item.clone()), false));
    columns.push(Arc::new(Int64Array::from(seg_starts)));
    columns.push(Arc::new(Int64Array::from(seg_ends)));
    columns.push(Arc::new(LargeListArray::new(
        item,
        OffsetBuffer::new(ScalarBuffer::from(list_offsets)),
        Arc::new(Int64Array::from(items)),
        None,
    )));
    Ok(RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(segments)),
    )?)
}
