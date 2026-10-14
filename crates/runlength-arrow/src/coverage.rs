use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, PrimitiveArray, RecordBatch, UInt64Array};
use arrow_buffer::ScalarBuffer;
use arrow_select::take::take;
use runlength::RleBatch;

use crate::batch::{ArrowRleBatch, key_converter, key_rows};
use crate::error::{Error, Result};
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

/// Runs `$body` with `$T` bound to the Arrow type of `$values`, for the
/// numeric types coverage supports.
macro_rules! with_numeric_type {
    ($values:expr, |$T:ident| $body:expr) => {
        match $values.data_type() {
            arrow_schema::DataType::Int8 => { type $T = Int8Type; $body }
            arrow_schema::DataType::Int16 => { type $T = Int16Type; $body }
            arrow_schema::DataType::Int32 => { type $T = Int32Type; $body }
            arrow_schema::DataType::Int64 => { type $T = Int64Type; $body }
            arrow_schema::DataType::UInt8 => { type $T = UInt8Type; $body }
            arrow_schema::DataType::UInt16 => { type $T = UInt16Type; $body }
            arrow_schema::DataType::UInt32 => { type $T = UInt32Type; $body }
            arrow_schema::DataType::UInt64 => { type $T = UInt64Type; $body }
            arrow_schema::DataType::Float32 => { type $T = Float32Type; $body }
            arrow_schema::DataType::Float64 => { type $T = Float64Type; $body }
            other => Err(Error::UnsupportedType(other.clone())),
        }
    };
}

/// Coverage of half-open intervals `[start, end)`.
///
/// With `values`, each position holds the sum of the values of the intervals
/// covering it, in the values' own numeric type (which must have no nulls).
/// Without, it holds the number of covering intervals as `Int64`. Uncovered
/// positions are 0. The result covers `0..max(end)`, or `0..length`.
pub fn coverage(starts: &[i64], ends: &[i64], values: Option<&dyn Array>, length: Option<i64>) -> Result<ArrowRle> {
    let Some(values) = values else {
        let rle = runlength::coverage_count(starts, ends, length)?;
        return Ok(ArrowRle::from_rle::<Int64Type>(&rle));
    };
    with_numeric_type!(values, |T| {
        let rle = runlength::coverage_sum(starts, ends, native::<T>(values)?, length)?;
        Ok(ArrowRle::from_rle::<T>(&rle))
    })
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

fn into_arrow_batch<T: ArrowPrimitiveType>(keys: &RecordBatch, first_rows: Vec<u64>, batch: RleBatch<usize, T::Native>) -> Result<ArrowRleBatch> {
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
pub fn coverage_by_key(keys: &RecordBatch, starts: &[i64], ends: &[i64], values: Option<&dyn Array>) -> Result<ArrowRleBatch> {
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
