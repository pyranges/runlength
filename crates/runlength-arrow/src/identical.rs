use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_ord::cmp::not_distinct;
use arrow_ord::ord::make_comparator;
use arrow_schema::{DataType, SortOptions};
use runlength::Identical;

use crate::error::Result;

/// For each pair of neighbouring values, whether they are identical by the
/// same rule as the core crate's [`Identical`]: nulls are identical to nulls,
/// `NaN` to `NaN`, and `-0.0` is not identical to `0.0`.
///
/// Returns `values.len() - 1` entries (none for fewer than two values). Works
/// for every Arrow type, including nested ones.
pub fn identical_neighbours(values: &dyn Array) -> Result<Vec<bool>> {
    let n = values.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    // Arrow compares floats by total order, under which NaNs with different
    // bit patterns differ, so floats use the core rule directly.
    match values.data_type() {
        DataType::Float16 => return Ok(float_mask(values.as_primitive::<Float16Type>(), |v| v.to_f32())),
        DataType::Float32 => return Ok(float_mask(values.as_primitive::<Float32Type>(), |v| v)),
        DataType::Float64 => return Ok(float_mask(values.as_primitive::<Float64Type>(), |v| v)),
        _ => {}
    }
    let (left, right) = (values.slice(0, n - 1), values.slice(1, n - 1));
    if let Ok(mask) = not_distinct(&left, &right) {
        return Ok(mask.values().iter().collect());
    }
    // Types the vectorised kernel does not support, such as lists and structs.
    let cmp = make_comparator(values, values, SortOptions::default())?;
    Ok((0..n - 1).map(|i| cmp(i, i + 1).is_eq()).collect())
}

fn float_mask<T, F>(values: &PrimitiveArray<T>, convert: impl Fn(T::Native) -> F) -> Vec<bool>
where
    T: ArrowPrimitiveType,
    F: Identical,
{
    (0..values.len() - 1)
        .map(|i| match (values.is_valid(i), values.is_valid(i + 1)) {
            (false, false) => true,
            (true, true) => convert(values.value(i)).identical(&convert(values.value(i + 1))),
            _ => false,
        })
        .collect()
}
