//! Length-weighted reductions for sequences of numeric values.
//!
//! These borrow the Arrow values directly, without copying, and use the core
//! crate's reductions. Values must be a numeric type without nulls, except
//! for `min` and `max`, which work on any type Arrow can sort.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, PrimitiveArray, new_null_array};
use arrow_cast::cast;
use arrow_ord::sort::{SortOptions, sort_to_indices};
use arrow_schema::DataType;
use arrow_select::take::take;
use runlength::{RleLike, RleRef};

use crate::error::{Error, Result};
use crate::numeric::with_numeric_type;
use crate::rle::ArrowRle;

/// Booleans count as 0 and 1 in sums, means and variances, as in numpy and
/// polars.
fn as_numbers(rle: &ArrowRle) -> Result<std::borrow::Cow<'_, ArrowRle>> {
    if rle.data_type() == &DataType::Boolean {
        let values = cast(rle.values.as_ref(), &DataType::Int64)?;
        return Ok(std::borrow::Cow::Owned(ArrowRle {
            ends: rle.ends.clone(),
            values,
        }));
    }
    Ok(std::borrow::Cow::Borrowed(rle))
}

fn core_view<'a, T: ArrowPrimitiveType>(rle: &'a ArrowRle) -> Result<RleRef<'a, T::Native>> {
    let values = rle.values.as_primitive::<T>();
    if values.null_count() > 0 {
        return Err(Error::UnexpectedNulls);
    }
    Ok(RleRef::from_ends(&rle.ends, values.values())?)
}

fn single<T: ArrowPrimitiveType>(value: Option<T::Native>) -> ArrayRef {
    Arc::new(std::iter::once(value).collect::<PrimitiveArray<T>>())
}

impl ArrowRle {
    /// The sum over all positions, as one value of the values' type
    /// (`Int64` for booleans, counting the true positions). Integer sums wrap
    /// on overflow.
    pub fn sum(&self) -> Result<ArrayRef> {
        let rle = as_numbers(self)?;
        with_numeric_type!(rle.values, |T| Ok(single::<T>(Some(
            core_view::<T>(&rle)?.sum()
        ))))
    }

    /// The mean over all positions (for booleans, the fraction that is true),
    /// or `None` if the sequence is empty.
    pub fn mean(&self) -> Result<Option<f64>> {
        let rle = as_numbers(self)?;
        with_numeric_type!(rle.values, |T| Ok(core_view::<T>(&rle)?.mean()))
    }

    /// The variance over all positions with `ddof` delta degrees of freedom,
    /// or `None` if `len() <= ddof`.
    pub fn variance(&self, ddof: i64) -> Result<Option<f64>> {
        let rle = as_numbers(self)?;
        with_numeric_type!(rle.values, |T| Ok(core_view::<T>(&rle)?.variance(ddof)))
    }

    /// The standard deviation; see [`variance`](ArrowRle::variance).
    pub fn std(&self, ddof: i64) -> Result<Option<f64>> {
        Ok(self.variance(ddof)?.map(f64::sqrt))
    }

    /// The smallest value ignoring `NaN`, as one value (null if there is none).
    /// Numbers must not be null; other types skip nulls.
    pub fn min(&self) -> Result<ArrayRef> {
        if !self.values.data_type().is_numeric() {
            return self.sorted_first(false);
        }
        with_numeric_type!(self.values, |T| Ok(single::<T>(
            core_view::<T>(self)?.min().copied()
        )))
    }

    /// The largest value ignoring `NaN`, as one value (null if there is none).
    /// Numbers must not be null; other types skip nulls.
    pub fn max(&self) -> Result<ArrayRef> {
        if !self.values.data_type().is_numeric() {
            return self.sorted_first(true);
        }
        with_numeric_type!(self.values, |T| Ok(single::<T>(
            core_view::<T>(self)?.max().copied()
        )))
    }

    /// The first non-null value in sort order, for types other than numbers
    /// (strings, dates, booleans, ...). Run lengths don't matter here.
    fn sorted_first(&self, descending: bool) -> Result<ArrayRef> {
        let options = SortOptions {
            descending,
            nulls_first: false,
        };
        let first = sort_to_indices(self.values.as_ref(), Some(options), Some(1))?;
        Ok(match first.values().first() {
            Some(&i) if self.values.is_valid(i as usize) => {
                take(self.values.as_ref(), &first, None)?
            }
            _ => new_null_array(self.values.data_type(), 1),
        })
    }
}
