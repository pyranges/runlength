//! Length-weighted reductions for sequences of numeric values.
//!
//! These borrow the Arrow values directly, without copying, and use the core
//! crate's reductions. Values must be a numeric type without nulls.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, PrimitiveArray};
use runlength::{RleLike, RleRef};

use crate::error::{Error, Result};
use crate::numeric::with_numeric_type;
use crate::rle::ArrowRle;

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
    /// The sum over all positions, as one value of the values' type. Integer
    /// sums wrap on overflow.
    pub fn sum(&self) -> Result<ArrayRef> {
        with_numeric_type!(self.values, |T| Ok(single::<T>(Some(
            core_view::<T>(self)?.sum()
        ))))
    }

    /// The mean over all positions, or `None` if the sequence is empty.
    pub fn mean(&self) -> Result<Option<f64>> {
        with_numeric_type!(self.values, |T| Ok(core_view::<T>(self)?.mean()))
    }

    /// The variance over all positions with `ddof` delta degrees of freedom,
    /// or `None` if `len() <= ddof`.
    pub fn variance(&self, ddof: i64) -> Result<Option<f64>> {
        with_numeric_type!(self.values, |T| Ok(core_view::<T>(self)?.variance(ddof)))
    }

    /// The standard deviation; see [`variance`](ArrowRle::variance).
    pub fn std(&self, ddof: i64) -> Result<Option<f64>> {
        Ok(self.variance(ddof)?.map(f64::sqrt))
    }

    /// The smallest value ignoring `NaN`, as one value (null if there is none).
    pub fn min(&self) -> Result<ArrayRef> {
        with_numeric_type!(self.values, |T| Ok(single::<T>(
            core_view::<T>(self)?.min().copied()
        )))
    }

    /// The largest value ignoring `NaN`, as one value (null if there is none).
    pub fn max(&self) -> Result<ArrayRef> {
        with_numeric_type!(self.values, |T| Ok(single::<T>(
            core_view::<T>(self)?.max().copied()
        )))
    }
}
