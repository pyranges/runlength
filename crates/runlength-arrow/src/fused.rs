//! A single-pass path for the built-in operators on numbers without nulls.
//!
//! The general path gathers both sides onto the plan's segments, applies an
//! Arrow kernel, and merges the result: about eight passes over the data. For
//! the common case (both sides the same numeric type, no nulls) this runs the
//! core crate's fused loop on the Arrow buffers directly instead. It uses
//! Arrow's own checked arithmetic and comparisons, so results are identical;
//! where Arrow would raise an error (integer overflow, division by zero) it
//! gives up, and the general path raises it.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, BooleanArray, PrimitiveArray,
};
use arrow_schema::ArrowError;
use runlength::{Identical, Rle, RleRef, zip_with};

use crate::kernels::Kernel;
use crate::numeric::with_numeric_type;
use crate::rle::ArrowRle;

/// `a <kernel> b` padded with zero, or `None` if the general path must do it.
pub(crate) fn fused(a: &ArrowRle, b: &ArrowRle, kernel: Kernel) -> Option<ArrowRle> {
    if a.data_type() != b.data_type() || a.values.null_count() > 0 || b.values.null_count() > 0 {
        return None;
    }
    let out: crate::Result<Option<ArrowRle>> =
        with_numeric_type!(a.values, |T| Ok(fused_typed::<T>(a, b, kernel)));
    out.ok().flatten()
}

fn fused_typed<T>(a: &ArrowRle, b: &ArrowRle, kernel: Kernel) -> Option<ArrowRle>
where
    T: ArrowPrimitiveType,
    T::Native: ArrowNativeTypeOp + Identical,
{
    let a = RleRef::from_ends(&a.ends, a.values.as_primitive::<T>().values()).ok()?;
    let b = RleRef::from_ends(&b.ends, b.values.as_primitive::<T>().values()).ok()?;
    match kernel {
        Kernel::Add => arithmetic::<T>(a, b, |x, y| x.add_checked(y)),
        Kernel::Sub => arithmetic::<T>(a, b, |x, y| x.sub_checked(y)),
        Kernel::Mul => arithmetic::<T>(a, b, |x, y| x.mul_checked(y)),
        Kernel::Div => arithmetic::<T>(a, b, |x, y| x.div_checked(y)),
        Kernel::Eq => Some(compare(a, b, |x, y| x.is_eq(y))),
        Kernel::Ne => Some(compare(a, b, |x, y| x.is_ne(y))),
        Kernel::Lt => Some(compare(a, b, |x, y| x.is_lt(y))),
        Kernel::Le => Some(compare(a, b, |x, y| x.is_le(y))),
        Kernel::Gt => Some(compare(a, b, |x, y| x.is_gt(y))),
        Kernel::Ge => Some(compare(a, b, |x, y| x.is_ge(y))),
        Kernel::And | Kernel::Or => None,
    }
}

fn arithmetic<T>(
    a: RleRef<'_, T::Native>,
    b: RleRef<'_, T::Native>,
    f: impl Fn(T::Native, T::Native) -> Result<T::Native, ArrowError>,
) -> Option<ArrowRle>
where
    T: ArrowPrimitiveType,
    T::Native: ArrowNativeTypeOp + Identical,
{
    let zero = T::Native::ZERO;
    let mut failed = false;
    let rle = zip_with(a, b, Some(&zero), Some(&zero), |&x, &y| {
        f(x, y).unwrap_or_else(|_| {
            failed = true;
            zero
        })
    })
    .ok()?;
    (!failed).then(|| {
        into_arrow(rle, |values| {
            Arc::new(PrimitiveArray::<T>::new(values.into(), None))
        })
    })
}

fn compare<N: ArrowNativeTypeOp>(
    a: RleRef<'_, N>,
    b: RleRef<'_, N>,
    f: impl Fn(N, N) -> bool,
) -> ArrowRle {
    let zero = N::ZERO;
    let rle = zip_with(a, b, Some(&zero), Some(&zero), |&x, &y| f(x, y))
        .expect("both sides have a fill value");
    into_arrow(rle, |values| Arc::new(BooleanArray::from(values)))
}

fn into_arrow<V>(rle: Rle<V>, values: impl FnOnce(Vec<V>) -> ArrayRef) -> ArrowRle {
    let (ends, vals) = rle.into_parts();
    ArrowRle {
        ends,
        values: values(vals),
    }
}
