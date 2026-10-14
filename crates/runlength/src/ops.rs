use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use crate::error::{Error, Result};
use crate::identical::Identical;
use crate::plan::Segments;
use crate::rle::{Rle, RleLike, RleRef};

fn check_lengths<A, B>(a: RleRef<'_, A>, b: RleRef<'_, B>, fill_a: bool, fill_b: bool) -> Result<()> {
    let (len_a, len_b) = (a.len(), b.len());
    if (len_a < len_b && !fill_a) || (len_b < len_a && !fill_b) {
        return Err(Error::UnequalLength {
            left: len_a,
            right: len_b,
        });
    }
    Ok(())
}

#[inline]
fn pick<'v, T>(values: &'v [T], i: usize, fill: Option<&'v T>) -> &'v T {
    values
        .get(i)
        .or(fill)
        .expect("the length check guarantees a fill value past the end")
}

/// Combines two sequences position by position with `f`, merging identical
/// neighbours in the result.
///
/// If the sequences have different lengths, the shorter one is padded with
/// its fill value. With no fill value for the shorter side this is an
/// [`Error::UnequalLength`].
///
/// ```
/// use runlength::{zip_with, Rle, RleLike};
///
/// let a = Rle::from_runs(&[3, 2], vec![1, 2]).unwrap();
/// let b = Rle::from_runs(&[1, 2], vec![10, 20]).unwrap();
/// let c = zip_with(a.view(), b.view(), None, Some(&0), |x, y| x + y).unwrap();
/// assert_eq!(c.to_dense(), vec![11, 21, 21, 2, 2]);
/// ```
pub fn zip_with<A, B, C, F>(
    a: RleRef<'_, A>,
    b: RleRef<'_, B>,
    fill_a: Option<&A>,
    fill_b: Option<&B>,
    mut f: F,
) -> Result<Rle<C>>
where
    C: Identical,
    F: FnMut(&A, &B) -> C,
{
    check_lengths(a, b, fill_a.is_some(), fill_b.is_some())?;
    let mut out = Rle::new();
    for (end, i, j) in Segments::new(a.ends, b.ends) {
        let value = f(pick(a.values, i, fill_a), pick(b.values, j, fill_b));
        out.push_merged(end, value);
    }
    Ok(out)
}

/// Like [`zip_with`] but keeps one run per plan segment, without merging.
pub fn zip_with_raw<A, B, C, F>(
    a: RleRef<'_, A>,
    b: RleRef<'_, B>,
    fill_a: Option<&A>,
    fill_b: Option<&B>,
    mut f: F,
) -> Result<Rle<C>>
where
    F: FnMut(&A, &B) -> C,
{
    check_lengths(a, b, fill_a.is_some(), fill_b.is_some())?;
    let capacity = a.num_runs() + b.num_runs();
    let mut ends = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);
    for (end, i, j) in Segments::new(a.ends, b.ends) {
        ends.push(end);
        values.push(f(pick(a.values, i, fill_a), pick(b.values, j, fill_b)));
    }
    Ok(Rle { ends, values })
}

pub(crate) fn compare<V: Default>(
    a: RleRef<'_, V>,
    b: RleRef<'_, V>,
    f: impl FnMut(&V, &V) -> bool,
) -> Rle<bool> {
    let zero = V::default();
    zip_with(a, b, Some(&zero), Some(&zero), f).expect("both sides have a fill value")
}

// Arithmetic and logical operators pad the shorter side with `V::default()`
// (zero for numbers) and merge the result.
macro_rules! rle_binop {
    ($Trait:ident, $method:ident) => {
        impl<V> $Trait<&Rle<V>> for &Rle<V>
        where
            V: $Trait<Output = V> + Clone + Default + Identical,
        {
            type Output = Rle<V>;

            fn $method(self, rhs: &Rle<V>) -> Rle<V> {
                let zero = V::default();
                zip_with(self.view(), rhs.view(), Some(&zero), Some(&zero), |a, b| {
                    $Trait::$method(a.clone(), b.clone())
                })
                .expect("both sides have a fill value")
            }
        }

        impl<V> $Trait<Rle<V>> for Rle<V>
        where
            V: $Trait<Output = V> + Clone + Default + Identical,
        {
            type Output = Rle<V>;

            fn $method(self, rhs: Rle<V>) -> Rle<V> {
                $Trait::$method(&self, &rhs)
            }
        }
    };
}

rle_binop!(Add, add);
rle_binop!(Sub, sub);
rle_binop!(Mul, mul);
rle_binop!(Div, div);
rle_binop!(BitAnd, bitand);
rle_binop!(BitOr, bitor);
rle_binop!(BitXor, bitxor);

// Negation and logical not map identical values to identical values and
// distinct values to distinct values, so the result needs no merging.
macro_rules! rle_unop {
    ($Trait:ident, $method:ident) => {
        impl<V> $Trait for &Rle<V>
        where
            V: $Trait<Output = V> + Clone,
        {
            type Output = Rle<V>;

            fn $method(self) -> Rle<V> {
                self.map_raw(|v| $Trait::$method(v.clone()))
            }
        }

        impl<V> $Trait for Rle<V>
        where
            V: $Trait<Output = V>,
        {
            type Output = Rle<V>;

            fn $method(self) -> Rle<V> {
                Rle {
                    ends: self.ends,
                    values: self.values.into_iter().map($Trait::$method).collect(),
                }
            }
        }
    };
}

rle_unop!(Neg, neg);
rle_unop!(Not, not);
