use std::ops::{Add, Mul, Sub};

/// Numeric value types that support length-weighted reductions and coverage.
///
/// `Default::default()` is used as zero.
pub trait Num:
    Copy
    + PartialOrd
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Send
    + Sync
{
    /// Converts a run length to this type (with `as` semantics).
    fn from_i64(n: i64) -> Self;
    fn to_f64(self) -> f64;
}

macro_rules! impl_num {
    ($($t:ty),*) => {
        $(
            impl Num for $t {
                #[inline]
                fn from_i64(n: i64) -> Self {
                    n as $t
                }
                #[inline]
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_num!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
