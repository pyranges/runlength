/// The equality used to decide whether two neighbouring runs merge.
///
/// Values merge when they are *identical*, which differs from `==` for floats:
///
/// * `NaN` is identical to `NaN`, so stretches of `NaN` merge into one run.
/// * `-0.0` and `0.0` are not identical, so the sign of zero is never lost.
/// * `None` is identical to `None`; `None` is never identical to `Some(NaN)`.
///
/// Every merge in this crate uses this rule, so results do not depend on which
/// code path produced them.
pub trait Identical {
    fn identical(&self, other: &Self) -> bool;
}

macro_rules! identical_via_eq {
    ($($t:ty),*) => {
        $(
            impl Identical for $t {
                #[inline]
                fn identical(&self, other: &Self) -> bool {
                    self == other
                }
            }
        )*
    };
}

identical_via_eq!(
    bool,
    char,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    String,
    str,
    ()
);

macro_rules! identical_float {
    ($($t:ty),*) => {
        $(
            impl Identical for $t {
                #[inline]
                fn identical(&self, other: &Self) -> bool {
                    (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
                }
            }
        )*
    };
}

identical_float!(f32, f64);

impl<T: Identical + ?Sized> Identical for &T {
    #[inline]
    fn identical(&self, other: &Self) -> bool {
        (**self).identical(*other)
    }
}

impl<T: Identical + ?Sized> Identical for Box<T> {
    #[inline]
    fn identical(&self, other: &Self) -> bool {
        (**self).identical(other)
    }
}

impl<T: Identical> Identical for Option<T> {
    #[inline]
    fn identical(&self, other: &Self) -> bool {
        match (self, other) {
            (None, None) => true,
            (Some(a), Some(b)) => a.identical(b),
            _ => false,
        }
    }
}

impl<T: Identical> Identical for [T] {
    fn identical(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a.identical(b))
    }
}

impl<T: Identical> Identical for Vec<T> {
    fn identical(&self, other: &Self) -> bool {
        self.as_slice().identical(other.as_slice())
    }
}

macro_rules! identical_tuple {
    ($($name:ident $idx:tt),+) => {
        impl<$($name: Identical),+> Identical for ($($name,)+) {
            #[inline]
            fn identical(&self, other: &Self) -> bool {
                $(self.$idx.identical(&other.$idx))&&+
            }
        }
    };
}

identical_tuple!(A 0);
identical_tuple!(A 0, B 1);
identical_tuple!(A 0, B 1, C 2);
identical_tuple!(A 0, B 1, C 2, D 3);

#[cfg(test)]
mod tests {
    use super::Identical;

    #[test]
    fn floats() {
        assert!(f64::NAN.identical(&f64::NAN));
        assert!(!0.0f64.identical(&-0.0));
        assert!(1.5f64.identical(&1.5));
        assert!(!Some(f64::NAN).identical(&None));
        assert!(None::<f64>.identical(&None));
    }
}
