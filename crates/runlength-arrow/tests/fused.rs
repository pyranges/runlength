//! The operators take a single-pass path for numbers without nulls. Check it
//! against the general path (gather, Arrow kernel, merge) on inputs that
//! overflow, divide by zero, and hold NaN and signed zeros.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int8Array, Int64Array};
use proptest::prelude::*;
use runlength_arrow::kernels::Kernel;
use runlength_arrow::{ArrowRle, default_fill};

const KERNELS: [Kernel; 10] = [
    Kernel::Add,
    Kernel::Sub,
    Kernel::Mul,
    Kernel::Div,
    Kernel::Eq,
    Kernel::Ne,
    Kernel::Lt,
    Kernel::Le,
    Kernel::Gt,
    Kernel::Ge,
];

fn general(a: &ArrowRle, b: &ArrowRle, kernel: Kernel) -> runlength_arrow::Result<ArrowRle> {
    let fill = default_fill(a.data_type());
    a.zip_with(b, fill.as_deref(), fill.as_deref(), |x, y| {
        kernel.apply(x, y)
    })
}

fn check(a: &ArrowRle, b: &ArrowRle) -> Result<(), TestCaseError> {
    for kernel in KERNELS {
        match (a.zip_with_default(b, kernel), general(a, b, kernel)) {
            (Ok(fast), Ok(slow)) => {
                prop_assert_eq!(fast.ends(), slow.ends(), "{:?}", kernel);
                prop_assert_eq!(fast.data_type(), slow.data_type(), "{:?}", kernel);
                prop_assert!(fast == slow, "{:?}: {:?} vs {:?}", kernel, fast, slow);
            }
            (Err(_), Err(_)) => {}
            (fast, slow) => prop_assert!(false, "{:?}: {:?} vs {:?}", kernel, fast, slow),
        }
    }
    Ok(())
}

fn rle(values: ArrayRef, runs: &[i64]) -> ArrowRle {
    ArrowRle::from_runs(runs, values).unwrap()
}

proptest! {
    #[test]
    fn int8_with_overflow(pairs in prop::collection::vec((1i64..4, any::<i8>()), 0..8),
                          others in prop::collection::vec((1i64..4, any::<i8>()), 0..8)) {
        let (ra, va): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let (rb, vb): (Vec<_>, Vec<_>) = others.into_iter().unzip();
        check(&rle(Arc::new(Int8Array::from(va)), &ra), &rle(Arc::new(Int8Array::from(vb)), &rb))?;
    }

    #[test]
    fn int64_with_zeros(pairs in prop::collection::vec((1i64..4, -2i64..3), 0..8),
                        others in prop::collection::vec((1i64..4, -2i64..3), 0..8)) {
        let (ra, va): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let (rb, vb): (Vec<_>, Vec<_>) = others.into_iter().unzip();
        check(&rle(Arc::new(Int64Array::from(va)), &ra), &rle(Arc::new(Int64Array::from(vb)), &rb))?;
    }

    #[test]
    fn float64_with_nan_and_signed_zeros(pairs in prop::collection::vec((1i64..4, 0usize..6), 0..8),
                                         others in prop::collection::vec((1i64..4, 0usize..6), 0..8)) {
        let pick = [f64::NAN, -0.0, 0.0, 1.5, -2.0, f64::INFINITY];
        let (ra, ia): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let (rb, ib): (Vec<_>, Vec<_>) = others.into_iter().unzip();
        let va: Vec<f64> = ia.iter().map(|&i| pick[i]).collect();
        let vb: Vec<f64> = ib.iter().map(|&i| pick[i]).collect();
        check(&rle(Arc::new(Float64Array::from(va)), &ra), &rle(Arc::new(Float64Array::from(vb)), &rb))?;
    }
}

#[test]
fn nulls_and_mixed_types_take_the_general_path() {
    let with_null = rle(Arc::new(Int64Array::from(vec![Some(1), None])), &[2, 2]);
    let plain = rle(Arc::new(Int64Array::from(vec![5])), &[4]);
    let sum = with_null.zip_with_default(&plain, Kernel::Add).unwrap();
    assert!(sum == general(&with_null, &plain, Kernel::Add).unwrap());
    assert_eq!(sum.values().null_count(), 1);
}
