//! Generic run-length encoded sequences.
//!
//! A run-length encoded sequence stores runs of repeated values once:
//! `a a a b b` becomes runs `[3, 2]` with values `[a, b]`. This crate works
//! with any value type, and is built around one idea: combining sequences is
//! split into a *plan*, computed from run boundaries alone, and a separate step
//! that computes values.
//!
//! ```
//! use runlength::prelude::*;
//!
//! let a = Rle::from_runs(&[3, 2], vec![1, 2]).unwrap();
//! let b = Rle::from_runs(&[1, 4], vec![10, 20]).unwrap();
//!
//! // The plan says which run of each input covers each output segment.
//! let p = a.plan_with(&b);
//! assert_eq!(p.ends, vec![1, 3, 5]);
//! assert_eq!(p.idx_a, vec![0, 0, 1]);
//! assert_eq!(p.idx_b, vec![0, 1, 1]);
//!
//! // Any operation can be applied to any value types...
//! let labels = a.zip_with(&b, None, None, |x, y| format!("{x}/{y}")).unwrap();
//! assert_eq!(labels.get(0), Some(&"1/10".to_string()));
//!
//! // ...and the usual operators pad the shorter side with zero.
//! let sum = &a + &b;
//! assert_eq!(sum.to_dense(), vec![11, 21, 21, 22, 22]);
//! assert_eq!(sum.runs(), vec![1, 2, 2]);
//! ```
//!
//! # Main types
//!
//! * [`Rle`] owns a sequence; [`RleRef`] borrows one. Their shared
//!   operations are on the [`RleLike`] trait.
//! * [`RleBatch`] stores many sequences under arbitrary keys in shared
//!   arrays, and combines batches key by key in parallel.
//! * [`Plan`] and [`MergePlan`] describe how runs combine without touching
//!   values, so callers can compute values any way they like.
//! * [`coverage_sum`], [`coverage_count`] and their `_by_key` variants build
//!   sequences from intervals.
//!
//! # Conventions
//!
//! * Positions and lengths are `i64`, and every sequence starts at 0.
//! * Intervals are half-open, `[start, end)`.
//! * Neighbouring runs merge when their values are [`Identical`]: `NaN`
//!   merges with `NaN`, `-0.0` does not merge with `0.0`.
//! * Operations merge their result by default; `_raw` variants do not.

mod batch;
mod coverage;
mod error;
mod identical;
mod many;
mod merge;
mod num;
mod ops;
mod plan;
mod reduce;
mod rle;
mod slice;
mod transform;

pub use batch::{How, RleBatch};
pub use coverage::{
    OverlapPlan, coverage_count, coverage_count_by_key, coverage_max, coverage_max_by_key,
    coverage_min, coverage_min_by_key, coverage_sum, coverage_sum_by_key, coverage_sum_opt,
    overlap_plan,
};
pub use error::{Error, Result};
pub use identical::Identical;
pub use many::{PlanMany, plan_many, zip_many};
pub use merge::{Keep, MergePlan, merge_plan};
pub use num::Num;
pub use ops::{zip_with, zip_with_raw};
pub use plan::{Plan, plan};
pub use rle::{Rle, RleLike, RleRef};
pub use transform::concat;

/// The types and traits needed for everyday use.
pub mod prelude {
    pub use crate::{How, Identical, Keep, Num, Rle, RleBatch, RleLike, RleRef};
}
