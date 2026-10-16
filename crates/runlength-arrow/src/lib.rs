//! Run-length encoded sequences over Apache Arrow arrays.
//!
//! This crate builds on [`runlength`] and uses its plans and merge rules, but
//! stores values as an Arrow array, so values can be of any Arrow type:
//! numbers, strings, dictionaries, lists, structs, and nulls in any of them.
//!
//! * Combining sequences gathers values through a [`runlength::Plan`] and
//!   applies any Arrow kernel ([`kernels`] has the usual ones).
//! * Merging uses the same rule as the core crate for every type
//!   ([`identical_neighbours`]).
//! * [`ArrowRleBatch`] keys sequences by a table of key columns of any
//!   types, and converts to one long table for polars or pandas.
//!
//! ```
//! use std::sync::Arc;
//! use arrow_array::{ArrayRef, StringArray};
//! use runlength_arrow::{ArrowRle, kernels};
//!
//! let names: ArrayRef = Arc::new(StringArray::from(vec!["a", "a", "b", "c", "c"]));
//! let rle = ArrowRle::from_dense(names).unwrap();
//! assert_eq!(rle.runs(), vec![2, 1, 2]);
//!
//! let other: ArrayRef = Arc::new(StringArray::from(vec!["a", "x", "c", "c", "c"]));
//! let other = ArrowRle::from_dense(other).unwrap();
//! let same = rle.zip_with(&other, None, None, kernels::eq).unwrap();
//! assert_eq!(same.runs(), vec![1, 2, 2]);
//! ```

#![warn(missing_docs)]

mod batch;
mod coverage;
mod error;
mod fused;
mod identical;
pub mod kernels;
mod numeric;
mod reduce;
mod rle;
mod slices;
mod transform;

pub use batch::{ArrowRleBatch, RunExtents};
pub use coverage::{
    Agg, coverage, coverage_agg, coverage_agg_by_key, coverage_by_key, overlap_plan_by_key,
};
pub use error::{Error, Result};
pub use identical::identical_neighbours;
pub use numeric::{cast_checked, common_numeric_type};
pub use rle::{ArrowRle, default_fill};
pub use runlength::{How, Keep, OverlapPlan, PlanMany, overlap_plan, plan_many};
pub use slices::SliceAgg;
pub use transform::{concat, zip_many};
