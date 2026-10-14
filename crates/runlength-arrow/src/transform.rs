use std::collections::HashMap;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{ArrowError, DataType};
use arrow_select::concat::concat as concat_arrays;
use arrow_select::take::take;
use arrow_select::zip::zip;
use rayon::prelude::*;
use runlength::{Rle, RleLike, plan_many};

use crate::batch::{ArrowRleBatch, key_converter, key_rows};
use crate::error::{Error, Result};
use crate::kernels::Kernel;
use crate::rle::{ArrowRle, default_fill};

/// Joins sequences end to end and merges identical neighbours, including at
/// the joins. All sequences must have the same value type.
pub fn concat(rles: &[ArrowRle]) -> Result<ArrowRle> {
    let Some(first) = rles.first() else {
        return Ok(ArrowRle::empty(&DataType::Null));
    };
    let mut ends = Vec::with_capacity(rles.iter().map(|r| r.num_runs()).sum());
    let mut offset = 0;
    for rle in rles {
        ends.extend(rle.ends().iter().map(|&e| e + offset));
        offset += rle.len();
    }
    let values: Vec<&dyn Array> = rles.iter().map(|r| r.values().as_ref()).collect();
    let values = if values.len() == 1 {
        first.values().clone()
    } else {
        concat_arrays(&values)?
    };
    ArrowRle::try_new(ends, values)?.merge()
}

/// Combines several sequences position by position, and merges the result.
///
/// `f` receives each input's values gathered onto the segments of the
/// [`plan_many`] (equal-length arrays, one per input) and must return one
/// value per segment. Shorter inputs are padded with their fill value (one
/// optional fill per input); with none, unequal lengths are an error.
pub fn zip_many<F>(rles: &[ArrowRle], fills: &[Option<&dyn Array>], f: F) -> Result<ArrowRle>
where
    F: FnOnce(&[ArrayRef]) -> std::result::Result<ArrayRef, ArrowError>,
{
    if fills.len() != rles.len() {
        return Err(runlength::Error::LengthMismatch {
            what: "fills and sequences",
            left: fills.len(),
            right: rles.len(),
        }
        .into());
    }
    let max = rles.iter().map(|r| r.len()).max().unwrap_or(0);
    for (rle, fill) in rles.iter().zip(fills) {
        if rle.len() < max && fill.is_none() {
            return Err(runlength::Error::UnequalLength {
                left: rle.len(),
                right: max,
            }
            .into());
        }
    }
    let ends: Vec<&[i64]> = rles.iter().map(|r| r.ends()).collect();
    let plan = plan_many(&ends);
    let gathered = rles
        .iter()
        .zip(fills)
        .zip(&plan.idx)
        .map(|((rle, fill), idx)| crate::rle::gather(rle.values(), idx, *fill))
        .collect::<Result<Vec<_>>>()?;
    let values = f(&gathered)?;
    if values.len() != plan.ends.len() {
        return Err(Error::WrongLength {
            expected: plan.ends.len(),
            actual: values.len(),
        });
    }
    ArrowRle::try_new(plan.ends, values)?.merge()
}

impl ArrowRle {
    /// Shifts values `distance` positions right (left if negative), keeping
    /// the length; positions shifted in hold `fill`, a single value of the
    /// same type. The result is merged.
    pub fn shift(&self, distance: i64, fill: &dyn Array) -> Result<ArrowRle> {
        crate::rle::check_fill(Some(fill), self.data_type())?;
        let len = self.len();
        if distance == 0 {
            return self.clone().merge();
        }
        let fill_run = |n: i64| ArrowRle::from_runs(&[n], crate::rle::array_ref(fill));
        if distance.unsigned_abs() >= len as u64 {
            return fill_run(len);
        }
        if distance > 0 {
            concat(&[fill_run(distance)?, self.slice(0, len - distance)])
        } else {
            concat(&[self.slice(-distance, len), fill_run(-distance)?])
        }
    }

    /// The sequence back to front.
    pub fn reversed(&self) -> Result<ArrowRle> {
        let rle = Rle::from_ends(self.ends().to_vec(), vec![(); self.num_runs()])?.reversed();
        let n = self.num_runs() as u64;
        let indices = UInt64Array::from_iter_values((0..n).rev());
        ArrowRle::try_new(
            rle.into_parts().0,
            take(self.values().as_ref(), &indices, None)?,
        )
    }

    /// For a boolean sequence, the half-open ranges `(starts, ends)` where it
    /// is true (nulls count as false), with touching ranges joined.
    pub fn true_ranges(&self) -> Result<(Vec<i64>, Vec<i64>)> {
        let values = self
            .values()
            .as_boolean_opt()
            .ok_or_else(|| Error::UnsupportedType(self.data_type().clone()))?;
        let flags: Vec<bool> = values.iter().map(|v| v == Some(true)).collect();
        let rle = Rle::from_ends(self.ends().to_vec(), flags)?;
        Ok(rle.ranges_where(|&v| v))
    }

    /// Replaces nulls with `fill`, a single value of the same type.
    pub fn fill_null(&self, fill: &dyn Array) -> Result<ArrowRle> {
        crate::rle::check_fill(Some(fill), self.data_type())?;
        if self.values().null_count() == 0 {
            return Ok(self.clone());
        }
        let is_null = arrow_arith::boolean::is_null(self.values().as_ref())?;
        let fill = arrow_array::Scalar::new(crate::rle::array_ref(fill));
        let values = zip(&is_null, &fill, self.values())?;
        ArrowRle::try_new(self.ends().to_vec(), values)?.merge()
    }
}

impl ArrowRleBatch {
    /// Combines the sequences whose keys agree on the `keep` columns, folding
    /// each group with `kernel` in key order, in parallel across groups.
    ///
    /// For example, keeping only `Chromosome` with [`kernels::add`] adds the
    /// strands of each chromosome together. Sequences of different lengths
    /// are padded with [`default_fill`].
    ///
    /// [`kernels::add`]: crate::kernels::add
    pub fn collapse(&self, keep: &[&str], kernel: Kernel) -> Result<ArrowRleBatch> {
        let keys = self.keys();
        let indices = keep
            .iter()
            .map(|name| keys.schema().index_of(name))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let kept = keys.project(&indices)?;
        let converter = key_converter(&kept)?;
        let rows = key_rows(&converter, &kept)?;

        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut first_rows = Vec::new();
        let mut positions = HashMap::new();
        for (i, row) in rows.iter().enumerate() {
            let g = *positions.entry(row).or_insert_with(|| {
                first_rows.push(i as u64);
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i);
        }

        let data_type = self.values().data_type();
        let fill = default_fill(data_type);
        let rles = groups
            .par_iter()
            .enumerate()
            .map(|(position, members)| {
                let fold = || {
                    let mut acc = self.get_index(members[0]).expect("in range").merge()?;
                    for &i in &members[1..] {
                        let next = self.get_index(i).expect("in range");
                        acc = acc.zip_with(&next, fill.as_deref(), fill.as_deref(), kernel)?;
                    }
                    Ok(acc)
                };
                fold().map_err(|e: Error| e.in_key(position))
            })
            .collect::<Result<Vec<_>>>()?;

        let first_rows = UInt64Array::from(first_rows);
        let columns = kept
            .columns()
            .iter()
            .map(|c| take(c.as_ref(), &first_rows, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let new_keys = RecordBatch::try_new(kept.schema(), columns)?;
        let out_type = rles
            .first()
            .map_or(data_type.clone(), |r| r.data_type().clone());
        ArrowRleBatch::from_rles(new_keys, rles, &out_type)
    }
}
