use rustc_hash::FxHashMap;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow_schema::{ArrowError, DataType};
use arrow_select::concat::concat as concat_arrays;
use arrow_select::interleave::interleave;
use arrow_select::take::take;
use arrow_select::zip::zip;
use rayon::prelude::*;
use runlength::{Rle, RleLike, plan_many};

use crate::batch::{ArrowRleBatch, key_converter, key_rows};
use crate::error::{Error, Result};
use crate::kernels::Kernel;
use crate::rle::ArrowRle;

/// Joins sequences end to end and merges identical neighbours, including at
/// the joins. All sequences must have the same value type.
pub fn concat(rles: &[ArrowRle]) -> Result<ArrowRle> {
    let Some(first) = rles.first() else {
        return Ok(ArrowRle::empty(&DataType::Null));
    };
    let mut ends = Vec::with_capacity(rles.iter().map(|r| r.num_runs()).sum());
    let mut offset: i64 = 0;
    for rle in rles {
        ends.extend(rle.ends().iter().map(|&e| e + offset));
        offset = offset
            .checked_add(rle.len())
            .ok_or(runlength::Error::Overflow)?;
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
    /// are padded with [`default_fill`](crate::default_fill).
    ///
    /// [`kernels::add`]: crate::kernels::add
    pub fn collapse(&self, keep: &[&str], kernel: Kernel) -> Result<ArrowRleBatch> {
        let keys = self.keys();
        let indices = keep
            .iter()
            .map(|name| keys.schema().index_of(name))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let kept = keys.project(&indices)?;
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut first_rows = Vec::new();
        if kept.num_columns() == 0 {
            // Every sequence goes into one group, with an empty key.
            if !self.is_empty() {
                groups.push((0..self.len()).collect());
                first_rows.push(0);
            }
        } else {
            let converter = key_converter(&kept)?;
            let rows = key_rows(&converter, &kept)?;
            let mut positions = FxHashMap::default();
            for (i, row) in rows.iter().enumerate() {
                let g = *positions.entry(row).or_insert_with(|| {
                    first_rows.push(i as u64);
                    groups.push(Vec::new());
                    groups.len() - 1
                });
                groups[g].push(i);
            }
        }

        let data_type = self.values().data_type();
        let rles = groups
            .par_iter()
            .enumerate()
            .map(|(position, members)| {
                let fold = || {
                    let mut acc = self.get_index(members[0]).expect("in range").merge()?;
                    for &i in &members[1..] {
                        let next = self.get_index(i).expect("in range");
                        acc = acc.op(&next, kernel)?;
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
        // Keeping no columns leaves one group with an empty key.
        let new_keys = RecordBatch::try_new_with_options(
            kept.schema(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(first_rows.len())),
        )?;
        let out_type = rles
            .first()
            .map_or(data_type.clone(), |r| r.data_type().clone());
        ArrowRleBatch::from_rles(new_keys, rles, &out_type)
    }
}

impl ArrowRleBatch {
    /// Shifts every sequence, in parallel; see [`ArrowRle::shift`].
    pub fn shift(&self, distance: i64, fill: &dyn Array) -> Result<ArrowRleBatch> {
        self.map_rles(|r| r.shift(distance, fill))
    }

    /// Reverses every sequence, in parallel.
    pub fn reversed(&self) -> Result<ArrowRleBatch> {
        self.map_rles(ArrowRle::reversed)
    }

    /// Transforms every sequence in parallel, keeping the keys. All results
    /// must have the same value type.
    pub fn map_rles<F>(&self, f: F) -> Result<ArrowRleBatch>
    where
        F: Fn(&ArrowRle) -> Result<ArrowRle> + Sync,
    {
        let rles = (0..self.len())
            .into_par_iter()
            .map(|i| f(&self.get_index(i).expect("in range")).map_err(|e| e.in_key(i)))
            .collect::<Result<Vec<_>>>()?;
        let data_type = rles
            .first()
            .map_or(self.values().data_type().clone(), |r| r.data_type().clone());
        ArrowRleBatch::from_rles(self.keys().clone(), rles, &data_type)
    }

    /// Joins batches end to end key by key, in parallel: each key's sequences
    /// are concatenated in batch order, and a batch without the key adds
    /// nothing. Keys appear in order of first appearance. All batches must
    /// have the same key columns and value type.
    pub fn concat(batches: &[ArrowRleBatch]) -> Result<ArrowRleBatch> {
        let Some(first) = batches.first() else {
            return Err(Error::NoInputs);
        };
        let converter = key_converter(first.keys())?;
        let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut positions = FxHashMap::default();
        for (b, batch) in batches.iter().enumerate() {
            first.check_key_schema(batch.keys())?;
            let rows = key_rows(&converter, batch.keys())?;
            for (i, row) in rows.iter().enumerate() {
                let g = *positions.entry(row.owned()).or_insert_with(|| {
                    groups.push(Vec::new());
                    groups.len() - 1
                });
                groups[g].push((b, i));
            }
        }
        let rles = groups
            .par_iter()
            .enumerate()
            .map(|(position, members)| {
                let parts: Vec<ArrowRle> = members
                    .iter()
                    .map(|&(b, i)| batches[b].get_index(i).expect("in range"))
                    .collect();
                concat(&parts).map_err(|e| e.in_key(position))
            })
            .collect::<Result<Vec<_>>>()?;
        let sources: Vec<(usize, usize)> = groups.iter().map(|g| g[0]).collect();
        let columns = (0..first.keys().num_columns())
            .map(|c| {
                let arrays: Vec<&dyn Array> = batches
                    .iter()
                    .map(|b| b.keys().column(c).as_ref())
                    .collect();
                interleave(&arrays, &sources)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let keys = RecordBatch::try_new_with_options(
            first.keys().schema(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(sources.len())),
        )?;
        ArrowRleBatch::from_rles(keys, rles, first.values().data_type())
    }
}
