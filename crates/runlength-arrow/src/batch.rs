use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, RecordBatchOptions, UInt64Array, new_empty_array};
use arrow_row::{RowConverter, Rows, SortField};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use arrow_select::concat::concat;
use arrow_select::interleave::interleave;
use arrow_select::take::take;
use rayon::prelude::*;
use runlength::How;

use crate::error::{Error, Result};
use crate::kernels::Kernel;
use crate::rle::{ArrowRle, default_fill};

/// Many Arrow-backed sequences, each identified by a row of a key table.
///
/// The keys are a [`RecordBatch`] with any number of columns of any types
/// (for example a chromosome and a strand column), one row per sequence.
/// Sequences are stored back to back: sequence `i` is runs
/// `offsets[i]..offsets[i + 1]` of `ends` and `values`, with ends restarting
/// at 0. Batch operations run in parallel across sequences.
#[derive(Clone, Debug)]
pub struct ArrowRleBatch {
    keys: RecordBatch,
    offsets: Vec<usize>,
    ends: Vec<i64>,
    values: ArrayRef,
}

/// A key table with one column.
pub(crate) fn single_key_column(name: &str, column: ArrayRef) -> RecordBatch {
    let schema = Schema::new(vec![Field::new(name, column.data_type().clone(), column.is_nullable())]);
    RecordBatch::try_new(Arc::new(schema), vec![column]).expect("one column matches its own schema")
}

/// Encodes key rows so they can be hashed and compared across batches.
pub(crate) fn key_rows(converter: &RowConverter, keys: &RecordBatch) -> Result<Rows> {
    Ok(converter.convert_columns(keys.columns())?)
}

pub(crate) fn key_converter(keys: &RecordBatch) -> Result<RowConverter> {
    let fields = keys
        .schema()
        .fields()
        .iter()
        .map(|f| SortField::new(f.data_type().clone()))
        .collect();
    Ok(RowConverter::new(fields)?)
}

fn concat_values(rles: &[ArrowRle], data_type: &DataType) -> Result<ArrayRef> {
    if rles.is_empty() {
        return Ok(new_empty_array(data_type));
    }
    let arrays: Vec<&dyn Array> = rles.iter().map(|r| r.values.as_ref()).collect();
    Ok(concat(&arrays)?)
}

impl ArrowRleBatch {
    /// Builds a batch from its flat parts, validating them. Keys must be
    /// unique.
    pub fn try_new(keys: RecordBatch, offsets: Vec<usize>, ends: Vec<i64>, values: ArrayRef) -> Result<Self> {
        let n = keys.num_rows();
        if offsets.len() != n + 1 {
            return Err(runlength::Error::LengthMismatch {
                what: "offsets and keys + 1",
                left: offsets.len(),
                right: n + 1,
            }
            .into());
        }
        if ends.len() != values.len() {
            return Err(runlength::Error::LengthMismatch {
                what: "ends and values",
                left: ends.len(),
                right: values.len(),
            }
            .into());
        }
        if offsets[0] != 0 || offsets[n] != ends.len() || offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(runlength::Error::InvalidOffsets.into());
        }
        for (position, w) in offsets.windows(2).enumerate() {
            let rle_ends = ends[w[0]..w[1]].to_vec();
            let values = values.slice(w[0], w[1] - w[0]);
            ArrowRle::try_new(rle_ends, values).map_err(|e| e.in_key(position))?;
        }
        let converter = key_converter(&keys)?;
        let rows = key_rows(&converter, &keys)?;
        let mut seen = HashSet::with_capacity(n);
        for (position, row) in rows.iter().enumerate() {
            if !seen.insert(row) {
                return Err(runlength::Error::DuplicateKey { position }.into());
            }
        }
        Ok(ArrowRleBatch {
            keys,
            offsets,
            ends,
            values,
        })
    }

    pub(crate) fn from_parts_unchecked(keys: RecordBatch, offsets: Vec<usize>, ends: Vec<i64>, values: ArrayRef) -> Result<Self> {
        debug_assert_eq!(offsets.len(), keys.num_rows() + 1);
        Ok(ArrowRleBatch {
            keys,
            offsets,
            ends,
            values,
        })
    }

    /// Builds a batch from one key row and one sequence per entry. All
    /// sequences must have the same value type, `data_type`.
    pub fn from_rles(keys: RecordBatch, rles: Vec<ArrowRle>, data_type: &DataType) -> Result<Self> {
        if keys.num_rows() != rles.len() {
            return Err(runlength::Error::LengthMismatch {
                what: "keys and sequences",
                left: keys.num_rows(),
                right: rles.len(),
            }
            .into());
        }
        let mut offsets = Vec::with_capacity(rles.len() + 1);
        offsets.push(0);
        let mut ends = Vec::new();
        for rle in &rles {
            ends.extend_from_slice(&rle.ends);
            offsets.push(ends.len());
        }
        let values = concat_values(&rles, data_type)?;
        Self::try_new(keys, offsets, ends, values)
    }

    /// The number of sequences.
    pub fn len(&self) -> usize {
        self.keys.num_rows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The key table, one row per sequence.
    pub fn keys(&self) -> &RecordBatch {
        &self.keys
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The run ends of all sequences, back to back.
    pub fn ends(&self) -> &[i64] {
        &self.ends
    }

    /// The run values of all sequences, back to back.
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }

    /// The sequence at `position`. The values are a zero-copy slice.
    pub fn get_index(&self, position: usize) -> Option<ArrowRle> {
        (position < self.len()).then(|| self.rle_at(position))
    }

    fn rle_at(&self, position: usize) -> ArrowRle {
        let (start, end) = (self.offsets[position], self.offsets[position + 1]);
        ArrowRle {
            ends: self.ends[start..end].to_vec(),
            values: self.values.slice(start, end - start),
        }
    }

    /// The sequences in key order.
    pub fn iter(&self) -> impl Iterator<Item = ArrowRle> + '_ {
        (0..self.len()).map(|i| self.rle_at(i))
    }

    /// The position of each row of `keys` in this batch, or `None` if absent.
    /// `keys` must have the same column types as this batch's keys.
    pub fn positions(&self, keys: &RecordBatch) -> Result<Vec<Option<usize>>> {
        self.check_key_schema(keys)?;
        let converter = key_converter(&self.keys)?;
        let mine = key_rows(&converter, &self.keys)?;
        let index: HashMap<_, _> = mine.iter().enumerate().map(|(i, row)| (row, i)).collect();
        let theirs = key_rows(&converter, keys)?;
        Ok(theirs.iter().map(|row| index.get(&row).copied()).collect())
    }

    fn check_key_schema(&self, keys: &RecordBatch) -> Result<()> {
        let (mine, theirs) = (self.keys.schema(), keys.schema());
        let same = mine.fields().len() == theirs.fields().len()
            && mine
                .fields()
                .iter()
                .zip(theirs.fields())
                .all(|(a, b)| a.name() == b.name() && a.data_type() == b.data_type());
        if same { Ok(()) } else { Err(Error::KeySchemaMismatch) }
    }

    /// The output keys of a join, as (side, row) pairs for `interleave`, and
    /// each key's position on both sides.
    #[allow(clippy::type_complexity)]
    fn join(&self, other: &ArrowRleBatch, how: How) -> Result<(Vec<(usize, usize)>, Vec<(Option<usize>, Option<usize>)>)> {
        self.check_key_schema(&other.keys)?;
        let converter = key_converter(&self.keys)?;
        let (rows_a, rows_b) = (key_rows(&converter, &self.keys)?, key_rows(&converter, &other.keys)?);
        let index_a: HashMap<_, _> = rows_a.iter().enumerate().map(|(i, row)| (row, i)).collect();
        let index_b: HashMap<_, _> = rows_b.iter().enumerate().map(|(j, row)| (row, j)).collect();

        let (mut sources, mut pairs) = (Vec::new(), Vec::new());
        for (i, row) in rows_a.iter().enumerate() {
            let j = index_b.get(&row).copied();
            if j.is_some() || matches!(how, How::Outer | How::Left) {
                sources.push((0, i));
                pairs.push((Some(i), j));
            }
        }
        if matches!(how, How::Outer | How::Right) {
            for (j, row) in rows_b.iter().enumerate() {
                if !index_a.contains_key(&row) {
                    sources.push((1, j));
                    pairs.push((None, Some(j)));
                }
            }
        }
        Ok((sources, pairs))
    }

    /// Combines two batches key by key, in parallel, merging each result.
    ///
    /// Both key tables must have the same column names and types. `how` picks
    /// the keys, which appear in the left batch's order followed by keys only
    /// in the right batch. A key missing on one side is an empty sequence
    /// there, padded with that side's fill value. See
    /// [`ArrowRle::zip_with`] for `f`.
    pub fn zip_with<F>(&self, other: &ArrowRleBatch, how: How, fill_a: Option<&dyn Array>, fill_b: Option<&dyn Array>, f: F) -> Result<ArrowRleBatch>
    where
        F: Fn(&dyn Array, &dyn Array) -> std::result::Result<ArrayRef, ArrowError> + Sync,
    {
        let (sources, pairs) = self.join(other, how)?;
        let rles = pairs
            .par_iter()
            .enumerate()
            .map(|(position, &(i, j))| {
                let a = i.map_or_else(|| ArrowRle::empty(self.values.data_type()), |i| self.rle_at(i));
                let b = j.map_or_else(|| ArrowRle::empty(other.values.data_type()), |j| other.rle_at(j));
                a.zip_with(&b, fill_a, fill_b, &f).map_err(|e| e.in_key(position))
            })
            .collect::<Result<Vec<_>>>()?;

        let columns = (0..self.keys.num_columns())
            .map(|c| interleave(&[self.keys.column(c).as_ref(), other.keys.column(c).as_ref()], &sources))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let keys = RecordBatch::try_new_with_options(
            self.keys.schema(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(sources.len())),
        )?;
        let data_type = rles.first().map_or(DataType::Null, |r| r.data_type().clone());
        let mut offsets = Vec::with_capacity(rles.len() + 1);
        offsets.push(0);
        let mut ends = Vec::new();
        for rle in &rles {
            ends.extend_from_slice(&rle.ends);
            offsets.push(ends.len());
        }
        let values = concat_values(&rles, &data_type)?;
        Self::from_parts_unchecked(keys, offsets, ends, values)
    }

    /// The batch as one long table: the key columns, then `start`, `end` and
    /// `value`, with one row per run. This is the natural shape for polars or
    /// pandas.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let mut key_index = Vec::with_capacity(self.ends.len());
        let mut starts = Vec::with_capacity(self.ends.len());
        for (k, w) in self.offsets.windows(2).enumerate() {
            let mut prev = 0;
            for &end in &self.ends[w[0]..w[1]] {
                key_index.push(k as u64);
                starts.push(prev);
                prev = end;
            }
        }
        let key_index = UInt64Array::from(key_index);
        let mut fields: Vec<Field> = self.keys.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        let mut columns = self
            .keys
            .columns()
            .iter()
            .map(|c| take(c.as_ref(), &key_index, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        fields.push(Field::new("start", DataType::Int64, false));
        fields.push(Field::new("end", DataType::Int64, false));
        fields.push(Field::new("value", self.values.data_type().clone(), true));
        columns.push(Arc::new(Int64Array::from(starts)));
        columns.push(Arc::new(Int64Array::from(self.ends.clone())));
        columns.push(self.values.clone());
        Ok(RecordBatch::try_new_with_options(
            Arc::new(Schema::new(fields)),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(self.ends.len())),
        )?)
    }

    /// Combines with `other` by an outer join, using [`default_fill`].
    fn op(&self, other: &ArrowRleBatch, f: Kernel) -> Result<ArrowRleBatch> {
        let fill_a = default_fill(self.values.data_type());
        let fill_b = default_fill(other.values.data_type());
        self.zip_with(other, How::Outer, fill_a.as_deref(), fill_b.as_deref(), f)
    }
}

crate::kernels::kernel_methods!(
    ArrowRleBatch,
    " Keys are outer-joined and shorter sides padded with [`default_fill`]."
);

impl PartialEq for ArrowRleBatch {
    /// Same keys in the same order, and equal sequences for each.
    fn eq(&self, other: &Self) -> bool {
        self.keys == other.keys && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
