use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Sub};
use std::sync::OnceLock;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::identical::Identical;
use crate::ops;
use crate::rle::{Rle, RleLike, RleRef, validate_ends};
use crate::slice;

/// Which keys a combination of two batches keeps, as in a DataFrame join.
///
/// A key missing from one side is treated as a sequence of length 0 there,
/// so it is padded entirely with that side's fill value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum How {
    /// Keys from either side.
    #[default]
    Outer,
    /// Keys present on both sides.
    Inner,
    /// Keys of the left side.
    Left,
    /// Keys of the right side.
    Right,
}

/// Many run-length encoded sequences, each identified by a key.
///
/// The sequences are stored back to back in shared arrays, like an Arrow list
/// column: sequence `i` is runs `offsets[i]..offsets[i + 1]` of `ends` and
/// `values`, and its ends restart at 0. Keys can be any hashable type; use a
/// tuple for keys made of several parts. Key order is preserved.
///
/// Operations over the whole batch run in parallel across sequences.
pub struct RleBatch<K, V> {
    keys: Vec<K>,
    offsets: Vec<usize>,
    ends: Vec<i64>,
    values: Vec<V>,
    /// Key → position, built on the first lookup by key.
    index: OnceLock<HashMap<K, usize>>,
}

impl<K, V> RleBatch<K, V> {
    /// An empty batch.
    pub fn new() -> Self {
        RleBatch {
            keys: Vec::new(),
            offsets: vec![0],
            ends: Vec::new(),
            values: Vec::new(),
            index: OnceLock::new(),
        }
    }

    /// Builds a batch by calling `fill(k, ends, values)` for each key, which
    /// must append valid runs for key `k`. Keys must be unique.
    pub(crate) fn from_slices(
        keys: Vec<K>,
        mut fill: impl FnMut(usize, &mut Vec<i64>, &mut Vec<V>),
    ) -> Self {
        let mut out = RleBatch::new();
        out.offsets.reserve(keys.len());
        for k in 0..keys.len() {
            fill(k, &mut out.ends, &mut out.values);
            out.offsets.push(out.ends.len());
        }
        out.keys = keys;
        out
    }

    /// Concatenates sequences under keys that are known to be unique.
    pub(crate) fn from_rles_unchecked(keys: Vec<K>, rles: Vec<Rle<V>>) -> Self {
        debug_assert_eq!(keys.len(), rles.len());
        let total = rles.iter().map(|r| r.ends.len()).sum();
        let mut out = RleBatch::new();
        out.ends.reserve(total);
        out.values.reserve(total);
        for rle in rles {
            out.ends.extend(rle.ends);
            out.values.extend(rle.values);
            out.offsets.push(out.ends.len());
        }
        out.keys = keys;
        out
    }

    /// The number of sequences.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    /// Run boundaries per sequence; `len() + 1` entries.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The run ends of all sequences, back to back.
    pub fn ends(&self) -> &[i64] {
        &self.ends
    }

    /// The run values of all sequences, back to back.
    pub fn values(&self) -> &[V] {
        &self.values
    }

    /// The sequence at `position`, or `None` if out of range.
    pub fn get_index(&self, position: usize) -> Option<RleRef<'_, V>> {
        (position < self.len()).then(|| self.rle_at(position))
    }

    fn rle_at(&self, position: usize) -> RleRef<'_, V> {
        let (start, end) = (self.offsets[position], self.offsets[position + 1]);
        RleRef::new_unchecked(&self.ends[start..end], &self.values[start..end])
    }

    /// Iterates over `(key, sequence)` in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, RleRef<'_, V>)> {
        self.keys
            .iter()
            .enumerate()
            .map(|(i, k)| (k, self.rle_at(i)))
    }

    /// Splits the batch into keys, offsets, ends and values.
    pub fn into_parts(self) -> (Vec<K>, Vec<usize>, Vec<i64>, Vec<V>) {
        (self.keys, self.offsets, self.ends, self.values)
    }

    /// Computes `f` for every sequence in parallel, in key order.
    pub fn reduce<T, F>(&self, f: F) -> Vec<T>
    where
        K: Send + Sync,
        V: Sync,
        T: Send,
        F: Fn(RleRef<'_, V>) -> T + Sync,
    {
        (0..self.len())
            .into_par_iter()
            .map(|i| f(self.rle_at(i)))
            .collect()
    }

    /// Transforms every sequence in parallel, keeping the keys.
    pub fn map_rles<W, F>(&self, f: F) -> RleBatch<K, W>
    where
        K: Clone + Send + Sync,
        V: Sync,
        W: Send,
        F: Fn(RleRef<'_, V>) -> Rle<W> + Sync,
    {
        RleBatch::from_rles_unchecked(self.keys.clone(), self.reduce(f))
    }
}

impl<K, V> RleBatch<K, V> {
    /// Combines the sequences whose keys map to the same new key, folding
    /// each group with `f` in key order, in parallel across groups.
    ///
    /// For example, `batch.collapse(|(chrom, _strand)| chrom.clone(), Some(&0),
    /// |a, b| a + b)` adds the strands of each chromosome together. New keys
    /// appear in order of first appearance. Sequences of different lengths are
    /// padded with `fill`; with no fill value that is an error.
    pub fn collapse<K2, F, G>(&self, key: F, fill: Option<&V>, f: G) -> Result<RleBatch<K2, V>>
    where
        K: Send + Sync,
        K2: Hash + Eq + Clone + Send + Sync,
        V: Clone + Identical + Send + Sync,
        F: Fn(&K) -> K2,
        G: Fn(&V, &V) -> V + Sync,
    {
        let mut positions: HashMap<K2, usize> = HashMap::new();
        let mut unique = Vec::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (i, k) in self.keys.iter().enumerate() {
            let new_key = key(k);
            let g = *positions.entry(new_key.clone()).or_insert_with(|| {
                unique.push(new_key);
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i);
        }
        let rles = groups
            .par_iter()
            .enumerate()
            .map(|(position, members)| {
                let mut acc = self.rle_at(members[0]).merged();
                for &i in &members[1..] {
                    acc = ops::zip_with(acc.view(), self.rle_at(i), fill, fill, &f).map_err(
                        |source| Error::InKey {
                            position,
                            source: Box::new(source),
                        },
                    )?;
                }
                Ok(acc)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RleBatch::from_rles_unchecked(unique, rles))
    }
}

impl<K: Hash + Eq + Clone, V> RleBatch<K, V> {
    /// Builds a batch from `(key, sequence)` pairs. Duplicate keys are an error.
    pub fn from_rles<I: IntoIterator<Item = (K, Rle<V>)>>(rles: I) -> Result<Self> {
        let mut out = RleBatch::new();
        for (key, rle) in rles {
            out.push(key, rle)?;
        }
        Ok(out)
    }

    /// Builds a batch from its flat parts, validating them.
    pub fn from_parts(
        keys: Vec<K>,
        offsets: Vec<usize>,
        ends: Vec<i64>,
        values: Vec<V>,
    ) -> Result<Self> {
        if offsets.len() != keys.len() + 1 {
            return Err(Error::LengthMismatch {
                what: "offsets and keys + 1",
                left: offsets.len(),
                right: keys.len() + 1,
            });
        }
        if ends.len() != values.len() {
            return Err(Error::LengthMismatch {
                what: "ends and values",
                left: ends.len(),
                right: values.len(),
            });
        }
        if offsets[0] != 0
            || offsets[keys.len()] != ends.len()
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(Error::InvalidOffsets);
        }
        for (position, w) in offsets.windows(2).enumerate() {
            validate_ends(&ends[w[0]..w[1]]).map_err(|source| Error::InKey {
                position,
                source: Box::new(source),
            })?;
        }
        let mut index = HashMap::with_capacity(keys.len());
        for (position, key) in keys.iter().enumerate() {
            if index.insert(key.clone(), position).is_some() {
                return Err(Error::DuplicateKey { position });
            }
        }
        Ok(RleBatch {
            keys,
            offsets,
            ends,
            values,
            index: OnceLock::from(index),
        })
    }

    fn index(&self) -> &HashMap<K, usize> {
        self.index.get_or_init(|| {
            self.keys
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, k)| (k, i))
                .collect()
        })
    }

    /// Appends a sequence under a new key.
    pub fn push(&mut self, key: K, rle: Rle<V>) -> Result<()> {
        let position = self.keys.len();
        self.index();
        let index = self.index.get_mut().expect("initialised above");
        if index.contains_key(&key) {
            return Err(Error::DuplicateKey { position });
        }
        index.insert(key.clone(), position);
        self.keys.push(key);
        self.ends.extend(rle.ends);
        self.values.extend(rle.values);
        self.offsets.push(self.ends.len());
        Ok(())
    }

    /// The position of `key`, if present.
    pub fn position<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.index().get(key).copied()
    }

    /// The sequence for `key`, if present.
    pub fn get<Q>(&self, key: &Q) -> Option<RleRef<'_, V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.position(key).map(|i| self.rle_at(i))
    }

    /// Many slices at once: slice `k` is `keys[k]`'s sequence at
    /// `starts[k]..ends[k]`, clamped. A missing key gives an empty slice.
    /// The result is keyed by slice number.
    pub fn getitems<Q>(
        &self,
        keys: &[Q],
        starts: &[i64],
        ends: &[i64],
    ) -> Result<RleBatch<usize, V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
        V: Clone,
    {
        slice::validate_intervals(starts, ends)?;
        if keys.len() != starts.len() {
            return Err(Error::LengthMismatch {
                what: "keys and starts",
                left: keys.len(),
                right: starts.len(),
            });
        }
        Ok(RleBatch::from_slices(
            (0..keys.len()).collect(),
            |k, out_ends, out_values| {
                let rle = self.get(&keys[k]).unwrap_or_else(RleRef::empty);
                slice::slice_into(rle, starts[k], ends[k], out_ends, out_values)
            },
        ))
    }

    /// The output keys of a join and the position of each on both sides.
    fn join<'a, B>(
        &'a self,
        other: &'a RleBatch<K, B>,
        how: How,
    ) -> Vec<(&'a K, Option<usize>, Option<usize>)> {
        let mut out = Vec::with_capacity(self.len().max(other.len()));
        for (i, key) in self.keys.iter().enumerate() {
            let j = other.position(key);
            if j.is_some() || matches!(how, How::Outer | How::Left) {
                out.push((key, Some(i), j));
            }
        }
        if matches!(how, How::Outer | How::Right) {
            for (j, key) in other.keys.iter().enumerate() {
                if self.position(key).is_none() {
                    out.push((key, None, Some(j)));
                }
            }
        }
        out
    }

    /// Combines the sequences of two batches key by key with `f`, in parallel,
    /// merging identical neighbours in each result.
    ///
    /// `how` picks the keys; keys appear in the left batch's order, followed
    /// by keys only in the right batch. Sequences of different lengths, and
    /// keys missing on one side, are padded with that side's fill value; with
    /// no fill value that is an error naming the output position.
    pub fn zip_with<B, C, F>(
        &self,
        other: &RleBatch<K, B>,
        how: How,
        fill_a: Option<&V>,
        fill_b: Option<&B>,
        f: F,
    ) -> Result<RleBatch<K, C>>
    where
        K: Send + Sync,
        V: Sync,
        B: Sync,
        C: Identical + Send,
        F: Fn(&V, &B) -> C + Sync,
    {
        let pairs = self.join(other, how);
        let rles = pairs
            .par_iter()
            .enumerate()
            .map(|(position, &(_, i, j))| {
                let a = i.map_or_else(RleRef::empty, |i| self.rle_at(i));
                let b = j.map_or_else(RleRef::empty, |j| other.rle_at(j));
                ops::zip_with(a, b, fill_a, fill_b, &f).map_err(|source| Error::InKey {
                    position,
                    source: Box::new(source),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let keys = pairs.into_iter().map(|(key, _, _)| key.clone()).collect();
        Ok(RleBatch::from_rles_unchecked(keys, rles))
    }
}

impl<K, V> Default for RleBatch<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Clone, V: Clone> Clone for RleBatch<K, V> {
    fn clone(&self) -> Self {
        RleBatch {
            keys: self.keys.clone(),
            offsets: self.offsets.clone(),
            ends: self.ends.clone(),
            values: self.values.clone(),
            index: OnceLock::new(),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for RleBatch<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Same keys in the same order, and each pair of sequences
/// [`equals`](RleLike::equals).
impl<K: PartialEq, V: Identical> PartialEq for RleBatch<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.keys == other.keys
            && self
                .iter()
                .zip(other.iter())
                .all(|((_, a), (_, b))| a.equals(&b))
    }
}

// Operators do an outer join and pad with `V::default()` (zero for numbers).
macro_rules! batch_binop {
    ($Trait:ident, $method:ident) => {
        impl<K, V> $Trait<&RleBatch<K, V>> for &RleBatch<K, V>
        where
            K: Hash + Eq + Clone + Send + Sync,
            V: $Trait<Output = V> + Clone + Default + Identical + Send + Sync,
        {
            type Output = RleBatch<K, V>;

            fn $method(self, rhs: &RleBatch<K, V>) -> RleBatch<K, V> {
                let zero = V::default();
                self.zip_with(rhs, How::Outer, Some(&zero), Some(&zero), |a, b| {
                    $Trait::$method(a.clone(), b.clone())
                })
                .expect("both sides have a fill value")
            }
        }
    };
}

batch_binop!(Add, add);
batch_binop!(Sub, sub);
batch_binop!(Mul, mul);
batch_binop!(Div, div);
batch_binop!(BitAnd, bitand);
batch_binop!(BitOr, bitor);
batch_binop!(BitXor, bitxor);
