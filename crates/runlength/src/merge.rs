use crate::error::{Error, Result};
use crate::identical::Identical;
use crate::rle::{Rle, RleRef};

/// Which value a merged group keeps when its runs hold different values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Keep {
    #[default]
    First,
    Last,
}

/// How runs are grouped by a merge mask, without any values involved.
///
/// Group `g` consists of the input runs `offsets[g]..offsets[g + 1]` and ends
/// at `ends[g]`. Use it to aggregate values per group in any way, for example
/// a length-weighted mean.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergePlan {
    /// Cumulative ends of the merged groups.
    pub ends: Vec<i64>,
    /// Group boundaries in input-run indices; `num_groups() + 1` entries.
    pub offsets: Vec<usize>,
}

impl MergePlan {
    pub fn num_groups(&self) -> usize {
        self.ends.len()
    }

    /// The group of each input run.
    pub fn group_ids(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.offsets.last().copied().unwrap_or(0));
        for (group, bounds) in self.offsets.windows(2).enumerate() {
            ids.extend(std::iter::repeat_n(group, bounds[1] - bounds[0]));
        }
        ids
    }
}

fn check_mask(num_runs: usize, mask: &[bool]) -> Result<()> {
    let expected = num_runs.saturating_sub(1);
    if mask.len() != expected {
        return Err(Error::LengthMismatch {
            what: "merge mask and neighbouring run pairs",
            left: mask.len(),
            right: expected,
        });
    }
    Ok(())
}

pub(crate) fn merge_plan(ends: &[i64], mask: &[bool]) -> Result<MergePlan> {
    check_mask(ends.len(), mask)?;
    let mut out = MergePlan {
        ends: Vec::new(),
        offsets: vec![0],
    };
    for (r, &end) in ends.iter().enumerate() {
        if mask.get(r).copied().unwrap_or(false) {
            continue;
        }
        out.ends.push(end);
        out.offsets.push(r + 1);
    }
    Ok(out)
}

pub(crate) fn is_merged<V: Identical>(values: &[V]) -> bool {
    values.windows(2).all(|w| !w[0].identical(&w[1]))
}

pub(crate) fn merged<V: Clone + Identical>(rle: RleRef<'_, V>) -> Rle<V> {
    let mut out = Rle::new();
    for (&end, value) in rle.ends.iter().zip(rle.values) {
        out.push_merged(end, value.clone());
    }
    out
}

impl<V> Rle<V> {
    /// Merges neighbouring runs with identical values (see
    /// [`Identical`]), in place.
    pub fn merge(self) -> Self
    where
        V: Identical,
    {
        self.merge_where(|values, w, r| values[w].identical(&values[r]), Keep::First)
    }

    /// Merges neighbouring runs for which `pred(left, right)` is true,
    /// keeping the first value of each merged group.
    ///
    /// `pred` sees the original neighbouring values, so a tolerance check
    /// can chain: with `|a, b| (a - b).abs() < 0.1`, the values `1.0, 1.05,
    /// 1.1` all merge.
    pub fn merge_by<F: FnMut(&V, &V) -> bool>(self, mut pred: F) -> Self {
        let mask: Vec<bool> = self.values.windows(2).map(|w| pred(&w[0], &w[1])).collect();
        self.merge_mask(&mask, Keep::First)
            .expect("the mask has one entry per neighbouring pair")
    }

    /// Merges run `i + 1` into run `i` wherever `mask[i]` is true. `mask`
    /// must have one entry per pair of neighbouring runs.
    pub fn merge_mask(self, mask: &[bool], keep: Keep) -> Result<Self> {
        check_mask(self.ends.len(), mask)?;
        Ok(self.merge_where(|_, _, r| mask[r - 1], keep))
    }

    /// Compacts runs in place. `join(values, w, r)` decides whether run `r`
    /// joins the group currently kept at `w`; `values[r]` is still the
    /// original value of run `r` when it is called.
    fn merge_where<F>(mut self, mut join: F, keep: Keep) -> Self
    where
        F: FnMut(&[V], usize, usize) -> bool,
    {
        if self.ends.is_empty() {
            return self;
        }
        let mut w = 0;
        for r in 1..self.ends.len() {
            if join(&self.values, w, r) {
                self.ends[w] = self.ends[r];
                if keep == Keep::Last {
                    self.values.swap(w, r);
                }
            } else {
                w += 1;
                self.ends[w] = self.ends[r];
                self.values.swap(w, r);
            }
        }
        self.ends.truncate(w + 1);
        self.values.truncate(w + 1);
        self
    }
}
