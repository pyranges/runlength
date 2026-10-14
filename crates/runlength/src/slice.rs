use crate::batch::RleBatch;
use crate::error::{Error, Result};
use crate::rle::{Rle, RleLike, RleRef};

pub(crate) fn run_index(ends: &[i64], pos: i64) -> Option<usize> {
    let len = ends.last().copied().unwrap_or(0);
    if pos < 0 || pos >= len {
        return None;
    }
    Some(ends.partition_point(|&end| end <= pos))
}

/// Checks that every interval has `0 <= start <= end`.
pub(crate) fn validate_intervals(starts: &[i64], ends: &[i64]) -> Result<()> {
    if starts.len() != ends.len() {
        return Err(Error::LengthMismatch {
            what: "starts and ends",
            left: starts.len(),
            right: ends.len(),
        });
    }
    for (index, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        if start < 0 || end < start {
            return Err(Error::InvalidInterval { index, start, end });
        }
    }
    Ok(())
}

/// Appends the runs of `rle[start..end]`, shifted to start at 0, to `ends`
/// and `values`. The range is clamped to the sequence.
pub(crate) fn slice_into<V: Clone>(
    rle: RleRef<'_, V>,
    start: i64,
    end: i64,
    ends: &mut Vec<i64>,
    values: &mut Vec<V>,
) {
    let (start, end) = (start.max(0), end.min(rle.len()));
    if start >= end {
        return;
    }
    let first = rle.ends.partition_point(|&e| e <= start);
    let last = rle.ends.partition_point(|&e| e < end);
    ends.extend(rle.ends[first..last].iter().map(|&e| e - start));
    ends.push(end - start);
    values.extend_from_slice(&rle.values[first..=last]);
}

pub(crate) fn slice<V: Clone>(rle: RleRef<'_, V>, start: i64, end: i64) -> Rle<V> {
    let mut out = Rle::new();
    slice_into(rle, start, end, &mut out.ends, &mut out.values);
    out
}

pub(crate) fn getitems<V: Clone>(
    rle: RleRef<'_, V>,
    starts: &[i64],
    ends: &[i64],
) -> Result<RleBatch<usize, V>> {
    validate_intervals(starts, ends)?;
    Ok(RleBatch::from_slices(
        (0..starts.len()).collect(),
        |k, out_ends, out_values| slice_into(rle, starts[k], ends[k], out_ends, out_values),
    ))
}
