/// How to combine two sequences, computed from their run ends alone.
///
/// The output segments are the union of both inputs' breakpoints. For each
/// segment, `idx_a` and `idx_b` give the run of each input that covers it. An
/// index equal to the input's number of runs (`fill_a` / `fill_b`) means the
/// segment lies past the end of that input and must be filled. This only
/// happens at the tail, in `[min_len, max_len)`.
///
/// Because no values are involved, a plan works for any value types and can be
/// reused for several operations on the same pair. Applying it is a gather:
/// `f(values_a[idx_a[k]], values_b[idx_b[k]])` for each segment `k`, with the
/// fill value appended to each `values` array.
///
/// ```
/// use runlength::plan;
///
/// // a: runs [3, 2], b: runs [1, 4]
/// let p = plan(&[3, 5], &[1, 5]);
/// assert_eq!(p.ends, vec![1, 3, 5]);
/// assert_eq!(p.idx_a, vec![0, 0, 1]);
/// assert_eq!(p.idx_b, vec![0, 1, 1]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Cumulative ends of the output segments.
    pub ends: Vec<i64>,
    /// For each segment, the run of `a` covering it, or `fill_a`.
    pub idx_a: Vec<usize>,
    /// For each segment, the run of `b` covering it, or `fill_b`.
    pub idx_b: Vec<usize>,
    /// The index that marks a filled segment in `idx_a` (`a`'s run count).
    pub fill_a: usize,
    /// The index that marks a filled segment in `idx_b` (`b`'s run count).
    pub fill_b: usize,
}

impl Plan {
    /// The number of output segments.
    pub fn num_segments(&self) -> usize {
        self.ends.len()
    }

    /// Whether any segment needs a fill value for `a`.
    pub fn needs_fill_a(&self) -> bool {
        self.idx_a.last() == Some(&self.fill_a)
    }

    /// Whether any segment needs a fill value for `b`.
    pub fn needs_fill_b(&self) -> bool {
        self.idx_b.last() == Some(&self.fill_b)
    }

    /// Gathers `a`'s values onto the segments, using `fill` past its end.
    pub fn gather_a<A: Clone>(&self, values: &[A], fill: &A) -> Vec<A> {
        gather(&self.idx_a, values, fill)
    }

    /// Gathers `b`'s values onto the segments, using `fill` past its end.
    pub fn gather_b<B: Clone>(&self, values: &[B], fill: &B) -> Vec<B> {
        gather(&self.idx_b, values, fill)
    }
}

fn gather<T: Clone>(idx: &[usize], values: &[T], fill: &T) -> Vec<T> {
    idx.iter()
        .map(|&i| values.get(i).unwrap_or(fill).clone())
        .collect()
}

/// Computes the [`Plan`] for combining sequences with run ends `a` and `b`.
pub fn plan(a: &[i64], b: &[i64]) -> Plan {
    let capacity = a.len() + b.len();
    let mut out = Plan {
        ends: Vec::with_capacity(capacity),
        idx_a: Vec::with_capacity(capacity),
        idx_b: Vec::with_capacity(capacity),
        fill_a: a.len(),
        fill_b: b.len(),
    };
    for (end, i, j) in Segments::new(a, b) {
        out.ends.push(end);
        out.idx_a.push(i);
        out.idx_b.push(j);
    }
    out
}

/// Walks the union of two sets of run ends, yielding `(end, i, j)` for each
/// output segment. `i == a.len()` (or `j == b.len()`) marks a segment past the
/// end of that side.
pub(crate) struct Segments<'a> {
    a: &'a [i64],
    b: &'a [i64],
    i: usize,
    j: usize,
}

impl<'a> Segments<'a> {
    pub(crate) fn new(a: &'a [i64], b: &'a [i64]) -> Self {
        Segments { a, b, i: 0, j: 0 }
    }
}

impl Iterator for Segments<'_> {
    type Item = (i64, usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // While both sides have runs left, advance without branching on which
        // end comes first: on real data that branch is unpredictable.
        if let (Some(&x), Some(&y)) = (self.a.get(self.i), self.b.get(self.j)) {
            let end = x.min(y);
            let item = (end, self.i, self.j);
            self.i += usize::from(x == end);
            self.j += usize::from(y == end);
            return Some(item);
        }
        let end_a = self.a.get(self.i).copied();
        let end_b = self.b.get(self.j).copied();
        let end = match (end_a, end_b) {
            (None, None) => return None,
            (Some(x), None) | (None, Some(x)) => x,
            (Some(x), Some(y)) => x.min(y),
        };
        let item = (end, self.i, self.j);
        if end_a == Some(end) {
            self.i += 1;
        }
        if end_b == Some(end) {
            self.j += 1;
        }
        Some(item)
    }
}
