//! Property tests for slice views, batch transforms and per-key overlap
//! plans, checked against the copying and single-sequence versions.

use proptest::prelude::*;
use runlength::prelude::*;
use runlength::{overlap_plan, overlap_plan_by_key};

fn rle() -> impl Strategy<Value = Rle<i64>> {
    prop::collection::vec((0i64..4, -2i64..3), 0..8).prop_map(|pairs| {
        let (runs, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Rle::from_runs(&runs, values).unwrap()
    })
}

fn slices() -> impl Strategy<Value = (Vec<i64>, Vec<i64>)> {
    prop::collection::vec((0i64..20, 0i64..8), 0..10)
        .prop_map(|v| v.into_iter().map(|(s, len)| (s, s + len)).unzip())
}

fn batch() -> impl Strategy<Value = RleBatch<u8, i64>> {
    prop::collection::vec(rle(), 0..4).prop_map(|rles| {
        RleBatch::from_rles(rles.into_iter().enumerate().map(|(k, r)| (k as u8 * 2, r))).unwrap()
    })
}

proptest! {
    #[test]
    fn slice_view_matches_slice(a in rle(), start in 0i64..20, len in 0i64..10) {
        let view = a.slice_view(start, start + len);
        let copy = a.slice(start, start + len);
        prop_assert_eq!(view.to_rle().to_dense(), copy.to_dense());
        prop_assert_eq!(view.len(), copy.len());
        prop_assert_eq!(view.num_runs(), copy.num_runs());
        prop_assert_eq!(view.sum(), copy.sum());
        prop_assert_eq!(view.mean(), copy.mean());
        prop_assert_eq!(view.variance(1), copy.variance(1));
        prop_assert_eq!(view.min(), copy.min());
        prop_assert_eq!(view.max(), copy.max());
    }

    #[test]
    fn slice_reduce_matches_getitems(a in rle(), (starts, ends) in slices()) {
        let means = a.slice_reduce(&starts, &ends, |s| s.mean()).unwrap();
        let copies = a.getitems(&starts, &ends).unwrap();
        let expected: Vec<Option<f64>> = copies.iter().map(|(_, r)| r.mean()).collect();
        prop_assert_eq!(means, expected);
    }

    #[test]
    fn batch_slice_reduce_matches_getitems(
        b in batch(),
        (starts, ends) in slices(),
        picks in prop::collection::vec(0u8..8, 10),
    ) {
        let keys = &picks[..starts.len()];
        let sums = b.slice_reduce(keys, &starts, &ends, |s| s.sum()).unwrap();
        let copies = b.getitems(keys, &starts, &ends).unwrap();
        let expected: Vec<i64> = copies.iter().map(|(_, r)| r.sum()).collect();
        prop_assert_eq!(sums, expected);
    }

    #[test]
    fn batch_concat_matches_dense(parts in prop::collection::vec(batch(), 0..3)) {
        let refs: Vec<&RleBatch<u8, i64>> = parts.iter().collect();
        let joined = RleBatch::concat(&refs);
        for (key, rle) in joined.iter() {
            let dense: Vec<i64> = parts
                .iter()
                .filter_map(|p| p.get(key))
                .flat_map(|r| r.to_dense())
                .collect();
            prop_assert_eq!(rle.to_dense(), dense);
            prop_assert!(rle.is_merged());
        }
        let expected = parts.iter().flat_map(|p| p.keys()).fold(Vec::new(), |mut acc, k| {
            if !acc.contains(k) {
                acc.push(*k);
            }
            acc
        });
        prop_assert_eq!(joined.keys(), &expected[..]);
    }

    #[test]
    fn batch_shift_and_reverse_match_each(b in batch(), distance in -6i64..6) {
        let shifted = b.shift(distance, 7);
        let reversed = b.reversed();
        for ((key, rle), ((_, s), (_, r))) in b.iter().zip(shifted.iter().zip(reversed.iter())) {
            prop_assert_eq!(s.to_rle(), rle.shift(distance, 7), "key {}", key);
            prop_assert_eq!(r.to_rle(), rle.reversed());
        }
    }

    #[test]
    fn overlap_plan_by_key_matches_per_key(
        iv in prop::collection::vec((0u8..3, 0i64..12, 0i64..5), 0..10)
    ) {
        let keys: Vec<u8> = iv.iter().map(|t| t.0).collect();
        let starts: Vec<i64> = iv.iter().map(|t| t.1).collect();
        let ends: Vec<i64> = iv.iter().map(|t| t.1 + t.2).collect();
        let (unique, plans) = overlap_plan_by_key(&keys, &starts, &ends).unwrap();
        prop_assert_eq!(unique.len(), plans.len());
        for (key, plan) in unique.iter().zip(&plans) {
            let idx: Vec<usize> = (0..keys.len()).filter(|&i| keys[i] == *key).collect();
            let sub_starts: Vec<i64> = idx.iter().map(|&i| starts[i]).collect();
            let sub_ends: Vec<i64> = idx.iter().map(|&i| ends[i]).collect();
            let expected = overlap_plan(&sub_starts, &sub_ends, None).unwrap();
            prop_assert_eq!(&plan.ends, &expected.ends);
            prop_assert_eq!(&plan.offsets, &expected.offsets);
            let mapped: Vec<usize> = expected.interval_ids.iter().map(|&k| idx[k]).collect();
            prop_assert_eq!(&plan.interval_ids, &mapped);
        }
    }
}

#[test]
fn slice_view_clamps_and_handles_empty() {
    let a = Rle::from_runs(&[3, 2], vec![1.0, 4.0]).unwrap();
    let view = a.slice_view(4, 100);
    assert_eq!(view.len(), 1);
    assert_eq!(view.mean(), Some(4.0));
    let empty = a.slice_view(7, 9);
    assert!(empty.is_empty());
    assert_eq!(empty.mean(), None);
    assert_eq!(empty.sum(), 0.0);
    assert_eq!(empty.max(), None);
    assert!(a.slice_reduce(&[2], &[1], |s| s.len()).is_err());
}
