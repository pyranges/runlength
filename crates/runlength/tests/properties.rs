//! Property tests: every operation is checked against the same operation on
//! plain dense vectors.

use proptest::prelude::*;
use runlength::prelude::*;
use runlength::{
    coverage_count, coverage_count_by_key, coverage_sum, coverage_sum_by_key, coverage_sum_opt,
};

/// Runs and values with small ranges, so zero-length runs and identical
/// neighbours are common.
fn runs_values() -> impl Strategy<Value = (Vec<i64>, Vec<i64>)> {
    prop::collection::vec((0i64..4, 0i64..3), 0..8).prop_map(|pairs| pairs.into_iter().unzip())
}

fn rle() -> impl Strategy<Value = Rle<i64>> {
    runs_values().prop_map(|(runs, values)| Rle::from_runs(&runs, values).unwrap())
}

fn expand(runs: &[i64], values: &[i64]) -> Vec<i64> {
    runs.iter()
        .zip(values)
        .flat_map(|(&r, &v)| std::iter::repeat_n(v, r as usize))
        .collect()
}

fn padded(v: &[i64], n: usize) -> Vec<i64> {
    let mut v = v.to_vec();
    v.resize(n, 0);
    v
}

fn zip_dense(a: &[i64], b: &[i64], f: impl Fn(i64, i64) -> i64) -> Vec<i64> {
    let n = a.len().max(b.len());
    padded(a, n)
        .into_iter()
        .zip(padded(b, n))
        .map(|(x, y)| f(x, y))
        .collect()
}

fn intervals() -> impl Strategy<Value = Vec<(i64, i64, i64)>> {
    prop::collection::vec((0i64..12, 0i64..5, -2i64..4), 0..8)
        .prop_map(|v| v.into_iter().map(|(s, len, w)| (s, s + len, w)).collect())
}

fn dense_coverage(iv: &[(i64, i64, i64)], weighted: bool) -> Vec<i64> {
    let n = iv.iter().map(|&(_, e, _)| e).max().unwrap_or(0) as usize;
    let mut out = vec![0; n];
    for &(s, e, w) in iv {
        for x in &mut out[s as usize..e as usize] {
            *x += if weighted { w } else { 1 };
        }
    }
    out
}

proptest! {
    #[test]
    fn construction_round_trips((runs, values) in runs_values()) {
        let dense = expand(&runs, &values);
        let r = Rle::from_runs(&runs, values).unwrap();
        prop_assert_eq!(r.to_dense(), dense.clone());
        prop_assert!(r.runs().iter().all(|&x| x > 0));
        prop_assert_eq!(r.len(), dense.len() as i64);

        let d = Rle::from_dense(dense.clone());
        prop_assert!(d.is_merged());
        prop_assert_eq!(d.to_dense(), dense);
        prop_assert!(d.equals(&r));
        prop_assert!(r.merged().is_merged());
        prop_assert_eq!(r.clone().merge(), d);
    }

    #[test]
    fn arithmetic_matches_dense(a in rle(), b in rle()) {
        let (da, db) = (a.to_dense(), b.to_dense());
        let sum = &a + &b;
        prop_assert!(sum.is_merged());
        prop_assert_eq!(sum.to_dense(), zip_dense(&da, &db, |x, y| x + y));
        prop_assert_eq!((&a - &b).to_dense(), zip_dense(&da, &db, |x, y| x - y));
        prop_assert_eq!((&a * &b).to_dense(), zip_dense(&da, &db, |x, y| x * y));
        prop_assert_eq!((-&a).to_dense(), da.iter().map(|x| -x).collect::<Vec<_>>());
    }

    #[test]
    fn comparisons_match_dense(a in rle(), b in rle()) {
        let (da, db) = (a.to_dense(), b.to_dense());
        let n = da.len().max(db.len());
        let (pa, pb) = (padded(&da, n), padded(&db, n));
        let dense_cmp = |f: fn(&i64, &i64) -> bool| pa.iter().zip(&pb).map(|(x, y)| f(x, y)).collect::<Vec<_>>();
        prop_assert_eq!(a.elem_lt(&b).to_dense(), dense_cmp(|x, y| x < y));
        prop_assert_eq!(a.elem_ge(&b).to_dense(), dense_cmp(|x, y| x >= y));
        prop_assert_eq!(a.elem_eq(&b).to_dense(), dense_cmp(|x, y| x == y));
    }

    #[test]
    fn strict_zip_requires_equal_length(a in rle(), b in rle()) {
        let result = a.zip_with(&b, None, None, |x, y| x + y);
        prop_assert_eq!(result.is_ok(), a.len() == b.len());
    }

    #[test]
    fn plan_is_union_of_breakpoints_and_gathers_correctly(a in rle(), b in rle()) {
        let p = a.plan_with(&b);
        let mut expected: Vec<i64> = a.ends().iter().chain(b.ends()).copied().collect();
        expected.sort();
        expected.dedup();
        prop_assert_eq!(&p.ends, &expected);

        let raw = Rle::from_ends(p.ends.clone(), {
            let va = p.gather_a(a.values(), &0);
            let vb = p.gather_b(b.values(), &0);
            va.into_iter().zip(vb).map(|(x, y)| x * 10 + y).collect()
        }).unwrap();
        prop_assert_eq!(raw.to_dense(), zip_dense(&a.to_dense(), &b.to_dense(), |x, y| x * 10 + y));
        prop_assert_eq!(p.needs_fill_a(), a.len() < b.len());
        prop_assert_eq!(p.needs_fill_b(), b.len() < a.len());
    }

    #[test]
    fn slicing_matches_dense(a in rle(), start in -2i64..25, len in 0i64..25) {
        let end = start + len;
        let dense = a.to_dense();
        let (s, e) = (start.clamp(0, dense.len() as i64) as usize, end.clamp(0, dense.len() as i64) as usize);
        let expected = if s < e { dense[s..e].to_vec() } else { vec![] };
        prop_assert_eq!(a.slice(start, end).to_dense(), expected);
        if (0..a.len()).contains(&start) {
            prop_assert_eq!(a.get(start), Some(&dense[start as usize]));
        } else {
            prop_assert_eq!(a.get(start), None);
        }
    }

    #[test]
    fn getitems_matches_slices(a in rle(), iv in prop::collection::vec((0i64..25, 0i64..10), 0..6)) {
        let starts: Vec<i64> = iv.iter().map(|x| x.0).collect();
        let ends: Vec<i64> = iv.iter().map(|x| x.0 + x.1).collect();
        let batch = a.getitems(&starts, &ends).unwrap();
        prop_assert_eq!(batch.len(), iv.len());
        for (k, (key, r)) in batch.iter().enumerate() {
            prop_assert_eq!(*key, k);
            prop_assert_eq!(r.to_rle(), a.slice(starts[k], ends[k]));
        }
    }

    #[test]
    fn merge_mask_matches_model(
        a in rle(),
        bits in prop::collection::vec(any::<bool>(), 0..40),
        last in any::<bool>(),
    ) {
        let n = a.num_runs();
        let mask: Vec<bool> = (0..n.saturating_sub(1)).map(|i| bits.get(i).copied().unwrap_or(false)).collect();
        let keep = if last { Keep::Last } else { Keep::First };

        // Model: group runs, give each group its first or last value.
        let plan = a.merge_plan(&mask).unwrap();
        let runs = a.runs();
        let mut dense = vec![];
        for g in plan.offsets.windows(2) {
            let value = a.values()[if last { g[1] - 1 } else { g[0] }];
            let length: i64 = runs[g[0]..g[1]].iter().sum();
            dense.extend(std::iter::repeat_n(value, length as usize));
        }
        let merged = a.clone().merge_mask(&mask, keep).unwrap();
        prop_assert_eq!(merged.ends(), &plan.ends[..]);
        prop_assert_eq!(merged.to_dense(), dense);
        prop_assert_eq!(plan.group_ids().len(), n);
    }

    #[test]
    fn reductions_match_dense(a in rle()) {
        let dense = a.to_dense();
        prop_assert_eq!(a.sum(), dense.iter().sum::<i64>());
        prop_assert_eq!(a.min(), dense.iter().min());
        prop_assert_eq!(a.max(), dense.iter().max());
        let n = dense.len() as f64;
        match a.mean() {
            None => prop_assert!(dense.is_empty()),
            Some(m) => {
                let expected = dense.iter().sum::<i64>() as f64 / n;
                prop_assert!((m - expected).abs() < 1e-9);
                let var: f64 = dense.iter().map(|&x| (x as f64 - expected).powi(2)).sum::<f64>();
                if dense.len() > 1 {
                    prop_assert!((a.variance(1).unwrap() - var / (n - 1.0)).abs() < 1e-9);
                }
                prop_assert!((a.variance(0).unwrap() - var / n).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn coverage_matches_dense(iv in intervals()) {
        let starts: Vec<i64> = iv.iter().map(|x| x.0).collect();
        let ends: Vec<i64> = iv.iter().map(|x| x.1).collect();
        let weights: Vec<i64> = iv.iter().map(|x| x.2).collect();

        let sum = coverage_sum(&starts, &ends, &weights, None).unwrap();
        let count = coverage_count(&starts, &ends, None).unwrap();
        prop_assert!(sum.is_merged() && count.is_merged());
        prop_assert_eq!(sum.to_dense(), dense_coverage(&iv, true));
        prop_assert_eq!(count.to_dense(), dense_coverage(&iv, false));

        // Uncovered positions are None, covered ones carry the sum.
        let opt = coverage_sum_opt(&starts, &ends, &weights, Some(20)).unwrap();
        prop_assert_eq!(opt.len(), 20);
        let dense_count = padded(&count.to_dense(), 20);
        let dense_sum = padded(&sum.to_dense(), 20);
        for (i, v) in opt.to_dense().into_iter().enumerate() {
            prop_assert_eq!(v, (dense_count[i] > 0).then_some(dense_sum[i]));
        }
    }

    #[test]
    fn coverage_by_key_matches_per_key(iv in intervals(), raw_keys in prop::collection::vec(0u8..3, 8)) {
        let keys: Vec<u8> = raw_keys[..iv.len()].to_vec();
        let starts: Vec<i64> = iv.iter().map(|x| x.0).collect();
        let ends: Vec<i64> = iv.iter().map(|x| x.1).collect();
        let weights: Vec<i64> = iv.iter().map(|x| x.2).collect();

        let sums = coverage_sum_by_key(&keys, &starts, &ends, &weights).unwrap();
        let counts = coverage_count_by_key(&keys, &starts, &ends).unwrap();
        let mut seen = vec![];
        for k in &keys { if !seen.contains(k) { seen.push(*k); } }
        prop_assert_eq!(sums.keys(), &seen[..]);
        for key in &seen {
            let mine: Vec<_> = iv.iter().zip(&keys).filter(|(_, k)| *k == key).map(|(x, _)| *x).collect();
            prop_assert_eq!(sums.get(key).unwrap().to_dense(), dense_coverage(&mine, true));
            prop_assert_eq!(counts.get(key).unwrap().to_dense(), dense_coverage(&mine, false));
        }
    }

    #[test]
    fn batch_add_matches_per_key(
        left in prop::collection::vec((0u8..4, rle()), 0..4),
        right in prop::collection::vec((0u8..4, rle()), 0..4),
    ) {
        let dedup = |v: Vec<(u8, Rle<i64>)>| {
            let mut seen = vec![];
            v.into_iter().filter(|(k, _)| if seen.contains(k) { false } else { seen.push(*k); true }).collect::<Vec<_>>()
        };
        let (left, right) = (dedup(left), dedup(right));
        let a = RleBatch::from_rles(left.clone()).unwrap();
        let b = RleBatch::from_rles(right.clone()).unwrap();

        let sum = &a + &b;
        let mut expected_keys: Vec<u8> = left.iter().map(|x| x.0).collect();
        expected_keys.extend(right.iter().map(|x| x.0).filter(|k| !left.iter().any(|l| l.0 == *k)));
        prop_assert_eq!(sum.keys(), &expected_keys[..]);
        for key in &expected_keys {
            let ra = a.get(key).map(|r| r.to_rle()).unwrap_or_default();
            let rb = b.get(key).map(|r| r.to_rle()).unwrap_or_default();
            prop_assert_eq!(sum.get(key).unwrap().to_rle(), &ra + &rb);
        }

        let inner = a.zip_with(&b, How::Inner, None, None, |x, y| x + y);
        let keys_equal_len = left.iter().all(|(k, r)| b.get(k).is_none_or(|o| o.len() == r.len()));
        prop_assert_eq!(inner.is_ok(), keys_equal_len);
        if let Ok(inner) = inner {
            prop_assert!(inner.keys().iter().all(|k| a.get(k).is_some() && b.get(k).is_some()));
        }
    }
}

#[test]
fn nan_and_signed_zero_merging() {
    let r = Rle::from_dense(vec![f64::NAN, f64::NAN, 0.0, -0.0, 1.0, 1.0]);
    assert_eq!(r.runs(), vec![2, 1, 1, 2]);
    let opt = Rle::from_dense(vec![None, None, Some(f64::NAN), Some(f64::NAN)]);
    assert_eq!(opt.runs(), vec![2, 2]);
}

#[test]
fn merge_by_tolerance_chains() {
    let r = Rle::from_runs(&[1, 1, 1, 1], vec![1.0, 1.05, 1.1, 5.0]).unwrap();
    let m = r.merge_by(|a: &f64, b: &f64| (a - b).abs() < 0.1);
    assert_eq!(m.runs(), vec![3, 1]);
    assert_eq!(m.values(), &[1.0, 5.0]);
}

#[test]
fn float_coverage_has_no_residue_in_gaps() {
    let cov = coverage_sum(&[0, 0, 5], &[2, 2, 6], &[0.1, 0.2, 1.0], None).unwrap();
    assert_eq!(cov.to_dense()[2..5], [0.0, 0.0, 0.0]);
    assert!(cov.is_merged());
}

#[test]
fn generic_values() {
    let a = Rle::from_runs(&[2, 2], vec!["x".to_string(), "y".to_string()]).unwrap();
    let b = Rle::from_runs(&[1, 3], vec![1, 2]).unwrap();
    let c = a
        .zip_with(&b, None, None, |s, n| s.repeat(*n as usize))
        .unwrap();
    assert_eq!(c.to_dense(), vec!["x", "xx", "yy", "yy"]);
    assert_eq!(c.runs(), vec![1, 1, 2]);
}

#[test]
fn invalid_input_is_rejected() {
    assert!(Rle::from_runs(&[1, -1], vec![0, 0]).is_err());
    assert!(Rle::from_runs(&[1], vec![0, 0]).is_err());
    assert!(Rle::from_ends(vec![2, 2], vec![0, 1]).is_err());
    assert!(coverage_count(&[3], &[2], None).is_err());
    assert!(coverage_count(&[0], &[5], Some(4)).is_err());
    assert!(RleBatch::from_rles(vec![(1, Rle::<i64>::new()), (1, Rle::new())]).is_err());
}
