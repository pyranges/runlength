//! Property tests for the v2 operations, checked against dense vectors.

use proptest::prelude::*;
use runlength::prelude::*;
use runlength::{
    concat, coverage_max, coverage_max_by_key, coverage_min, overlap_plan, plan_many, zip_many,
};

fn rle() -> impl Strategy<Value = Rle<i64>> {
    prop::collection::vec((0i64..4, 0i64..3), 0..8).prop_map(|pairs| {
        let (runs, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Rle::from_runs(&runs, values).unwrap()
    })
}

fn intervals() -> impl Strategy<Value = Vec<(i64, i64, i64)>> {
    prop::collection::vec((0i64..12, 0i64..5, -2i64..4), 0..8)
        .prop_map(|v| v.into_iter().map(|(s, len, w)| (s, s + len, w)).collect())
}

fn covering(iv: &[(i64, i64, i64)], pos: i64) -> Vec<usize> {
    (0..iv.len())
        .filter(|&i| iv[i].0 <= pos && pos < iv[i].1)
        .collect()
}

proptest! {
    #[test]
    fn shift_matches_dense(a in rle(), distance in -12i64..12) {
        let dense = a.to_dense();
        let n = dense.len() as i64;
        let expected: Vec<i64> = (0..n)
            .map(|i| {
                let source = i - distance;
                if (0..n).contains(&source) { dense[source as usize] } else { 9 }
            })
            .collect();
        let shifted = a.shift(distance, 9);
        prop_assert_eq!(shifted.to_dense(), expected);
        prop_assert!(shifted.is_merged());
    }

    #[test]
    fn reversed_matches_dense(a in rle()) {
        let mut dense = a.to_dense();
        dense.reverse();
        prop_assert_eq!(a.reversed().to_dense(), dense);
        prop_assert_eq!(a.reversed().reversed(), a);
    }

    #[test]
    fn concat_matches_dense(parts in prop::collection::vec(rle(), 0..4)) {
        let joined = concat(parts.iter().map(|p| p.view())).unwrap();
        let dense: Vec<i64> = parts.iter().flat_map(|p| p.to_dense()).collect();
        prop_assert_eq!(joined.to_dense(), dense);
        prop_assert!(joined.is_merged());
    }

    #[test]
    fn ranges_where_matches_dense(a in rle()) {
        let (starts, ends) = a.ranges_where(|&v| v > 0);
        let dense = a.to_dense();
        let mut expected = vec![false; dense.len()];
        for (&s, &e) in starts.iter().zip(&ends) {
            for x in &mut expected[s as usize..e as usize] { *x = true; }
        }
        prop_assert_eq!(expected, dense.iter().map(|&v| v > 0).collect::<Vec<_>>());
        // Ranges never touch: touching ones are joined.
        prop_assert!(starts.iter().skip(1).zip(&ends).all(|(s, e)| s > e));
    }

    #[test]
    fn zip_many_matches_dense(parts in prop::collection::vec(rle(), 1..5)) {
        let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
        let total = zip_many(&views, Some(&0), |v| v.iter().copied().sum::<i64>()).unwrap();
        let n = parts.iter().map(|p| p.len()).max().unwrap_or(0) as usize;
        let mut expected = vec![0; n];
        for p in &parts {
            for (i, v) in p.to_dense().into_iter().enumerate() { expected[i] += v; }
        }
        prop_assert_eq!(total.to_dense(), expected);

        let ends: Vec<&[i64]> = parts.iter().map(|p| p.ends()).collect();
        let plan = plan_many(&ends);
        let mut union: Vec<i64> = ends.iter().flat_map(|e| e.iter().copied()).collect();
        union.sort();
        union.dedup();
        prop_assert_eq!(plan.ends, union);
        prop_assert_eq!(plan.idx.len(), parts.len());

        let strict = zip_many(&views, None, |v| v.len());
        prop_assert_eq!(strict.is_ok(), parts.iter().all(|p| p.len() == parts[0].len()));
    }

    #[test]
    fn coverage_extremes_match_dense(iv in intervals(), length in 0i64..25) {
        let starts: Vec<i64> = iv.iter().map(|x| x.0).collect();
        let ends: Vec<i64> = iv.iter().map(|x| x.1).collect();
        let weights: Vec<i64> = iv.iter().map(|x| x.2).collect();
        let n = ends.iter().copied().max().unwrap_or(0);
        let max = coverage_max(&starts, &ends, &weights, None).unwrap();
        let min = coverage_min(&starts, &ends, &weights, None).unwrap();
        prop_assert!(max.is_merged() && min.is_merged());
        let expected_max: Vec<Option<i64>> = (0..n).map(|p| covering(&iv, p).iter().map(|&i| iv[i].2).max()).collect();
        let expected_min: Vec<Option<i64>> = (0..n).map(|p| covering(&iv, p).iter().map(|&i| iv[i].2).min()).collect();
        prop_assert_eq!(max.to_dense(), expected_max);
        prop_assert_eq!(min.to_dense(), expected_min);

        let padded = coverage_max(&starts, &ends, &weights, Some(length));
        prop_assert_eq!(padded.is_ok(), length >= n);
    }

    #[test]
    fn overlap_plan_matches_dense(iv in intervals()) {
        let starts: Vec<i64> = iv.iter().map(|x| x.0).collect();
        let ends: Vec<i64> = iv.iter().map(|x| x.1).collect();
        let plan = overlap_plan(&starts, &ends, None).unwrap();
        let n = ends.iter().copied().max().unwrap_or(0);
        prop_assert_eq!(plan.ends.last().copied().unwrap_or(0), n);
        let mut start = 0;
        for s in 0..plan.num_segments() {
            for pos in start..plan.ends[s] {
                prop_assert_eq!(plan.ids(s), &covering(&iv, pos)[..]);
            }
            start = plan.ends[s];
        }
    }

    #[test]
    fn collapse_matches_per_key_sums(entries in prop::collection::vec(((0u8..3, any::<bool>()), rle()), 0..6)) {
        let mut seen = vec![];
        let entries: Vec<_> = entries.into_iter().filter(|(k, _)| if seen.contains(k) { false } else { seen.push(*k); true }).collect();
        let batch = RleBatch::from_rles(entries.clone()).unwrap();
        let collapsed = batch.collapse(|(chrom, _)| *chrom, Some(&0), |a, b| a + b).unwrap();
        for (chrom, rle) in collapsed.iter() {
            let members: Vec<&Rle<i64>> = entries.iter().filter(|((c, _), _)| c == chrom).map(|(_, r)| r).collect();
            let expected = members.iter().skip(1).fold(members[0].clone(), |acc, r| &acc + *r);
            prop_assert!(rle.equals(&expected));
        }
    }
}

#[test]
fn coverage_max_works_for_strings_and_nan() {
    let names = ["b", "a", "c"].map(String::from);
    let max = coverage_max(&[0, 1, 3], &[2, 4, 5], &names, None).unwrap();
    let dense: Vec<Option<String>> = max.to_dense();
    let expected = ["b", "b", "a", "c", "c"].map(|s| Some(s.to_string()));
    assert_eq!(dense, expected);

    let floats = coverage_max(&[0, 0, 2], &[2, 1, 3], &[f64::NAN, 1.0, f64::NAN], None).unwrap();
    let dense = floats.to_dense();
    assert_eq!(dense[0], Some(1.0));
    assert!(dense[1].unwrap().is_nan() && dense[2].unwrap().is_nan());
}

#[test]
fn coverage_max_by_key_groups() {
    let batch = coverage_max_by_key(&["x", "y", "x"], &[0, 0, 1], &[2, 1, 3], &[1, 5, 3]).unwrap();
    assert_eq!(batch.keys(), &["x", "y"]);
    assert_eq!(
        batch.get(&"x").unwrap().to_dense(),
        vec![Some(1), Some(3), Some(3)]
    );
}

#[test]
fn concat_rejects_overflowing_lengths() {
    let long = Rle::from_runs(&[i64::MAX], vec![1]).unwrap();
    let one = Rle::from_runs(&[1], vec![2]).unwrap();
    assert_eq!(
        concat([long.view(), one.view()]).unwrap_err(),
        runlength::Error::Overflow
    );
}

#[test]
fn float_coverage_keeps_touching_intervals_apart() {
    let cov = runlength::coverage_sum(&[0, 1], &[1, 2], &[1e16, 1.0], None).unwrap();
    assert_eq!(cov.to_dense(), vec![1e16, 1.0]);
}

#[test]
fn nonfinite_weights_only_affect_their_own_intervals() {
    let nan = runlength::coverage_sum(&[0, 0], &[1, 3], &[f64::NAN, 1.0], None).unwrap();
    let dense = nan.to_dense();
    assert!(dense[0].is_nan());
    assert_eq!(dense[1..], [1.0, 1.0]);

    let inf = f64::INFINITY;
    let cov = runlength::coverage_sum(&[0, 0, 1], &[1, 3, 2], &[inf, 1.0, -inf], None).unwrap();
    assert_eq!(cov.to_dense()[0], inf);
    assert_eq!(cov.to_dense()[1], -inf);
    assert_eq!(cov.to_dense()[2], 1.0);

    let both = runlength::coverage_sum(&[0, 0], &[2, 1], &[inf, -inf], None).unwrap();
    assert!(both.to_dense()[0].is_nan());
    assert_eq!(both.to_dense()[1], inf);
}
