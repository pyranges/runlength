//! Slice aggregations, batch transforms and per-key overlap plans, checked
//! against the core crate.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use proptest::prelude::*;
use runlength::prelude::*;
use runlength_arrow::{ArrowRle, ArrowRleBatch, SliceAgg, overlap_plan_by_key};

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

fn names(values: &[&str]) -> RecordBatch {
    let schema = Schema::new(vec![Field::new("name", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(values.to_vec()))],
    )
    .unwrap()
}

fn batch(rles: &[Rle<i64>], key_names: &[&str]) -> ArrowRleBatch {
    let rles = rles.iter().map(ArrowRle::from_rle::<Int64Type>).collect();
    ArrowRleBatch::from_rles(names(key_names), rles, &DataType::Int64).unwrap()
}

fn floats(array: &ArrayRef) -> Vec<Option<f64>> {
    array.as_primitive::<Float64Type>().iter().collect()
}

proptest! {
    #[test]
    fn slice_agg_matches_core(a in rle(), (starts, ends) in slices()) {
        let arrow = ArrowRle::from_rle::<Int64Type>(&a);
        let core = |f: &dyn Fn(SliceView<'_, i64>) -> Option<f64>| -> Vec<Option<f64>> {
            starts.iter().zip(&ends).map(|(&s, &e)| f(a.slice_view(s, e))).collect()
        };
        let agg = |agg| arrow.slice_agg(&starts, &ends, agg).unwrap();

        let sums: Vec<i64> = agg(SliceAgg::Sum).as_primitive::<Int64Type>().values().to_vec();
        let expected: Vec<i64> = starts.iter().zip(&ends).map(|(&s, &e)| a.slice_view(s, e).sum()).collect();
        prop_assert_eq!(sums, expected);
        let counts: Vec<i64> = agg(SliceAgg::Count).as_primitive::<Int64Type>().values().to_vec();
        let expected: Vec<i64> = starts.iter().zip(&ends).map(|(&s, &e)| a.slice_view(s, e).len()).collect();
        prop_assert_eq!(counts, expected);
        prop_assert_eq!(floats(&agg(SliceAgg::Mean)), core(&|s| s.mean()));
        prop_assert_eq!(floats(&agg(SliceAgg::Var { ddof: 1 })), core(&|s| s.variance(1)));
        prop_assert_eq!(floats(&agg(SliceAgg::Std { ddof: 0 })), core(&|s| s.std(0)));
        let mins: Vec<Option<i64>> = agg(SliceAgg::Min).as_primitive::<Int64Type>().iter().collect();
        let expected: Vec<Option<i64>> = starts.iter().zip(&ends).map(|(&s, &e)| a.slice_view(s, e).min().copied()).collect();
        prop_assert_eq!(mins, expected);
    }

    #[test]
    fn batch_slice_agg_matches_single(
        rles in prop::collection::vec(rle(), 3),
        (starts, ends) in slices(),
        picks in prop::collection::vec(0usize..4, 10),
    ) {
        let b = batch(&rles, &["a", "b", "c"]);
        let all = ["a", "b", "c", "missing"];
        let picked: Vec<&str> = picks[..starts.len()].iter().map(|&p| all[p]).collect();
        let means = b.slice_agg(&names(&picked), &starts, &ends, SliceAgg::Mean).unwrap();
        let expected: Vec<Option<f64>> = (0..starts.len())
            .map(|k| match picks[k] {
                3 => None,
                p => rles[p].slice_view(starts[k], ends[k]).mean(),
            })
            .collect();
        prop_assert_eq!(floats(&means), expected);
    }

    #[test]
    fn batch_concat_matches_core(
        first in prop::collection::vec(rle(), 2),
        second in prop::collection::vec(rle(), 2),
    ) {
        let a = batch(&first, &["x", "y"]);
        let b = batch(&second, &["y", "z"]);
        let joined = ArrowRleBatch::concat(&[a, b]).unwrap();
        let expected = [
            first[0].clone(),
            runlength::concat([first[1].view(), second[0].view()]).unwrap(),
            second[1].clone(),
        ];
        prop_assert_eq!(joined.keys(), &names(&["x", "y", "z"]));
        for (rle, want) in joined.iter().zip(&expected) {
            prop_assert_eq!(rle.to_rle::<Int64Type>().unwrap(), want.merged());
        }
    }

    #[test]
    fn batch_shift_matches_core(rles in prop::collection::vec(rle(), 2), distance in -6i64..6) {
        let b = batch(&rles, &["x", "y"]);
        let fill = Int64Array::from(vec![9]);
        let shifted = b.shift(distance, &fill).unwrap();
        let reversed = b.reversed().unwrap();
        for ((s, r), rle) in shifted.iter().zip(reversed.iter()).zip(&rles) {
            prop_assert_eq!(s.to_rle::<Int64Type>().unwrap(), rle.shift(distance, 9));
            prop_assert_eq!(r.to_rle::<Int64Type>().unwrap(), rle.reversed());
        }
    }
}

#[test]
fn slice_agg_skips_nulls() {
    let values: ArrayRef = Arc::new(Float64Array::from(vec![
        Some(2.0),
        None,
        Some(f64::NAN),
        Some(4.0),
    ]));
    let rle = ArrowRle::from_runs(&[2, 3, 1, 2], values).unwrap();
    // Positions: 2 2 _ _ _ NaN 4 4
    let starts = [0, 2, 1, 5];
    let ends = [8, 5, 7, 8];
    let count = rle.slice_agg(&starts, &ends, SliceAgg::Count).unwrap();
    assert_eq!(
        count.as_primitive::<Int64Type>().values().to_vec(),
        vec![5, 0, 3, 3]
    );
    let max = rle.slice_agg(&starts, &ends, SliceAgg::Max).unwrap();
    let max: Vec<Option<f64>> = max.as_primitive::<Float64Type>().iter().collect();
    assert_eq!(max, vec![Some(4.0), None, Some(4.0), Some(4.0)]);
    let mean = rle.slice_agg(&[0, 2], &[2, 5], SliceAgg::Mean).unwrap();
    assert_eq!(floats(&mean), vec![Some(2.0), None]);
}

#[test]
fn slice_agg_on_booleans() {
    let values: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
    let rle = ArrowRle::from_runs(&[3, 1], values).unwrap();
    let mean = rle.slice_agg(&[0], &[4], SliceAgg::Mean).unwrap();
    assert_eq!(floats(&mean), vec![Some(0.75)]);
    let min = rle.slice_agg(&[0, 0], &[4, 2], SliceAgg::Min).unwrap();
    assert_eq!(
        min.as_boolean().iter().collect::<Vec<_>>(),
        vec![Some(false), Some(true)]
    );
    assert!(rle.slice_agg(&[3], &[1], SliceAgg::Sum).is_err());
}

#[test]
fn overlap_plan_by_key_table() {
    let keys = names(&["a", "b", "a"]);
    let table = overlap_plan_by_key(&keys, &[0, 0, 1], &[2, 1, 3]).unwrap();
    assert_eq!(table.num_rows(), 4);
    let name = table.column(0).as_string::<i32>();
    assert_eq!(
        name.iter().collect::<Vec<_>>(),
        vec![Some("a"); 3]
            .into_iter()
            .chain([Some("b")])
            .collect::<Vec<_>>()
    );
    let start = table
        .column(1)
        .as_primitive::<Int64Type>()
        .values()
        .to_vec();
    let end = table
        .column(2)
        .as_primitive::<Int64Type>()
        .values()
        .to_vec();
    assert_eq!((start, end), (vec![0, 1, 2, 0], vec![1, 2, 3, 1]));
    let ids = table.column(3).as_list::<i64>();
    let ids: Vec<Vec<i64>> = ids
        .iter()
        .map(|l| l.unwrap().as_primitive::<Int64Type>().values().to_vec())
        .collect();
    assert_eq!(ids, vec![vec![0], vec![0, 2], vec![2], vec![1]]);
    assert_eq!(table.column(3).len(), 4);
}
