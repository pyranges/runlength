use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use proptest::prelude::*;
use runlength::prelude::*;
use runlength_arrow::{
    Agg, ArrowRle, ArrowRleBatch, concat, coverage_agg, coverage_agg_by_key, kernels, zip_many,
};

fn rle() -> impl Strategy<Value = Rle<i64>> {
    prop::collection::vec((0i64..4, 0i64..3), 0..8).prop_map(|pairs| {
        let (runs, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Rle::from_runs(&runs, values).unwrap()
    })
}

fn arrow(r: &Rle<i64>) -> ArrowRle {
    ArrowRle::from_rle::<Int64Type>(r)
}

fn core(r: &ArrowRle) -> Rle<i64> {
    r.to_rle::<Int64Type>().unwrap()
}

fn options(r: &ArrowRle) -> Vec<Option<i64>> {
    r.to_dense()
        .unwrap()
        .as_primitive::<Int64Type>()
        .iter()
        .collect()
}

fn one(v: i64) -> ArrayRef {
    Arc::new(Int64Array::from(vec![v]))
}

proptest! {
    #[test]
    fn shift_reverse_concat_match_core(a in rle(), b in rle(), distance in -10i64..10) {
        prop_assert_eq!(core(&arrow(&a).shift(distance, one(7).as_ref()).unwrap()), a.shift(distance, 7));
        prop_assert_eq!(core(&arrow(&a).reversed().unwrap()), a.reversed());
        let joined = concat(&[arrow(&a), arrow(&b)]).unwrap();
        prop_assert_eq!(core(&joined), runlength::concat([a.view(), b.view()]));
    }

    #[test]
    fn zip_many_matches_core(parts in prop::collection::vec(rle(), 1..4)) {
        let arrows: Vec<ArrowRle> = parts.iter().map(arrow).collect();
        let fill = one(0);
        let fills = vec![Some(fill.as_ref()); parts.len()];
        let total = zip_many(&arrows, &fills, |columns| {
            columns.iter().skip(1).try_fold(columns[0].clone(), |acc, c| kernels::add(acc.as_ref(), c.as_ref()))
        }).unwrap();
        let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
        let expected = runlength::zip_many(&views, Some(&0), |v| v.iter().copied().sum::<i64>()).unwrap();
        prop_assert_eq!(core(&total), expected);
    }

    #[test]
    fn coverage_agg_matches_core(iv in prop::collection::vec((0i64..12, 0i64..5, -2i64..4), 0..8)) {
        let starts: Vec<i64> = iv.iter().map(|x| x.0).collect();
        let ends: Vec<i64> = iv.iter().map(|x| x.0 + x.1).collect();
        let weights: Vec<i64> = iv.iter().map(|x| x.2).collect();
        let values = Int64Array::from(weights.clone());
        let max = coverage_agg(&starts, &ends, Some(&values), None, Agg::Max).unwrap();
        prop_assert_eq!(options(&max), runlength::coverage_max(&starts, &ends, &weights, None).unwrap().to_dense());
        let min = coverage_agg(&starts, &ends, Some(&values), None, Agg::Min).unwrap();
        prop_assert_eq!(options(&min), runlength::coverage_min(&starts, &ends, &weights, None).unwrap().to_dense());
        let sum = coverage_agg(&starts, &ends, Some(&values), None, Agg::Sum).unwrap();
        prop_assert_eq!(options(&sum), runlength::coverage_sum_opt(&starts, &ends, &weights, None).unwrap().to_dense());
        let filled = sum.fill_null(one(0).as_ref()).unwrap();
        prop_assert_eq!(core(&filled), runlength::coverage_sum(&starts, &ends, &weights, None).unwrap());
    }
}

#[test]
fn true_ranges_of_boolean_sequences() {
    let values: ArrayRef = Arc::new(BooleanArray::from(vec![
        Some(true),
        None,
        Some(true),
        Some(false),
        Some(true),
    ]));
    let rle = ArrowRle::from_runs(&[2, 1, 1, 1, 3], values).unwrap();
    assert_eq!(rle.true_ranges().unwrap(), (vec![0, 3, 5], vec![2, 4, 8]));
    assert!(
        arrow(&Rle::from_runs(&[1], vec![1]).unwrap())
            .true_ranges()
            .is_err()
    );
}

#[test]
fn string_max_coverage_with_nulls_where_uncovered() {
    let names = StringArray::from(vec!["b", "a", "c"]);
    let max = coverage_agg(&[0, 1, 4], &[2, 3, 5], Some(&names), None, Agg::Max).unwrap();
    let dense = max.to_dense().unwrap();
    let dense: Vec<Option<&str>> = dense.as_string::<i32>().iter().collect();
    assert_eq!(
        dense,
        vec![Some("b"), Some("b"), Some("a"), None, Some("c")]
    );
}

#[test]
fn coverage_agg_by_key_and_collapse() {
    let schema = Schema::new(vec![
        Field::new("Chromosome", DataType::Utf8, false),
        Field::new("Strand", DataType::Utf8, false),
    ]);
    let keys = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec!["chr1", "chr1", "chr2", "chr1"])),
            Arc::new(StringArray::from(vec!["+", "-", "+", "+"])),
        ],
    )
    .unwrap();
    let starts = [0, 1, 0, 2];
    let ends = [3, 4, 2, 5];
    let values = Int64Array::from(vec![1, 2, 3, 4]);

    let max: ArrowRleBatch =
        coverage_agg_by_key(&keys, &starts, &ends, Some(&values), Agg::Max).unwrap();
    assert_eq!(max.len(), 3);
    assert_eq!(
        options(&max.get_index(0).unwrap()),
        vec![Some(1), Some(1), Some(4), Some(4), Some(4)]
    );
    assert!(coverage_agg_by_key(&keys, &starts, &ends, None, Agg::Max).is_err());

    let counts = coverage_agg_by_key(&keys, &starts, &ends, None, Agg::Count).unwrap();
    let per_chromosome = counts
        .collapse(&["Chromosome"], runlength_arrow::kernels::Kernel::Add)
        .unwrap();
    assert_eq!(per_chromosome.keys().num_columns(), 1);
    assert_eq!(per_chromosome.len(), 2);
    assert_eq!(
        core(&per_chromosome.get_index(0).unwrap()).to_dense(),
        vec![1, 2, 3, 2, 1]
    );
    assert!(
        counts
            .collapse(&["Nope"], runlength_arrow::kernels::Kernel::Add)
            .is_err()
    );
}
