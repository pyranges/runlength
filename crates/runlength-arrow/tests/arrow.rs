use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use proptest::prelude::*;
use runlength::prelude::*;
use runlength_arrow::{
    ArrowRle, ArrowRleBatch, RunExtents, coverage, coverage_by_key, identical_neighbours, kernels,
};

fn rle() -> impl Strategy<Value = Rle<i64>> {
    prop::collection::vec((0i64..4, 0i64..3), 0..8).prop_map(|pairs| {
        let (runs, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Rle::from_runs(&runs, values).unwrap()
    })
}

fn strings(values: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

fn keys(chrom: &[&str], strand: &[&str]) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("Chromosome", DataType::Utf8, false),
        Field::new("Strand", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(chrom.to_vec())),
            Arc::new(StringArray::from(strand.to_vec())),
        ],
    )
    .unwrap()
}

proptest! {
    #[test]
    fn arithmetic_matches_core(a in rle(), b in rle()) {
        let (aa, ab) = (ArrowRle::from_rle::<Int64Type>(&a), ArrowRle::from_rle::<Int64Type>(&b));
        prop_assert_eq!(aa.add(&ab).unwrap().to_rle::<Int64Type>().unwrap(), &a + &b);
        prop_assert_eq!(aa.sub(&ab).unwrap().to_rle::<Int64Type>().unwrap(), &a - &b);
        prop_assert_eq!(aa.mul(&ab).unwrap().to_rle::<Int64Type>().unwrap(), &a * &b);

        let lt = aa.elem_lt(&ab).unwrap();
        let expected = a.elem_lt(&b);
        prop_assert_eq!(lt.ends(), expected.ends());
        let got: Vec<bool> = lt.values().as_boolean().iter().map(Option::unwrap).collect();
        prop_assert_eq!(got, expected.values().to_vec());
    }

    #[test]
    fn merge_slice_and_dense_match_core(a in rle(), start in -2i64..20, len in 0i64..20) {
        let aa = ArrowRle::from_rle::<Int64Type>(&a);
        prop_assert_eq!(aa.clone().merge().unwrap().to_rle::<Int64Type>().unwrap(), a.merged());
        prop_assert!(aa.clone().merge().unwrap().is_merged().unwrap());
        prop_assert_eq!(aa.slice(start, start + len).to_rle::<Int64Type>().unwrap(), a.slice(start, start + len));
        let dense = aa.to_dense().unwrap();
        prop_assert_eq!(dense.as_primitive::<Int64Type>().values().to_vec(), a.to_dense());
        prop_assert!(ArrowRle::from_dense(dense).unwrap() == aa);
    }

    #[test]
    fn getitems_matches_core(a in rle(), iv in prop::collection::vec((0i64..20, 0i64..8), 0..6)) {
        let starts: Vec<i64> = iv.iter().map(|x| x.0).collect();
        let ends: Vec<i64> = iv.iter().map(|x| x.0 + x.1).collect();
        let batch = ArrowRle::from_rle::<Int64Type>(&a).getitems(&starts, &ends).unwrap();
        let expected = a.getitems(&starts, &ends).unwrap();
        prop_assert_eq!(batch.len(), expected.len());
        for (got, (_, want)) in batch.iter().zip(expected.iter()) {
            prop_assert_eq!(got.to_rle::<Int64Type>().unwrap(), want.to_rle());
        }
    }

    #[test]
    fn coverage_by_key_matches_core(
        iv in prop::collection::vec((0i64..12, 0i64..5, -2i64..4, 0usize..3), 0..10),
    ) {
        let names = ["chr1", "chr2", "chr10"];
        let starts: Vec<i64> = iv.iter().map(|x| x.0).collect();
        let ends: Vec<i64> = iv.iter().map(|x| x.0 + x.1).collect();
        let weights: Vec<i64> = iv.iter().map(|x| x.2).collect();
        let chrom: Vec<&str> = iv.iter().map(|x| names[x.3]).collect();
        let strand: Vec<&str> = iv.iter().map(|x| if x.2 > 0 { "+" } else { "-" }).collect();

        let weights_array = Int64Array::from(weights.clone());
        let batch = coverage_by_key(&keys(&chrom, &strand), &starts, &ends, Some(&weights_array)).unwrap();
        let key_pairs: Vec<(&str, &str)> = chrom.iter().copied().zip(strand.iter().copied()).collect();
        let expected = runlength::coverage_sum_by_key(&key_pairs, &starts, &ends, &weights).unwrap();
        prop_assert_eq!(batch.len(), expected.len());
        let got_chrom = batch.keys().column(0).as_string::<i32>();
        let got_strand = batch.keys().column(1).as_string::<i32>();
        for (k, (got, (key, want))) in batch.iter().zip(expected.iter()).enumerate() {
            prop_assert_eq!((got_chrom.value(k), got_strand.value(k)), *key);
            prop_assert_eq!(got.to_rle::<Int64Type>().unwrap(), want.to_rle());
        }
    }
}

#[test]
fn float_merging_follows_core_rule() {
    let values = Float64Array::from(vec![
        Some(f64::NAN),
        Some(-f64::NAN),
        Some(0.0),
        Some(-0.0),
        None,
        None,
        Some(1.0),
    ]);
    assert_eq!(
        identical_neighbours(&values).unwrap(),
        vec![true, false, false, false, true, false]
    );
    let rle = ArrowRle::from_dense(Arc::new(values)).unwrap();
    assert_eq!(rle.runs(), vec![2, 1, 1, 2, 1]);
}

#[test]
fn strings_with_nulls() {
    let rle =
        ArrowRle::from_dense(strings(&[Some("a"), Some("a"), None, None, Some("b")])).unwrap();
    assert_eq!(rle.runs(), vec![2, 2, 1]);

    let other =
        ArrowRle::from_dense(strings(&[Some("a"), Some("x"), None, Some("y"), Some("b")])).unwrap();
    let same = rle.zip_with(&other, None, None, kernels::eq).unwrap();
    let values: Vec<Option<bool>> = same.values().as_boolean().iter().collect();
    // Comparisons with null are null, as in SQL.
    assert_eq!(same.runs(), vec![1, 1, 2, 1]);
    assert_eq!(values, vec![Some(true), Some(false), None, Some(true)]);

    // Strings have no default fill, so unequal lengths need an explicit one.
    let short = rle.slice(0, 3);
    assert!(short.elem_eq(&rle).is_err());
    let fill = strings(&[Some("?")]);
    let filled = short
        .zip_with(&rle, Some(fill.as_ref()), None, kernels::ne)
        .unwrap();
    assert_eq!(filled.len(), 5);
}

#[test]
fn min_and_max_of_sortable_types_skip_nulls() {
    let rle = ArrowRle::from_dense(strings(&[Some("b"), None, Some("a"), Some("c")])).unwrap();
    assert_eq!(rle.min().unwrap().as_string::<i32>().value(0), "a");
    assert_eq!(rle.max().unwrap().as_string::<i32>().value(0), "c");

    let dates: ArrayRef = Arc::new(arrow_array::Date32Array::from(vec![20_000, 19_000]));
    let rle = ArrowRle::from_dense(dates).unwrap();
    let min = rle.min().unwrap();
    assert_eq!(min.data_type(), &DataType::Date32);
    assert_eq!(
        min.as_primitive::<arrow_array::types::Date32Type>()
            .value(0),
        19_000
    );

    let nulls = ArrowRle::from_dense(strings(&[None, None])).unwrap();
    assert!(nulls.max().unwrap().is_null(0));
}

#[test]
fn nested_values_merge() {
    let mut builder = ListBuilder::new(Int32Builder::new());
    for list in [vec![1, 2], vec![1, 2], vec![3], vec![3], vec![1, 2]] {
        builder.append_value(list.into_iter().map(Some));
    }
    let rle = ArrowRle::from_dense(Arc::new(builder.finish())).unwrap();
    assert_eq!(rle.runs(), vec![2, 2, 1]);
}

#[test]
fn custom_merge_with_mask_and_keep() {
    let rle =
        ArrowRle::from_runs(&[1, 2, 0, 3], Arc::new(Int64Array::from(vec![1, 2, 9, 3]))).unwrap();
    assert_eq!(rle.runs(), vec![1, 2, 3]);
    let merged = rle.merge_mask(&[true, false], Keep::Last).unwrap();
    assert_eq!(merged.runs(), vec![3, 3]);
    assert_eq!(
        merged
            .values()
            .as_primitive::<Int64Type>()
            .values()
            .to_vec(),
        vec![2, 3]
    );
}

#[test]
fn batch_join_with_multi_column_keys() {
    let ones = |n: i64| ArrowRle::from_runs(&[n], Arc::new(Float64Array::from(vec![1.0]))).unwrap();
    let a = ArrowRleBatch::from_rles(
        keys(&["chr1", "chr2"], &["+", "+"]),
        vec![ones(3), ones(2)],
        &DataType::Float64,
    )
    .unwrap();
    let b = ArrowRleBatch::from_rles(
        keys(&["chr2", "chr1"], &["+", "-"]),
        vec![ones(4), ones(1)],
        &DataType::Float64,
    )
    .unwrap();

    let sum = a.add(&b).unwrap();
    let chrom = sum.keys().column(0).as_string::<i32>();
    let strand = sum.keys().column(1).as_string::<i32>();
    let keys: Vec<(&str, &str)> = (0..sum.len())
        .map(|i| (chrom.value(i), strand.value(i)))
        .collect();
    assert_eq!(keys, vec![("chr1", "+"), ("chr2", "+"), ("chr1", "-")]);
    let dense: Vec<Vec<f64>> = sum
        .iter()
        .map(|r| {
            r.to_dense()
                .unwrap()
                .as_primitive::<Float64Type>()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(
        dense,
        vec![vec![1.0; 3], vec![2.0, 2.0, 1.0, 1.0], vec![1.0]]
    );

    let inner = a.zip_with(&b, How::Inner, None, None, kernels::add);
    assert!(inner.is_err(), "chr2 has lengths 2 and 4 and no fill value");

    let table = sum.to_record_batch().unwrap();
    let names: Vec<String> = table
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, vec!["Chromosome", "Strand", "start", "end", "value"]);
    assert_eq!(table.num_rows(), 4);

    let positions = sum
        .positions(&keys_of(&[("chr1", "-"), ("chrX", "+")]))
        .unwrap();
    assert_eq!(positions, vec![Some(2), None]);
}

fn keys_of(pairs: &[(&str, &str)]) -> RecordBatch {
    let chrom: Vec<&str> = pairs.iter().map(|p| p.0).collect();
    let strand: Vec<&str> = pairs.iter().map(|p| p.1).collect();
    keys(&chrom, &strand)
}

#[test]
fn coverage_keeps_value_type() {
    let starts = [0, 2];
    let ends = [3, 4];
    let counts = coverage(&starts, &ends, None, None).unwrap();
    assert_eq!(counts.data_type(), &DataType::Int64);
    let weights = Float64Array::from(vec![0.5, 1.0]);
    let sums = coverage(&starts, &ends, Some(&weights), Some(6)).unwrap();
    assert_eq!(sums.data_type(), &DataType::Float64);
    let dense = sums.to_dense().unwrap();
    assert_eq!(
        dense.as_primitive::<Float64Type>().values().to_vec(),
        vec![0.5, 0.5, 1.5, 1.0, 0.0, 0.0]
    );

    let strings = StringArray::from(vec!["a", "b"]);
    assert!(coverage(&starts, &ends, Some(&strings), None).is_err());
}

#[test]
fn invalid_input() {
    let values: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2]));
    assert!(ArrowRle::try_new(vec![2, 2], values.clone()).is_err());
    assert!(ArrowRle::from_runs(&[1, -1], values.clone()).is_err());
    let rle = ArrowRle::try_new(vec![1, 2], values).unwrap();
    let bad_fill: ArrayRef = Arc::new(BooleanArray::from(vec![true]));
    assert!(
        rle.zip_with(&rle, Some(bad_fill.as_ref()), None, kernels::add)
            .is_err()
    );
    let duplicate = keys(&["chr1", "chr1"], &["+", "+"]);
    assert!(
        ArrowRleBatch::from_rles(duplicate, vec![rle.clone(), rle], &DataType::UInt64).is_err()
    );
}

#[test]
fn run_table_groups_keys_by_first_appearance() {
    // chr2 + comes first; chr1 +'s rows are split by it.
    let keys = keys_of(&[("chr2", "+"), ("chr1", "+"), ("chr2", "+"), ("chr1", "+")]);
    let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4]));
    let starts = vec![0, 0, 2, 3];
    let ends = vec![2, 3, 5, 4];
    let lengths = vec![2, 3, 3, 1];
    for extents in [
        RunExtents::StartsEnds(&starts, &ends),
        RunExtents::Ends(&ends),
        RunExtents::Lengths(&lengths),
    ] {
        let batch = ArrowRleBatch::from_run_table(&keys, extents, values.clone()).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.offsets(), &[0, 2, 4]);
        assert_eq!(batch.ends(), &[2, 5, 3, 4]);
        let values = batch.values().as_primitive::<Int64Type>().values().to_vec();
        assert_eq!(values, vec![1, 3, 2, 4]);
        let chroms = batch.keys().column(0).as_string::<i32>();
        assert_eq!(
            chroms.iter().flatten().collect::<Vec<_>>(),
            ["chr2", "chr1"]
        );
    }
}

#[test]
fn run_table_round_trips() {
    let a = ArrowRle::from_runs(&[3, 2], strings(&[Some("x"), None])).unwrap();
    let b = ArrowRle::from_runs(&[4], strings(&[Some("y")])).unwrap();
    let batch = ArrowRleBatch::from_rles(
        keys_of(&[("chr1", "+"), ("chr1", "-")]),
        vec![a, b],
        &DataType::Utf8,
    )
    .unwrap();
    let table = batch.to_record_batch().unwrap();
    let column = |name: &str| table.column_by_name(name).unwrap().clone();
    let starts = column("start")
        .as_primitive::<Int64Type>()
        .values()
        .to_vec();
    let ends = column("end").as_primitive::<Int64Type>().values().to_vec();
    let keys = table.project(&[0, 1]).unwrap();
    let back = ArrowRleBatch::from_run_table(
        &keys,
        RunExtents::StartsEnds(&starts, &ends),
        column("value"),
    )
    .unwrap();
    assert_eq!(back.to_record_batch().unwrap(), table);
}

#[test]
fn run_table_drops_empty_runs_and_rejects_bad_extents() {
    let keys = keys_of(&[("chr1", "+"), ("chr1", "+"), ("chr1", "+")]);
    let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
    let batch =
        ArrowRleBatch::from_run_table(&keys, RunExtents::Lengths(&[2, 0, 1]), values.clone())
            .unwrap();
    assert_eq!(batch.ends(), &[2, 3]);
    assert_eq!(
        batch.values().as_primitive::<Int64Type>().values().to_vec(),
        vec![1, 3]
    );

    let gap = ArrowRleBatch::from_run_table(
        &keys,
        RunExtents::StartsEnds(&[0, 2, 5], &[2, 4, 6]),
        values.clone(),
    )
    .unwrap_err();
    assert_eq!(
        gap.to_string(),
        "row 2 starts at 5, expected 4: each key's runs must start at 0 and follow each other without gaps"
    );
    let unsorted =
        ArrowRleBatch::from_run_table(&keys, RunExtents::Ends(&[2, 1, 3]), values.clone());
    assert!(unsorted.is_err());
    let short = ArrowRleBatch::from_run_table(&keys, RunExtents::Lengths(&[1, 1]), values);
    assert!(short.is_err());
}
