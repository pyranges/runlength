//! Checks runlength against cases generated with pyrle by
//! `scripts/gen_pyrle_fixtures.py`.

use std::collections::HashMap;

use runlength::coverage_sum;
use runlength::prelude::*;

const CASES: &str = include_str!("fixtures/pyrle_cases.txt");

fn parse() -> Vec<(String, HashMap<String, Vec<i64>>)> {
    CASES
        .split("\n\n")
        .filter(|block| !block.trim().is_empty())
        .map(|block| {
            let mut lines = block.lines();
            let op = lines
                .next()
                .unwrap()
                .strip_prefix("case ")
                .unwrap()
                .to_string();
            let fields = lines
                .map(|l| {
                    let mut parts = l.split_whitespace();
                    let name = parts.next().unwrap().to_string();
                    (name, parts.map(|x| x.parse().unwrap()).collect())
                })
                .collect();
            (op, fields)
        })
        .collect()
}

fn rle(fields: &HashMap<String, Vec<i64>>, prefix: &str) -> Rle<i64> {
    Rle::from_runs(
        &fields[&format!("{prefix}_runs")],
        fields[&format!("{prefix}_values")].clone(),
    )
    .unwrap()
}

#[test]
fn matches_pyrle() {
    let cases = parse();
    assert!(cases.len() > 100);
    for (n, (op, f)) in cases.iter().enumerate() {
        let expected = rle(f, "out");
        let actual = match op.as_str() {
            "add" => &rle(f, "a") + &rle(f, "b"),
            "sub" => &rle(f, "a") - &rle(f, "b"),
            "mul" => &rle(f, "a") * &rle(f, "b"),
            "coverage" => coverage_sum(&f["starts"], &f["ends"], &f["weights"], None).unwrap(),
            other => panic!("unknown op {other}"),
        };
        assert_eq!(
            actual.to_dense(),
            expected.to_dense(),
            "case {n} ({op}): {f:?}"
        );
        // pyrle merges its results too, so the runs should agree exactly.
        assert_eq!(
            actual.runs(),
            expected.merged().runs(),
            "case {n} ({op}) runs: {f:?}"
        );
    }
}
