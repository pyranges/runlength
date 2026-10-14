import numpy as np
import polars as pl
import pyarrow as pa
import pytest

import runlength as rl


def dense(rle):
    return pa.array(rle.to_dense()).to_pylist()


def test_shift_with_default_explicit_and_null_fill():
    r = rl.Rle([2, 2], [1, 2])
    assert dense(r.shift(1)) == [0, 1, 1, 2]
    assert dense(r.shift(-1, fill=9)) == [1, 2, 2, 9]
    assert dense(r.shift(2, fill=None)) == [None, None, 1, 1]
    assert dense(rl.Rle([2], ["a"]).shift(1)) == [None, "a"]


def test_reverse_concat_and_ranges():
    r = rl.Rle([1, 2, 3], [1, 0, 1])
    assert dense(r.reverse()) == [1, 1, 1, 0, 0, 1]
    joined = rl.concat([r, rl.Rle([2], [1])])
    assert dense(joined) == [1, 0, 0, 1, 1, 1, 1, 1]
    assert pl.Series(joined.runs).to_list() == [1, 2, 5]
    starts, ends = (r > 0).ranges()
    assert (pl.Series(starts).to_list(), pl.Series(ends).to_list()) == ([0, 3], [1, 6])
    with pytest.raises(TypeError):
        r.ranges()


def test_fill_null():
    r = rl.Rle([1, 2], pa.array([None, 2]))
    assert dense(r.fill_null(2)) == [2, 2, 2]
    assert r.fill_null(2).num_runs == 1


def test_coverage_aggregations():
    starts, ends, values = [0, 1, 4], [2, 3, 5], [5, 1, 3]
    assert dense(rl.coverage(starts, ends, values, agg="max")) == [5, 5, 1, 0, 3]
    assert dense(rl.coverage(starts, ends, values, agg="min", fill=None)) == [5, 1, 1, None, 3]
    assert dense(rl.coverage(starts, ends, values, agg="sum", fill=-1)) == [5, 6, 1, -1, 3]
    assert dense(rl.coverage(starts, ends, agg="count")) == [1, 2, 1, 0, 1]
    names = ["b", "a", "c"]
    assert dense(rl.coverage(starts, ends, names, agg="max", fill=None)) == ["b", "b", "a", None, "c"]
    with pytest.raises(ValueError):
        rl.coverage(starts, ends, agg="max")


def test_coverage_by_key_with_aggregation_and_collapse():
    df = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr2", "chr1"],
            "Strand": ["+", "-", "+", "+"],
            "Start": [0, 1, 0, 2],
            "End": [3, 4, 2, 5],
            "Score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    keys = df.select("Chromosome", "Strand")
    best = rl.coverage_by_key(keys, df["Start"], df["End"], df["Score"], agg="max", fill=None)
    assert dense(best.get(("chr1", "-"))) == [None, 2.0, 2.0, 2.0]

    counts = rl.coverage_by_key(keys, df["Start"], df["End"])
    per_chrom = counts.collapse(drop="Strand")
    assert pl.DataFrame(per_chrom.keys).columns == ["Chromosome"]
    assert dense(per_chrom.get("chr1")) == [1, 2, 3, 2, 1]
    assert dense(counts.collapse(keep=["Chromosome"]).get("chr2")) == [1, 1]
    with pytest.raises(ValueError):
        counts.collapse()


def test_plan_many_and_zip_many():
    a, b, c = rl.Rle([2, 2], [1, 2]), rl.Rle([1, 3], [10, 20]), rl.Rle([3], [100])
    ends, idx = rl.plan_many([a, b, c])
    assert pl.Series(ends).to_list() == [1, 2, 3, 4]
    assert [pl.Series(i).to_list() for i in idx] == [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]]
    total = rl.zip_many([a, b, c], lambda *xs: sum(xs), fill=0, backend="numpy")
    assert dense(total) == [111, 121, 122, 22]
    with pytest.raises(ValueError):
        rl.zip_many([a, c], lambda *xs: xs[0])


def test_overlap_plan_with_polars_aggregation():
    starts, ends = [0, 2, 3], [5, 7, 4]
    scores = pl.Series([1.0, 10.0, 100.0])
    seg_ends, offsets, ids = rl.overlap_plan(starts, ends)
    assert pl.Series(seg_ends).to_list() == [2, 3, 4, 5, 7]
    offsets = np.asarray(offsets)
    segment = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    per_segment = (
        pl.DataFrame({"segment": segment, "score": scores.gather(pl.Series(ids))})
        .group_by("segment", maintain_order=True)
        .agg(pl.col("score").median())
    )
    assert per_segment["score"].to_list() == [1.0, 5.5, 10.0, 5.5, 10.0]
