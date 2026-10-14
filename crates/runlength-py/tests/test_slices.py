import numpy as np
import polars as pl
import pyarrow as pa
import pytest

import runlength as rl


def values(array):
    return pa.array(array).to_pylist()


def dense(rle):
    return pa.array(rle.to_dense()).to_pylist()


def test_slice_agg_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, 200).astype(float)
    r = rl.Rle.from_dense(data)
    starts = rng.integers(0, 190, 50)
    ends = starts + rng.integers(0, 30, 50)
    clamped = [data[s:e] for s, e in zip(starts, ends)]

    assert values(r.slice_agg(starts, ends, "sum")) == pytest.approx([c.sum() for c in clamped])
    assert values(r.slice_agg(starts, ends, "count")) == [len(c) for c in clamped]
    means = values(r.slice_agg(starts, ends))
    assert means == pytest.approx([c.mean() if len(c) else None for c in clamped])
    var = values(r.slice_agg(starts, ends, "var", ddof=0))
    assert var == pytest.approx([c.var() if len(c) else None for c in clamped])
    assert values(r.slice_agg(starts, ends, "max")) == [c.max() if len(c) else None for c in clamped]


def test_slice_agg_skips_nulls_and_validates():
    r = rl.Rle([2, 2, 2], [1, None, 3])
    assert values(r.slice_agg([0, 2], [6, 4], "mean")) == [2.0, None]
    assert values(r.slice_agg([0], [6], "count")) == [4]
    assert values(rl.Rle([3, 1], [True, False]).slice_agg([0], [4], "mean")) == [0.75]
    with pytest.raises(ValueError, match="agg must be"):
        r.slice_agg([0], [1], "median")
    with pytest.raises(ValueError):
        r.slice_agg([3], [1])
    with pytest.raises(TypeError):
        rl.Rle([1], ["a"]).slice_agg([0], [1])


def test_batch_slice_agg_with_keys_from_any_library():
    df = pl.DataFrame({"Chromosome": ["a", "a", "b"], "Start": [0, 2, 0], "End": [4, 6, 3]})
    cov = rl.coverage_by_key(df.select("Chromosome"), df["Start"], df["End"])
    # a: 1 1 2 2 1 1, b: 1 1 1
    q = pl.DataFrame({"Chromosome": ["a", "b", "c", "a"], "Start": [1, 0, 0, 2], "End": [4, 9, 1, 4]})
    expected = [5, 3, 0, 4]
    assert values(cov.slice_agg(q.select("Chromosome"), q["Start"], q["End"], "sum")) == expected
    assert values(cov.slice_agg(q["Chromosome"], q["Start"], q["End"], "sum")) == expected
    as_pandas = q.to_pandas()
    got = cov.slice_agg(as_pandas[["Chromosome"]], as_pandas["Start"], as_pandas["End"], "max")
    assert values(got) == [2, 1, None, 2]
    # Key columns are cast to the batch's types.
    large = q.select(pl.col("Chromosome").cast(pl.Categorical))
    assert values(cov.slice_agg(large, q["Start"], q["End"], "sum")) == expected
    with pytest.raises(ValueError, match="no column"):
        cov.slice_agg(pl.DataFrame({"chrom": ["a"]}), [0], [1])


def test_batch_shift_reverse_and_concat():
    keys = pl.DataFrame({"k": ["x", "y"]})
    a = rl.RleBatch.from_rles(keys, [rl.Rle([2, 1], [1, 2]), rl.Rle([1], [5])])
    shifted = a.shift(1)
    assert dense(shifted.get("x")) == [0, 1, 1]
    assert dense(a.shift(-1, fill=None).get("y")) == [None]
    assert dense(a.reverse().get("x")) == [2, 1, 1]

    b = rl.RleBatch.from_rles(pl.DataFrame({"k": ["z", "x"]}), [rl.Rle([1], [7]), rl.Rle([1], [2])])
    joined = rl.concat([a, b])
    assert pl.DataFrame(joined.keys)["k"].to_list() == ["x", "y", "z"]
    assert dense(joined.get("x")) == [1, 1, 2, 2]
    assert pl.Series(joined.get("x").runs).to_list() == [2, 2]
    assert dense(joined.get("z")) == [7]
    with pytest.raises(TypeError, match="list of Rles"):
        rl.concat([a, rl.Rle([1], [1])])


def test_overlap_plan_by_key_aggregates_with_polars():
    df = pl.DataFrame(
        {"Chromosome": ["a", "b", "a"], "Start": [0, 0, 1], "End": [2, 1, 3], "Score": [1.0, 5.0, 3.0]}
    )
    plan = pl.DataFrame(rl.overlap_plan_by_key(df.select("Chromosome"), df["Start"], df["End"]))
    assert plan.columns == ["Chromosome", "start", "end", "ids"]
    assert plan["ids"].to_list() == [[0], [0, 2], [2], [1]]
    scores = df.with_row_index("ids").select(pl.col("ids").cast(pl.Int64), "Score")
    median = (
        plan.with_row_index("segment")
        .explode("ids")
        .join(scores, on="ids")
        .group_by("segment", maintain_order=True)
        .agg(pl.col("Score").median())
    )
    assert median["Score"].to_list() == [1.0, 2.0, 3.0, 5.0]
