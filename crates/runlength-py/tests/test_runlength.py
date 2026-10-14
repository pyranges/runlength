import math

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import runlength as rl


def dense(rle):
    return np.asarray(rle.to_dense()).tolist()


# Construction ---------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        [1, 2],
        np.array([1, 2]),
        pl.Series([1, 2]),
        pa.array([1, 2]),
        pd.Series([1, 2]),
    ],
    ids=["list", "numpy", "polars", "pyarrow", "pandas"],
)
def test_construction_from_any_library(values):
    r = rl.Rle([3, 2], values)
    assert dense(r) == [1, 1, 1, 2, 2]
    assert len(r) == 5
    assert r.num_runs == 2


def test_zero_runs_dropped_and_invalid_input():
    r = rl.Rle([1, 0, 2], [1, 9, 2])
    assert pl.Series(r.runs).to_list() == [1, 2]
    with pytest.raises(ValueError):
        rl.Rle([1, -1], [1, 2])
    with pytest.raises(ValueError):
        rl.Rle([1], [1, 2])
    with pytest.raises(TypeError):
        rl.Rle([1.5], [1])


def test_from_dense_merges_nan_and_null_but_not_signed_zero():
    r = rl.Rle.from_dense(pa.array([math.nan, math.nan, 0.0, -0.0, None, None]))
    assert pl.Series(r.runs).to_list() == [2, 1, 1, 2]
    assert r.is_merged()


def test_strings_and_nulls():
    r = rl.Rle.from_dense(pl.Series(["a", "a", None, "b", "b"]))
    assert pl.Series(r.runs).to_list() == [2, 1, 2]
    assert r[0] == "a"
    assert r[2] is None
    assert r[-1] == "b"


# Indexing --------------------------------------------------------------------


def test_getitem_and_slices():
    r = rl.Rle([3, 2, 4], [1, 2, 3])
    assert r[0] == 1 and r[3] == 2 and r[-1] == 3
    assert dense(r[2:6]) == [1, 2, 2, 3]
    assert dense(r[-2:]) == [3, 3]
    assert len(r[100:200]) == 0
    with pytest.raises(IndexError):
        r[9]
    with pytest.raises(ValueError):
        r[::2]


def test_getitems():
    r = rl.Rle([3, 2, 4], [1, 2, 3])
    batch = r.getitems([0, 4, 8], [2, 7, 20])
    assert len(batch) == 3
    assert [dense(batch[i]) for i in range(3)] == [[1, 1], [2, 3, 3], [3]]
    assert pl.DataFrame(batch.keys)["slice"].to_list() == [0, 1, 2]


# Operators -------------------------------------------------------------------


def test_arithmetic_pads_with_zero_and_merges():
    a = rl.Rle([3, 2], [1, 2])
    b = rl.Rle([1, 2], [10, 20])
    s = a + b
    assert dense(s) == [11, 21, 21, 2, 2]
    assert s.is_merged()
    assert dense(a - b) == [-9, -19, -19, 2, 2]
    assert dense(a * b) == [10, 20, 20, 0, 0]


def test_scalars_and_numeric_coercion():
    a = rl.Rle([2, 2], [1, 2])
    assert dense(a + 1) == [2, 2, 3, 3]
    assert dense(10 - a) == [9, 9, 8, 8]
    assert dense(a * 0.5) == [0.5, 0.5, 1.0, 1.0]
    assert (a / 2).dtype == "Float64"
    f = rl.Rle([4], [1.5])
    assert dense(a + f) == [2.5, 2.5, 3.5, 3.5]
    assert dense(-a) == [-1, -1, -2, -2]


def test_comparisons_are_elementwise():
    a = rl.Rle([2, 2], [1, 2])
    b = rl.Rle([1, 3], [2, 2])
    assert dense(a < b) == [True, True, False, False]
    assert dense(a == b) == [False, False, True, True]
    assert dense(a >= 2) == [False, False, True, True]
    assert dense(~(a == 2)) == [True, True, False, False]
    assert dense((a > 1) | (b < 2)) == [False, False, True, True]
    assert a.equals(rl.Rle([1, 1, 2], [1, 1, 2]))
    with pytest.raises(TypeError):
        bool(a == b)


def test_strings_need_explicit_fill_for_unequal_lengths():
    a = rl.Rle([2], ["x"])
    b = rl.Rle([3], ["x"])
    with pytest.raises(ValueError):
        a == b
    c = a.zip_with(b, lambda x, y: pc.equal(x, y), fill_self="?", backend="pyarrow")
    assert dense(c) == [True, True, False]


# Plans, callbacks and merging -------------------------------------------------


def test_plan_and_numpy_gather():
    a = rl.Rle([3, 2], [1, 2])
    b = rl.Rle([1, 4], [10, 20])
    ends, ia, ib = a.plan(b)
    assert pl.Series(ends).to_list() == [1, 3, 5]
    va, vb = np.asarray(a.values), np.asarray(b.values)
    values = va[np.asarray(ia)] * 100 + vb[np.asarray(ib)]
    assert rl.Rle.from_ends(ends, values).equals(
        a.zip_with(b, lambda x, y: pc.add(pc.multiply(x, 100), y), backend="pyarrow")
    )


@pytest.mark.parametrize(
    "backend, func",
    [
        ("pyarrow", lambda x, y: pc.binary_join_element_wise(x, y, "-")),
        ("polars", lambda x, y: x + "-" + y),
        ("numpy", lambda x, y: np.char.add(np.char.add(x.astype(str), "-"), y.astype(str))),
        ("arrow", lambda x, y: pl.Series(x) + "-" + pl.Series(y)),
    ],
    ids=["pyarrow", "polars", "numpy", "arrow"],
)
def test_zip_with_any_library(backend, func):
    a = rl.Rle([2, 2], ["a", "b"])
    b = rl.Rle([1, 3], ["x", "y"])
    c = a.zip_with(b, func, backend=backend)
    assert dense(c) == ["a-x", "a-y", "b-y", "b-y"]
    assert pl.Series(c.runs).to_list() == [1, 1, 2]


def test_callback_errors_propagate():
    a = rl.Rle([2], [1])

    def boom(x, y):
        raise KeyError("from callback")

    with pytest.raises(KeyError, match="from callback"):
        a.zip_with(a, boom)


def test_map_and_merge_with_mask():
    r = rl.Rle([1, 1, 1, 1], [1.0, 1.05, 1.1, 5.0])
    rounded = r.map(lambda v: v.round(), backend="polars")
    assert pl.Series(rounded.runs).to_list() == [3, 1]

    values = np.asarray(r.values)
    mask = np.abs(np.diff(values)) < 0.1
    merged = r.merge(mask=mask, keep="last")
    assert pl.Series(merged.runs).to_list() == [3, 1]
    assert pl.Series(merged.values).to_list() == [1.1, 5.0]
    ends, offsets = r.merge_plan(mask)
    assert pl.Series(offsets).to_list() == [0, 3, 4]


def test_unmerged_results():
    a = rl.Rle([2, 2], [1, 2])
    b = rl.Rle([1, 3], [0, 0])
    raw = a.zip_with(b, lambda x, y: x * y, merge=False, backend="numpy")
    assert raw.num_runs == 3 and not raw.is_merged()
    assert raw.merge().num_runs == 1


# Reductions and conversion -----------------------------------------------------


def test_reductions_are_weighted_by_run_length():
    r = rl.Rle([1, 2, 1], [1, 2, 3])
    values = [1, 2, 2, 3]
    assert r.sum() == 8
    assert r.mean() == pytest.approx(np.mean(values))
    assert r.std() == pytest.approx(np.std(values, ddof=1))
    assert r.var(ddof=0) == pytest.approx(np.var(values))
    assert (r.min(), r.max()) == (1, 3)
    assert rl.Rle([], pa.array([], pa.int64())).mean() is None


def test_conversions():
    r = rl.Rle([2, 1], ["a", "b"])
    for table in [r.to_polars(), pl.DataFrame(r)]:
        assert table.columns == ["start", "end", "value"]
        assert table["value"].to_list() == ["a", "b"]
    assert r.to_pandas()["end"].tolist() == [2, 3]
    assert r.to_arrow().column("start").to_pylist() == [0, 2]
    assert rl.Rle([2], [1.5]).to_numpy().tolist() == [1.5, 1.5]
    assert pa.array(r.values).to_pylist() == ["a", "b"]


# Coverage and batches ------------------------------------------------------------


@pytest.fixture
def intervals():
    return pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr2", "chr1"],
            "Strand": ["+", "+", "-", "-"],
            "Start": [0, 2, 5, 1],
            "End": [3, 4, 7, 2],
            "Score": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_coverage(intervals):
    cov = rl.coverage(intervals["Start"], intervals["End"])
    assert dense(cov) == [1, 2, 2, 1, 0, 1, 1]
    weighted = rl.coverage([0, 2], [3, 4], values=[0.5, 1.0], length=6)
    assert dense(weighted) == [0.5, 0.5, 1.5, 1.0, 0.0, 0.0]


@pytest.mark.parametrize("frame", ["polars", "pandas", "dict"])
def test_coverage_by_key(intervals, frame):
    keys = intervals.select("Chromosome", "Strand")
    keys = {"polars": keys, "pandas": keys.to_pandas(), "dict": keys.to_dict(as_series=False)}[frame]
    batch = rl.coverage_by_key(keys, intervals["Start"], intervals["End"], intervals["Score"])
    assert len(batch) == 3
    assert [k for k, _ in batch.items()] == [("chr1", "+"), ("chr2", "-"), ("chr1", "-")]
    assert dense(batch.get(("chr1", "+"))) == [1.0, 1.0, 3.0, 2.0]
    assert batch.get(("chrX", "+")) is None
    table = batch.to_polars()
    assert table.columns == ["Chromosome", "Strand", "start", "end", "value"]


def test_batch_arithmetic_outer_joins_keys(intervals):
    keys = intervals.select("Chromosome", "Strand")
    a = rl.coverage_by_key(keys, intervals["Start"], intervals["End"])
    b = rl.coverage_by_key(keys[:2], intervals["Start"][:2], intervals["End"][:2] + 2)
    s = a + b
    assert [k for k, _ in s.items()] == [("chr1", "+"), ("chr2", "-"), ("chr1", "-")]
    assert dense(s.get(("chr1", "+"))) == [2, 2, 4, 3, 2, 1]
    assert dense(s.get(("chr2", "-"))) == dense(a.get(("chr2", "-")))
    lt = a < b
    assert lt.dtype == "Boolean"


def test_batch_aggregates(intervals):
    batch = rl.coverage_by_key(intervals.select("Chromosome"), intervals["Start"], intervals["End"])
    sums = pl.DataFrame(batch.sum())
    assert sums.columns == ["Chromosome", "sum"]
    assert sums["sum"].to_list() == [6, 2]
    means = pl.DataFrame(batch.mean())
    assert means["mean"].to_list() == pytest.approx([1.5, 2 / 7])


def test_batch_zip_with_callback_across_threads():
    n = 200
    keys = {"k": [i % 50 for i in range(n)]}
    starts = list(range(n))
    batch = rl.coverage_by_key(keys, starts, [s + 3 for s in starts])
    doubled = batch.zip_with(batch, lambda x, y: pc.add(x, y), how="inner", backend="pyarrow")
    assert len(doubled) == 50
    for (key, d), (_, original) in zip(doubled.items(), batch.items()):
        assert dense(d) == [2 * v for v in dense(original)]


def test_batch_errors():
    a = rl.RleBatch.from_rles({"k": ["x"]}, [rl.Rle([2], ["s"])])
    b = rl.RleBatch.from_rles({"k": ["y"]}, [rl.Rle([2], ["t"])])
    with pytest.raises(ValueError, match="key at output position"):
        a.zip_with(b, lambda x, y: pc.equal(x, y), backend="pyarrow")
    with pytest.raises(ValueError):
        rl.RleBatch.from_rles({"k": ["x", "x"]}, [rl.Rle([1], [1]), rl.Rle([1], [1])])
    with pytest.raises(TypeError):
        a + b.keys
