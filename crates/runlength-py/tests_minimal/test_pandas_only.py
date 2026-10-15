"""runlength with only pandas and numpy installed, as in a default pyranges
install. CI runs this file in an environment without pyarrow or polars."""

import numpy as np
import pandas as pd
import pytest

import runlength as rl


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "Chromosome": pd.Categorical(["a", "b", "a"]),
            "Strand": ["+", "-", "+"],
            "Start": np.array([0, 1, 2], dtype="int32"),
            "End": [3, 4, 5],
            "Score": [1.0, 2.0, 3.0],
        },
        index=[10, 20, 30],
    )


def test_inputs_from_pandas_and_numpy(df):
    assert np.asarray(rl.coverage(df.Start, df.End).values).tolist() == [1, 2, 3, 2, 1]
    assert rl.coverage(df.Start.to_numpy(), df.End.to_numpy(), df.Score.to_numpy()).num_runs == 5
    batch = rl.coverage_by_key(df[["Chromosome", "Strand"]], df.Start, df.End, df.Score)
    assert [key for key, _ in batch.items()] == [("a", "+"), ("b", "-")]
    by_series = rl.coverage_by_key(df.Chromosome, df.Start, df.End)
    assert by_series.keys.column_names == ["Chromosome"]
    by_numpy = rl.coverage_by_key(np.array(["a", "b", "a"]), df.Start, df.End)
    assert len(by_numpy) == 2


def test_outputs_to_pandas_and_numpy(df):
    batch = rl.coverage_by_key(df[["Chromosome"]], df.Start, df.End, df.Score)
    table = batch.to_pandas()
    assert table.columns.tolist() == ["Chromosome", "start", "end", "value"]
    assert table.value.tolist() == [1.0, 4.0, 3.0, 0.0, 2.0]
    assert rl.coverage(df.Start, df.End).to_pandas().end.tolist() == [1, 2, 3, 4, 5]
    scaled = (batch * 2).map(np.sqrt, backend="numpy")
    assert scaled.get("a").to_numpy().tolist() == pytest.approx([2**0.5] * 2 + [8**0.5] + [6**0.5] * 2)
    means = batch.slice_agg(df.Chromosome, df.Start, df.End, "mean")
    # a: 1 1 4 3 3, b: 0 2 2 2
    assert np.asarray(means).tolist() == pytest.approx([2.0, 2.0, 10 / 3])


def test_pyarrow_is_really_absent():
    with pytest.raises(ImportError):
        import pyarrow  # noqa: F401


def test_lists_of_other_types_need_pyarrow():
    import datetime

    with pytest.raises(TypeError, match="pass a polars Series or a pyarrow array"):
        rl.Rle([1], [datetime.date(2026, 1, 1)])
