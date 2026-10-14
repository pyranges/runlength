"""Reference port of pyranges' pyrle-based methods to runlength, checked
against pyranges1 + pyrle on random data.

The functions at the top are what pyranges (pandas) and polaranges (polars)
need; they take a pandas or polars DataFrame with Chromosome, Start, End and
optionally Strand and a value column.

    PYTHONPATH=~/code/pyrle python scripts/pyranges_port.py [cases]

needs pyranges1 (~/code/pyrunges), pyrle (built in place), runlength, pandas,
polars and numpy.
"""

import sys

import numpy as np
import pandas as pd
import polars as pl

import runlength as rl


def to_rle(df, value_col=None, *, strand=True, rpm=False):
    """pyranges' ``to_rle``: coverage (or the sum of ``value_col``) per
    chromosome, or per chromosome and strand, optionally in reads per million."""
    keys = df[["Chromosome", "Strand"]] if strand else df[["Chromosome"]]
    values = df[value_col] if value_col is not None else None
    cov = rl.coverage_by_key(keys, df["Start"], df["End"], values)
    if rpm:
        cov = cov * (1e6 / len(df))
    return cov


def to_ranges(cov):
    """pyrle's ``to_ranges``: the runs as intervals, without zero-valued runs."""
    runs = pl.DataFrame(cov.to_polars())
    runs = runs.rename({"start": "Start", "end": "End", "value": "Score"})
    return runs.filter(pl.col("Score") != 0)


def bigwig_data(df, value_col=None, *, divide=False, rpm=True):
    """pyranges' ``to_bigwig(..., return_data=True)``: unstranded coverage runs,
    or with ``divide`` log2(value coverage / coverage)."""
    if divide:
        # rpm scales both sides by the same factor, so it cancels; scaling
        # first would only add rounding noise that splits equal ratios.
        numerator = to_rle(df, value_col, strand=False)
        denominator = to_rle(df, strand=False)
        cov = (numerator / denominator).map(np.log2, backend="numpy")
    else:
        cov = to_rle(df, value_col, strand=False, rpm=rpm)
    return to_ranges(cov).sort("Chromosome", "Start")


# --- Checks against pyranges1 + pyrle ---------------------------------------


def random_frame(rng):
    n = int(rng.integers(1, 40))
    starts = rng.integers(0, 200, n)
    return pd.DataFrame(
        {
            "Chromosome": rng.choice(["chr1", "chr2", "chrX"], n),
            "Start": starts,
            "End": starts + rng.integers(1, 60, n),
            "Strand": rng.choice(["+", "-"], n),
            # Includes zeros and negative values, which can cancel exactly.
            "Score": rng.choice([0.0, 0.5, 1.0, 2.5, -1.0], n),
        }
    )


def dense(runs, values):
    return np.repeat(np.asarray(values, dtype=float), np.asarray(runs))


def check_to_rle(df, value_col, strand, rpm):
    import pyranges1 as pr

    expected = pr.PyRanges(df).to_rle(value_col, strand=strand, rpm=rpm)
    actual = to_rle(df, value_col, strand=strand, rpm=rpm)
    assert len(actual) == len(expected.rles), (len(actual), list(expected.rles))
    for key, rle in actual.items():
        want = expected.rles[key if strand else key[0]]
        got = dense(pl.Series(rle.runs), pl.Series(rle.values))
        want = dense(want.runs, want.values)
        # pyrle pads nothing at the end; both start at 0.
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-9, err_msg=str(key))


def check_bigwig(df, value_col, divide, rpm):
    import pyranges1 as pr

    expected = pr.PyRanges(df).to_bigwig(
        None, {}, value_col, divide=divide, rpm=rpm, return_data=True
    )
    expected = pl.from_pandas(pd.DataFrame(expected)[["Chromosome", "Start", "End", "Score"]])
    actual = bigwig_data(df, value_col, divide=divide, rpm=rpm)
    expected = expected.with_columns(pl.col("Chromosome").cast(pl.String))
    actual = actual.with_columns(pl.col("Chromosome").cast(pl.String))
    assert actual.select("Chromosome", "Start", "End").equals(
        expected.select("Chromosome", "Start", "End")
    ), f"{actual}\n{expected}"
    # pyrle merges the divided runs in float32 (`cdef float value` in
    # `_remove_dupes`), so its values are only float32-accurate there;
    # runlength's are exact float64.
    rtol = 1e-6 if divide else 1e-12
    np.testing.assert_allclose(actual["Score"].to_numpy(), expected["Score"].to_numpy(), rtol=rtol)


def check_polars_input(df, value_col, divide, rpm):
    """polaranges passes polars frames; the results must not depend on it."""
    frame = pl.from_pandas(df)
    for strand in (False, True):
        a = to_rle(df, value_col, strand=strand, rpm=rpm)
        b = to_rle(frame, value_col, strand=strand, rpm=rpm)
        assert pl.DataFrame(a.to_polars()).equals(pl.DataFrame(b.to_polars()))
    a = bigwig_data(df, value_col, divide=divide, rpm=rpm)
    b = bigwig_data(frame, value_col, divide=divide, rpm=rpm)
    assert a.equals(b), f"{a}\n{b}"


def main(cases):
    import warnings

    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    checks = 0
    for _ in range(cases):
        df = random_frame(rng)
        for value_col in (None, "Score"):
            for rpm in (False, True):
                for strand in (False, True):
                    check_to_rle(df, value_col, strand, rpm)
                    checks += 1
                for divide in (False, True) if value_col else (False,):
                    check_bigwig(df, value_col, divide, rpm)
                    check_polars_input(df, value_col, divide, rpm)
                    checks += 2
    print(f"{checks} checks on {cases} random frames match pyranges1 + pyrle")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
