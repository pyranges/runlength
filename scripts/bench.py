"""Timings for the README: the runlength Python package, and pyrle when it
can be imported, on the workload in crates/runlength/examples/bench.rs.

    PYTHONPATH=~/code/pyrle python scripts/bench.py

Each timing is the best of three runs. Prints a Markdown table.
"""

import time
import warnings

import numpy as np
import pandas as pd

import runlength as rl

try:
    from pyrle import Rle as PyrleRle
    from pyrle import RleDict
    from pyrle.src.coverage import _coverage
except ImportError:
    PyrleRle = None

warnings.simplefilter("ignore")
rng = np.random.default_rng(42)
N = 1_000_000
KEYS = 25


def best(f, repeat=3):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times) * 1e3


starts = rng.integers(0, 100_000_000, N)
ends = starts + 100
starts2 = rng.integers(0, 100_000_000, N)
ends2 = starts2 + 100
slice_starts = rng.integers(0, 99_000_000, N)
slice_ends = slice_starts + 1000
keys = np.array([f"k{i}" for i in range(KEYS)])[rng.integers(0, KEYS, N)]
# Keys arrive as a DataFrame column, as from pyranges.
key_frame = pd.DataFrame({"key": keys})

a = rl.coverage(starts, ends)
b = rl.coverage(starts2, ends2)
batch = rl.coverage_by_key(key_frame, starts, ends)
ours = {
    "coverage, 1M intervals": lambda: rl.coverage(starts, ends),
    "a + b": lambda: a + b,
    "a < b": lambda: a < b,
    "a.mean()": lambda: a.mean(),
    "1M slices of 1000, copied": lambda: a.getitems(slice_starts, slice_ends),
    "sum of 1M slices of 1000": lambda: a.slice_agg(slice_starts, slice_ends, "sum"),
    "coverage, 1M intervals, 25 keys": lambda: rl.coverage_by_key(key_frame, starts, ends),
    "batch + batch, 25 keys": lambda: batch + batch,
}

theirs = {}
if PyrleRle is not None:

    def pyrle_coverage(s, e):
        positions = np.concatenate([s, e])
        values = np.concatenate([np.ones(len(s)), -np.ones(len(e))])
        order = np.argsort(positions, kind="mergesort")
        runs, vals = _coverage(positions[order].copy(), values[order].copy())
        return PyrleRle(runs, vals)

    def pyrle_by_key():
        # What pyranges did: one coverage per key.
        return RleDict({k: pyrle_coverage(starts[keys == k], ends[keys == k]) for k in np.unique(keys)})

    pa_, pb_ = pyrle_coverage(starts, ends), pyrle_coverage(starts2, ends2)
    pbatch = pyrle_by_key()
    theirs = {
        "coverage, 1M intervals": lambda: pyrle_coverage(starts, ends),
        "a + b": lambda: pa_ + pb_,
        "a < b": lambda: pa_ < pb_,
        "a.mean()": lambda: pa_.mean(),
        "coverage, 1M intervals, 25 keys": pyrle_by_key,
        "batch + batch, 25 keys": lambda: pbatch + pbatch,
    }

print(f"a has {a.num_runs:,} runs, b has {b.num_runs:,}, a + b has {(a + b).num_runs:,}\n")
print("| Operation | runlength | pyrle |")
print("|---|--:|--:|")
for label, f in ours.items():
    other = f"{best(theirs[label]):.0f} ms" if label in theirs else "–"
    print(f"| {label} | {best(f):.0f} ms | {other} |")
