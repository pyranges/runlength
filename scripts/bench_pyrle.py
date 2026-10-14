"""pyrle timings for the workload in crates/runlength/examples/bench.rs.

    PYTHONPATH=~/code/pyrle python3.12 scripts/bench_pyrle.py
"""

import time
import warnings

import numpy as np

from pyrle import Rle
from pyrle.src.coverage import _coverage

warnings.simplefilter("ignore")
rng = np.random.default_rng(42)
N = 1_000_000


def timed(label, f):
    t = time.perf_counter()
    out = f()
    print(f"{label:<40} {(time.perf_counter() - t) * 1e3:>8.1f} ms")
    return out


def coverage(starts, ends):
    positions = np.concatenate([starts, ends])
    values = np.concatenate([np.ones(len(starts)), -np.ones(len(ends))])
    order = np.argsort(positions, kind="mergesort")
    runs, vals = _coverage(positions[order].copy(), values[order].copy())
    return Rle(runs, vals)


starts = rng.integers(0, 100_000_000, N)
a = timed("coverage_count, 1M intervals", lambda: coverage(starts, starts + 100))
starts2 = rng.integers(0, 100_000_000, N)
b = timed("coverage_count, 1M more intervals", lambda: coverage(starts2, starts2 + 100))
print(f"  runs: {len(a.runs)} and {len(b.runs)}")
s = timed("a + b", lambda: a + b)
print(f"  runs: {len(s.runs)}")
timed("a < b", lambda: a < b)
timed("a.mean()", lambda: a.mean())
