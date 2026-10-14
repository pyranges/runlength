"""Generate random test cases with pyrle, for cross-checking runlength.

Run with a Python that can import pyrle, from the repository root:

    PYTHONPATH=~/code/pyrle python3.12 scripts/gen_pyrle_fixtures.py

It writes crates/runlength/tests/fixtures/pyrle_cases.txt, which
tests/pyrle_crosscheck.rs reads. Each case is a block of lines:

    case <op>
    <name> <space-separated numbers>
    ...

Values are small integers stored as floats, so they print exactly.
"""

import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pyrle import Rle
from pyrle.src.coverage import _coverage

OUT = Path(__file__).resolve().parent.parent / "crates/runlength/tests/fixtures/pyrle_cases.txt"


def random_rle(rng):
    n = rng.randint(1, 8)
    runs = [rng.randint(1, 4) for _ in range(n)]
    values = [float(rng.randint(-2, 3)) for _ in range(n)]
    return runs, values


def line(name, xs):
    return name + " " + " ".join(str(int(x)) for x in xs)


def pyrle_coverage(starts, ends, weights):
    # pyrle.methods.coverage hands a read-only array to Cython under current
    # pandas, so build the same sorted event list here and call the kernel.
    df = pd.concat(
        [
            pd.DataFrame({"Position": starts, "Value": weights}),
            pd.DataFrame({"Position": ends, "Value": [-w for w in weights]}),
        ],
        ignore_index=True,
    ).sort_values("Position", kind="mergesort")
    positions = np.ascontiguousarray(df.Position.values, dtype=np.int64).copy()
    values = np.ascontiguousarray(df.Value.values, dtype=np.float64).copy()
    # pyrle's kernel triggers pandas copy-on-write warnings; silenced here,
    # and the results are checked against runlength anyway.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs, vals = _coverage(positions, values)
    rle = Rle(runs, vals)
    return rle.runs, rle.values


def main(n_cases=300, seed=0):
    rng = random.Random(seed)
    blocks = []
    ops = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}
    for _ in range(n_cases):
        for op, f in ops.items():
            (ar, av), (br, bv) = random_rle(rng), random_rle(rng)
            out = f(Rle(ar, av), Rle(br, bv))
            blocks.append(
                [f"case {op}", line("a_runs", ar), line("a_values", av), line("b_runs", br),
                 line("b_values", bv), line("out_runs", out.runs), line("out_values", out.values)]
            )

        n = rng.randint(1, 8)
        starts = [rng.randint(0, 15) for _ in range(n)]
        ends = [s + rng.randint(1, 6) for s in starts]
        weights = [float(rng.randint(-2, 4)) for _ in range(n)]
        runs, values = pyrle_coverage(starts, ends, weights)
        blocks.append(
            ["case coverage", line("starts", starts), line("ends", ends), line("weights", weights),
             line("out_runs", runs), line("out_values", values)]
        )

    OUT.write_text("\n\n".join("\n".join(b) for b in blocks) + "\n")
    print(f"wrote {len(blocks)} cases to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
