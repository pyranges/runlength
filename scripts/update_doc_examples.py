"""Rewrite the output of every `>>>` example in the README and docs/.

Runs the examples in each ```python block (all blocks of a file share one
namespace, like a notebook) and replaces whatever output was there with what
the code prints now. Run it after changing a repr or an example:

    python scripts/update_doc_examples.py [files...]

crates/runlength-py/tests/test_docs.py checks that the output stays correct.
"""

import contextlib
import io
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

import runlength as rl

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```(python|pycon)\n(.*?)^```", re.M | re.S)


def statements(block):
    """The (source lines) of each `>>>` statement, dropping old output."""
    out = []
    for line in block.split("\n"):
        if line.startswith(">>> "):
            out.append([line[4:]])
        elif out and (line.startswith("... ") or line == "..."):
            out[-1].append(line[4:])
    return out


def run(stmt, globs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            exec(compile("\n".join(stmt) + "\n", "<doc>", "single"), globs)
        except Exception as e:  # noqa: BLE001
            last = traceback.format_exception_only(e)[-1].rstrip()
            print(f"Traceback (most recent call last):\n    ...\n{last}")
    text = buf.getvalue().rstrip("\n")
    return [line if line.strip() else "<BLANKLINE>" for line in text.split("\n")] if text else []


def update(path):
    text = path.read_text()
    globs = {"rl": rl, "pl": pl, "np": np, "pa": pa, "pd": pd}

    def replace(match):
        lang, block = match.groups()
        if ">>>" not in block:
            return match.group(0)
        lines = []
        for stmt in statements(block):
            lines.append(">>> " + stmt[0])
            lines.extend(("... " + s).rstrip() for s in stmt[1:])
            lines.extend(run(stmt, globs))
        return f"```{lang}\n" + "\n".join(lines) + "\n```"

    new = FENCE.sub(replace, text)
    if new != text:
        path.write_text(new)
        print(f"updated {path.relative_to(ROOT)}")


paths = [Path(p).resolve() for p in sys.argv[1:]] or [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
for path in paths:
    update(path)
