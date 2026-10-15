"""Runs every example in the docstrings, the README and docs/ as a doctest.

Code blocks in Markdown files share one namespace per file, like a notebook,
and blocks without `>>>` are not run.
"""

import doctest
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import runlength as rl

ROOT = Path(__file__).resolve().parents[3]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
FENCE = re.compile(r"^```(?:python|pycon)\n(.*?)^```", re.M | re.S)


def namespace():
    return {"rl": rl, "pl": pl, "np": np, "pa": pa, "pd": pd}


def docstrings():
    for name in rl.__all__:
        obj = getattr(rl, name)
        if not callable(obj):
            continue
        yield name, obj.__doc__
        if inspect.isclass(obj):
            for attr in vars(obj):
                if not attr.startswith("_"):
                    yield f"{name}.{attr}", getattr(obj, attr).__doc__


def run(text, name, globs):
    test = doctest.DocTestParser().get_doctest(text, globs, name, name, 0)
    runner = doctest.DocTestRunner(optionflags=FLAGS)
    runner.run(test, clear_globs=False)
    # DocTest works on a copy; keep its names for the file's next block.
    globs.update(test.globs)
    result = runner.summarize(verbose=False)
    assert result.failed == 0, f"{result.failed} example(s) failed in {name}"


@pytest.mark.parametrize("name,doc", [(n, d) for n, d in docstrings() if d and ">>>" in d])
def test_docstring(name, doc):
    run(doc, name, namespace())


def test_all_public_methods_have_examples():
    missing = [n for n, d in docstrings() if not (d and ">>>" in d)]
    assert not missing, missing


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_markdown(path):
    globs = namespace()
    blocks = [b for b in FENCE.findall(path.read_text()) if ">>>" in b]
    for i, block in enumerate(blocks):
        run(block, f"{path.name} block {i + 1}", globs)
