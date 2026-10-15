"""Copy the docstrings of the compiled module into its type stub.

The docstrings are written once, as doc comments in crates/runlength-py/src.
Editors and the API reference read the stub (_runlength.pyi), so this copies
them there. Run it after changing a doc comment and rebuilding:

    python scripts/sync_stub.py          # rewrite the stub
    python scripts/sync_stub.py --check  # fail if the stub is out of date
"""

import ast
import sys
from pathlib import Path

import runlength._runlength as module

STUB = Path(__file__).resolve().parents[1] / "crates/runlength-py/python/runlength/_runlength.pyi"


def docstring_literal(doc, indent):
    body = "\n".join((indent + line).rstrip() for line in doc.split("\n"))
    return f'{indent}"""{body.lstrip()}\n{indent}"""'


def edits(tree, owner, prefix=""):
    """(start, end, replacement) line edits for every documented definition."""
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)]
    for i, node in enumerate(defs):
        # Of several @overloads, only the last one carries the docstring.
        if any(later.name == node.name for later in defs[i + 1 :]):
            continue
        obj = getattr(owner, node.name, None)
        doc = getattr(obj, "__doc__", None)
        if obj is None or node.name.startswith("__") or not doc:
            continue
        if isinstance(node, ast.ClassDef):
            yield from edits(node, obj, prefix + "    ")
        indent = prefix + "    "
        literal = docstring_literal(doc, indent)
        first = node.body[0]
        constant = first.value.value if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) else None
        if isinstance(constant, str):
            yield first.lineno - 1, first.end_lineno, literal.split("\n")
        elif constant is Ellipsis and len(node.body) == 1:
            # `def f(...) -> T: ...` becomes `def f(...) -> T:` plus the docstring.
            line = first.lineno - 1
            yield line, line + 1, [None, *literal.split("\n")]
        else:
            yield node.body[0].lineno - 1, node.body[0].lineno - 1, literal.split("\n")


def sync(text):
    lines = text.split("\n")
    tree = ast.parse(text)
    for start, end, new in sorted(edits(tree, module), reverse=True):
        if new[0] is None:
            # Strip the trailing `: ...` from a one-line definition.
            head = lines[start]
            assert head.rstrip().endswith("..."), head
            new = [head.rstrip()[:-3].rstrip(), *new[1:]]
        lines[start:end] = new
    return "\n".join(lines)


old = STUB.read_text()
new = sync(old)
if "--check" in sys.argv:
    if new != old:
        sys.exit(f"{STUB.name} is out of date: run python scripts/sync_stub.py")
else:
    STUB.write_text(new)
