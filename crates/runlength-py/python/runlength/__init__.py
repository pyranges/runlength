"""Run-length encoded sequences of any type, backed by Rust and Apache Arrow.

Arrays go in as polars or pandas Series, pyarrow or numpy arrays, or lists,
and come out as Arrow objects that any of those libraries can read without
copying (``pl.Series(rle.values)``, ``pa.array(rle.values)``,
``np.asarray(rle.values)``). Tables of runs are available with
``to_polars()``, ``to_pandas()`` and ``to_arrow()``.
"""

from runlength._runlength import Rle, RleBatch, __version__, coverage, coverage_by_key

__all__ = ["Rle", "RleBatch", "coverage", "coverage_by_key", "__version__"]
