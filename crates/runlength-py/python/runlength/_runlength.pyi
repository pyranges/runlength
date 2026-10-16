from typing import Any, Callable, Literal, Sequence, TypeVar

# Inputs: anything with the Arrow PyCapsule interface (polars, pyarrow,
# pandas, arro3, nanoarrow), numpy arrays, pandas Series, or lists.
ArrayLike = Any
TableLike = Any
# Keys: a table, or for a single key column an array (Series, list, ...).
KeysLike = Any
# Outputs implement __arrow_c_array__ / __arrow_c_stream__ and __array__.
ArrowArray = Any
ArrowTable = Any

Backend = Literal["arrow", "pyarrow", "polars", "numpy"]
How = Literal["outer", "inner", "left", "right"]
SliceAgg = Literal["sum", "count", "mean", "var", "std", "min", "max"]

__version__: str

class Rle:
    """A run-length encoded sequence of values of any Arrow type.

    `runs` are the run lengths and `values` one value per run. Both can be
    polars or pandas Series, pyarrow or numpy arrays, or lists. Zero-length
    runs are dropped; neighbouring identical values are kept as given (see
    `merge`). Every sequence starts at position 0.

    Arithmetic (`+ - * /`), logic (`& | ~`) and comparisons (`== != < <= > >=`)
    work position by position, with another Rle or a single value. Between
    two Rles the shorter one is padded with zero, and the result is merged.
    `/` divides integers as floats, as Python does.

    `len(rle)` is the number of positions, `rle[i]` the value at position `i`
    and `rle[a:b]` a new sequence starting at 0; negative positions count from
    the end.

    Examples:
        >>> rl.Rle([3, 2], [1.5, 0.0])
        Rle(length=5, runs=2, dtype=Float64)
          runs:   [3, 2]
          values: [1.5, 0.0]
        >>> rl.Rle([2, 1], ["a", "b"]).to_dense().to_pylist()
        ['a', 'a', 'b']
        >>> a = rl.Rle([3, 2], [1, 2])
        >>> b = rl.Rle([1, 4], [10, 20])
        >>> a + b
        Rle(length=5, runs=3, dtype=Int64)
          runs:   [1, 2, 2]
          values: [11, 21, 22]
        >>> a / 2
        Rle(length=5, runs=2, dtype=Float64)
          runs:   [3, 2]
          values: [0.5, 1.0]
        >>> a < b
        Rle(length=5, runs=1, dtype=Boolean)
          runs:   [5]
          values: [true]
        >>> len(a), a[3], a[-1]
        (5, 2, 2)
        >>> a[2:4]
        Rle(length=2, runs=2, dtype=Int64)
          runs:   [1, 1]
          values: [1, 2]
    """

    def __init__(self, runs: ArrayLike, values: ArrayLike) -> None: ...
    @staticmethod
    def from_dense(values: ArrayLike) -> Rle:
        """Encodes one value per position, merging identical neighbours.

        Examples:
            >>> rl.Rle.from_dense([1, 1, 1, 2, 2, None, None])
            Rle(length=7, runs=3, dtype=Int64)
              runs:   [3, 2, 2]
              values: [1, 2, null]
        """
    @staticmethod
    def from_ends(ends: ArrayLike, values: ArrayLike) -> Rle:
        """Builds a sequence from cumulative run ends and one value per run.

        Examples:
            >>> rl.Rle.from_ends([3, 5], ["x", "y"])
            Rle(length=5, runs=2, dtype=Utf8)
              runs:   [3, 2]
              values: ["x", "y"]
        """
    @property
    def runs(self) -> ArrowArray:
        """The length of each run.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).runs.to_pylist()
            [3, 2]
        """
    @property
    def ends(self) -> ArrowArray:
        """The cumulative end of each run.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).ends.to_pylist()
            [3, 5]
        """
    @property
    def starts(self) -> ArrowArray:
        """The start of each run.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).starts.to_pylist()
            [0, 3]
        """
    @property
    def values(self) -> ArrowArray:
        """One value per run.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).values.to_pylist()
            [1, 2]
        """
    @property
    def dtype(self) -> str:
        """The Arrow type of the values.

        Examples:
            >>> rl.Rle([3, 2], [1.0, 2.0]).dtype
            'Float64'
        """
    @property
    def num_runs(self) -> int:
        """The number of runs.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).num_runs
            2
        """
    def __len__(self) -> int: ...
    def __getitem__(self, key: int | slice) -> Any: ...
    def getitems(self, starts: ArrayLike, ends: ArrayLike) -> RleBatch:
        """Many slices at once, as an RleBatch keyed by slice number. Ends past the
        sequence are clamped.

        Examples:
            >>> r = rl.Rle([3, 2], [1, 2])
            >>> r.getitems([0, 2], [2, 5])
            RleBatch(keys=2, runs=3, dtype=Int64)
              (0,): length=2, runs=[2], values=[1]
              (1,): length=3, runs=[1, 2], values=[1, 2]
        """
    def slice_agg(
        self, starts: ArrayLike, ends: ArrayLike, agg: SliceAgg = "mean", ddof: int = 1
    ) -> ArrowArray:
        """Reduces many slices without copying them, in parallel: entry k reduces
        positions starts[k]:ends[k] (ends past the sequence are clamped).
        `agg` is "sum", "count", "mean", "var", "std", "min" or "max"; nulls
        are skipped and `ddof` applies to "var" and "std". Values must be
        numeric or boolean.

        Examples:
            >>> r = rl.Rle([3, 2], [1, 2])
            >>> r.slice_agg([0, 2], [2, 5], agg="sum").to_pylist()
            [2, 5]
            >>> r.slice_agg([0, 2], [2, 5]).to_pylist()
            [1.0, 1.6666666666666667]
        """
    def merge(self, mask: ArrayLike | None = None, keep: Literal["first", "last"] = "first") -> Rle:
        """Merges neighbouring runs. Without a mask, runs merge when their values
        are identical (NaN with NaN, null with null). With a mask (one boolean
        per pair of neighbouring runs), run i + 1 merges into run i where
        mask[i] is true, keeping the "first" or "last" value of each group.

        Examples:
            >>> r = rl.Rle([1, 1, 2], [5, 5, 7])
            >>> r.merge()
            Rle(length=4, runs=2, dtype=Int64)
              runs:   [2, 2]
              values: [5, 7]
            >>> r.merge(mask=[False, True], keep="last")
            Rle(length=4, runs=2, dtype=Int64)
              runs:   [1, 3]
              values: [5, 7]
        """
    def merge_plan(self, mask: ArrayLike) -> tuple[ArrowArray, ArrowArray]:
        """Groups runs by a mask without touching values: returns (ends, offsets),
        where group g is runs offsets[g]:offsets[g + 1] and ends at ends[g].

        Examples:
            >>> r = rl.Rle([1, 1, 2], [5, 5, 7])
            >>> ends, offsets = r.merge_plan([False, True])
            >>> ends.to_pylist(), offsets.to_pylist()
            ([1, 4], [0, 1, 3])
        """
    def is_merged(self) -> bool:
        """Whether no two neighbouring runs hold identical values.

        Examples:
            >>> rl.Rle([1, 1], [5, 5]).is_merged()
            False
        """
    def plan(self, other: Rle) -> tuple[ArrowArray, ArrowArray, ArrowArray]:
        """The plan for combining with `other`: (ends, idx_self, idx_other).

        Segment k ends at ends[k] and is covered by run idx_self[k] of this
        sequence and run idx_other[k] of `other`. An index equal to that side's
        number of runs means the segment lies past its end.

        Examples:
            >>> a = rl.Rle([3, 2], [1, 2])
            >>> b = rl.Rle([1, 4], [10, 20])
            >>> ends, idx_a, idx_b = a.plan(b)
            >>> ends.to_pylist(), idx_a.to_pylist(), idx_b.to_pylist()
            ([1, 3, 5], [0, 0, 1], [0, 1, 1])
        """
    def zip_with(
        self,
        other: Rle,
        func: Callable[[Any, Any], ArrayLike],
        fill_self: Any = None,
        fill_other: Any = None,
        merge: bool = True,
        backend: Backend = "arrow",
    ) -> Rle:
        """Combines with `other` position by position using `func`.

        `func(a, b)` receives two Arrow arrays of equal length, the values of
        both sides on each segment of the plan, and must return one value per
        segment (any Arrow-compatible array). The shorter side is padded with
        its fill value; without one, unequal lengths are an error.

        `backend` picks what `func` receives: "arrow" (objects with the Arrow
        PyCapsule interface), "pyarrow", "polars" or "numpy".

        Examples:
            >>> a = rl.Rle([3, 2], ["x", "y"])
            >>> b = rl.Rle([1, 4], ["p", "q"])
            >>> a.zip_with(b, lambda x, y: [i + j for i, j in zip(x.to_pylist(), y.to_pylist())])
            Rle(length=5, runs=3, dtype=Utf8)
              runs:   [1, 2, 2]
              values: ["xp", "xq", "yq"]
            >>> a.zip_with(b, lambda x, y: x + y, backend="polars")
            Rle(length=5, runs=3, dtype=Utf8View)
              runs:   [1, 2, 2]
              values: ["xp", "xq", "yq"]
        """
    def map(self, func: Callable[[Any], ArrayLike], backend: Backend = "arrow") -> Rle:
        """Applies `func` to the run values and merges the result. `func` must
        return one value per run; `backend` is as in `zip_with`.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).map(lambda v: v * 10, backend="numpy")
            Rle(length=5, runs=2, dtype=Int64)
              runs:   [3, 2]
              values: [10, 20]
        """
    def cast(self, dtype: str) -> Rle:
        """Casts the values to another Arrow type, given as a string such as
        "float64" or "int32".

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).cast("float64")
            Rle(length=5, runs=2, dtype=Float64)
              runs:   [3, 2]
              values: [1.0, 2.0]
        """
    def shift(self, distance: int, fill: Any = ...) -> Rle:
        """Shifts values `distance` positions right (left if negative), keeping
        the length. Positions shifted in hold `fill`: by default 0 for numbers,
        false for booleans and null otherwise; `fill=None` means null.

        Examples:
            >>> r = rl.Rle([3, 2], [1, 2])
            >>> r.shift(2)
            Rle(length=5, runs=2, dtype=Int64)
              runs:   [2, 3]
              values: [0, 1]
            >>> r.shift(-1, fill=None)
            Rle(length=5, runs=3, dtype=Int64)
              runs:   [2, 2, 1]
              values: [1, 2, null]
        """
    def reverse(self) -> Rle:
        """The sequence back to front.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).reverse()
            Rle(length=5, runs=2, dtype=Int64)
              runs:   [2, 3]
              values: [2, 1]
        """
    def ranges(self) -> tuple[ArrowArray, ArrowArray]:
        """For a boolean Rle, the ranges where it is true, as (starts, ends);
        nulls count as false. For example `(coverage > 0).ranges()`.

        Examples:
            >>> cov = rl.coverage([0, 2, 8], [3, 4, 10])
            >>> starts, ends = (cov > 0).ranges()
            >>> starts.to_pylist(), ends.to_pylist()
            ([0, 8], [4, 10])
        """
    def fill_null(self, value: Any) -> Rle:
        """Replaces nulls with `value`.

        Examples:
            >>> rl.Rle([2, 3], [None, 1]).fill_null(0)
            Rle(length=5, runs=2, dtype=Int64)
              runs:   [2, 3]
              values: [0, 1]
        """
    def equals(self, other: Rle) -> bool:
        """Whether both sequences hold identical values at every position,
        however their runs are split. (`==` compares element-wise.)

        Examples:
            >>> a = rl.Rle([2, 1], [1, 2])
            >>> b = rl.Rle([1, 1, 1], [1, 1, 2])
            >>> a.equals(b)
            True
            >>> a == b
            Rle(length=3, runs=1, dtype=Boolean)
              runs:   [3]
              values: [true]
        """
    def sum(self) -> Any:
        """The sum over all positions.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).sum()
            7
        """
    def mean(self) -> float | None:
        """The mean over all positions, or None if empty.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).mean()
            1.4
        """
    def var(self, ddof: int = 1) -> float | None:
        """The variance over all positions, with `ddof` delta degrees of freedom.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).var()
            0.3
            >>> rl.Rle([3, 2], [1, 2]).var(ddof=0)
            0.24
        """
    def std(self, ddof: int = 1) -> float | None:
        """The standard deviation over all positions, with `ddof` delta degrees of
        freedom.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).std(ddof=0)
            0.4898979485566356
        """
    def min(self) -> Any:
        """The smallest value, ignoring NaN.

        Examples:
            >>> rl.Rle([3, 2], [1.0, float("nan")]).min()
            1.0
        """
    def max(self) -> Any:
        """The largest value, ignoring NaN.

        Examples:
            >>> rl.Rle([3, 2], [1.0, float("nan")]).max()
            1.0
        """
    def to_dense(self) -> ArrowArray:
        """One value per position, as an Arrow array.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).to_dense().to_pylist()
            [1, 1, 1, 2, 2]
        """
    def to_numpy(self) -> Any:
        """One value per position, as a numpy array.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).to_numpy()
            array([1, 1, 1, 2, 2])
        """
    def to_arrow(self) -> Any:
        """The runs as a pyarrow Table with columns start, end and value.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).to_arrow()
            pyarrow.Table
            start: int64 not null
            end: int64 not null
            value: int64
            ----
            start: [[0,3]]
            end: [[3,5]]
            value: [[1,2]]
        """
    def to_polars(self) -> Any:
        """The runs as a polars DataFrame with columns start, end and value.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).to_polars()
            shape: (2, 3)
            ┌───────┬─────┬───────┐
            │ start ┆ end ┆ value │
            │ ---   ┆ --- ┆ ---   │
            │ i64   ┆ i64 ┆ i64   │
            ╞═══════╪═════╪═══════╡
            │ 0     ┆ 3   ┆ 1     │
            │ 3     ┆ 5   ┆ 2     │
            └───────┴─────┴───────┘
        """
    def to_pandas(self) -> Any:
        """The runs as a pandas DataFrame with columns start, end and value.

        Examples:
            >>> rl.Rle([3, 2], [1, 2]).to_pandas()
               start  end  value
            0      0    3      1
            1      3    5      2
        """
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...
    def __add__(self, other: Rle | Any) -> Rle: ...
    def __radd__(self, other: Any) -> Rle: ...
    def __sub__(self, other: Rle | Any) -> Rle: ...
    def __rsub__(self, other: Any) -> Rle: ...
    def __mul__(self, other: Rle | Any) -> Rle: ...
    def __rmul__(self, other: Any) -> Rle: ...
    def __truediv__(self, other: Rle | Any) -> Rle: ...
    def __rtruediv__(self, other: Any) -> Rle: ...
    def __and__(self, other: Rle | Any) -> Rle: ...
    def __rand__(self, other: Any) -> Rle: ...
    def __or__(self, other: Rle | Any) -> Rle: ...
    def __ror__(self, other: Any) -> Rle: ...
    def __lt__(self, other: Rle | Any) -> Rle: ...
    def __le__(self, other: Rle | Any) -> Rle: ...
    def __eq__(self, other: Rle | Any) -> Rle: ...  # type: ignore[override]
    def __ne__(self, other: Rle | Any) -> Rle: ...  # type: ignore[override]
    def __gt__(self, other: Rle | Any) -> Rle: ...
    def __ge__(self, other: Rle | Any) -> Rle: ...
    def __neg__(self) -> Rle: ...
    def __invert__(self) -> Rle: ...

class RleBatch:
    """Many Rles, each identified by a row of key columns of any types (for
    example Chromosome and Strand).

    Build one with `coverage_by_key`, `Rle.getitems`, `RleBatch.from_table` or
    `RleBatch.from_rles`.
    Operators combine two batches key by key (an outer join, padding missing
    keys and shorter sequences with zero), in parallel.

    `len(batch)` is the number of Rles and `batch[i]` the Rle at position `i`;
    look one up by key with `get`.

    Examples:
        >>> keys = pl.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "strand": ["+", "-", "+"]})
        >>> batch = rl.coverage_by_key(keys, [0, 2, 5], [3, 4, 7])
        >>> batch
        RleBatch(keys=3, runs=5, dtype=Int64)
          ('chr1', '+'): length=3, runs=[3], values=[1]
          ('chr1', '-'): length=4, runs=[2, 2], values=[0, 1]
          ('chr2', '+'): length=7, runs=[5, 2], values=[0, 1]
        >>> len(batch)
        3
        >>> batch[-1]
        Rle(length=7, runs=2, dtype=Int64)
          runs:   [5, 2]
          values: [0, 1]
    """

    @staticmethod
    def from_table(
        table: TableLike,
        keys: str | Sequence[str],
        values: str = "value",
        starts: str | None = "start",
        ends: str = "end",
        runs: str | None = None,
    ) -> RleBatch:
        """Builds a batch from a table with one row per run: a polars or pandas
        DataFrame, pyarrow Table, or dict of columns. This is the inverse of
        `to_polars`, `to_pandas` and `to_arrow`.

        `keys` names the key column(s). Each run lies at `starts`..`ends`;
        each key's runs must start at 0 and follow each other without gaps
        (for intervals with gaps, use `coverage_by_key`). With `starts=None`,
        each run ends at `ends` and starts where the previous one ended, and
        with `runs`, each run has that length. Keys appear in order of first
        appearance, and a key's rows need not be adjacent, but they must be in
        position order.

        Examples:
            >>> table = pl.DataFrame({
            ...     "chrom": ["chr1", "chr1", "chr2"],
            ...     "start": [0, 3, 0],
            ...     "end": [3, 5, 4],
            ...     "value": [1, 2, 7],
            ... })
            >>> rl.RleBatch.from_table(table, "chrom")
            RleBatch(keys=2, runs=3, dtype=Int64)
              ('chr1',): length=5, runs=[3, 2], values=[1, 2]
              ('chr2',): length=4, runs=[4], values=[7]
            >>> lengths = {"chrom": ["chr1", "chr1"], "width": [3, 2], "score": [0.5, 1.5]}
            >>> rl.RleBatch.from_table(lengths, "chrom", values="score", runs="width")
            RleBatch(keys=1, runs=2, dtype=Float64)
              ('chr1',): length=5, runs=[3, 2], values=[0.5, 1.5]
        """
    @staticmethod
    def from_rles(keys: KeysLike, rles: Sequence[Rle]) -> RleBatch:
        """Builds a batch from a key table (one row per Rle: a polars or pandas
        DataFrame, pyarrow Table, or dict of columns) and a list of Rles with
        the same value type.

        Examples:
            >>> rl.RleBatch.from_rles({"name": ["a", "b"]}, [rl.Rle([2], [1]), rl.Rle([1, 1], [3, 4])])
            RleBatch(keys=2, runs=3, dtype=Int64)
              ('a',): length=2, runs=[2], values=[1]
              ('b',): length=2, runs=[1, 1], values=[3, 4]
        """
    @property
    def keys(self) -> ArrowTable:
        """The key table, one row per Rle.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
            >>> pl.DataFrame(batch.keys)
            shape: (2, 1)
            ┌───────┐
            │ chrom │
            │ ---   │
            │ str   │
            ╞═══════╡
            │ chr1  │
            │ chr2  │
            └───────┘
        """
    @property
    def dtype(self) -> str:
        """The Arrow type of the values.

        Examples:
            >>> rl.coverage_by_key({"chrom": ["chr1"]}, [0], [3], values=[0.5]).dtype
            'Float64'
        """
    def __len__(self) -> int: ...
    def __getitem__(self, position: int) -> Rle: ...
    def get(self, key: Any) -> Rle | None:
        """The Rle for a key, or None. A key is a tuple with one value per key
        column, or a single value if there is one key column.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1"], "strand": ["+", "-"]}, [0, 2], [3, 4])
            >>> batch.get(("chr1", "-"))
            Rle(length=4, runs=2, dtype=Int64)
              runs:   [2, 2]
              values: [0, 1]
            >>> batch.get(("chr9", "+")) is None
            True
        """
    def items(self) -> list[tuple[tuple[Any, ...], Rle]]:
        """(key tuple, Rle) pairs in key order.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
            >>> [(key, rle.num_runs) for key, rle in batch.items()]
            [(('chr1',), 1), (('chr2',), 2)]
        """
    def zip_with(
        self,
        other: RleBatch,
        func: Callable[[Any, Any], ArrayLike],
        how: How = "outer",
        fill_self: Any = None,
        fill_other: Any = None,
        backend: Backend = "arrow",
    ) -> RleBatch:
        """Combines with `other` key by key using `func`; see `Rle.zip_with`.
        `how` is "outer", "inner", "left" or "right". A key missing on one
        side is an empty Rle there, padded with that side's fill value.

        Examples:
            >>> a = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
            >>> b = rl.coverage_by_key({"chrom": ["chr1", "chr3"]}, [1, 0], [2, 2])
            >>> a.zip_with(b, lambda x, y: x * y, how="inner", fill_other=0, backend="numpy")
            RleBatch(keys=1, runs=3, dtype=Int64)
              ('chr1',): length=3, runs=[1, 1, 1], values=[0, 1, 0]
            >>> a.zip_with(b, lambda x, y: x + y, how="left", fill_self=0, fill_other=0, backend="numpy")
            RleBatch(keys=2, runs=5, dtype=Int64)
              ('chr1',): length=3, runs=[1, 1, 1], values=[1, 2, 1]
              ('chr2',): length=7, runs=[5, 2], values=[0, 1]
            >>> a + b
            RleBatch(keys=3, runs=6, dtype=Int64)
              ('chr1',): length=3, runs=[1, 1, 1], values=[1, 2, 1]
              ('chr2',): length=7, runs=[5, 2], values=[0, 1]
              ('chr3',): length=2, runs=[2], values=[1]
        """
    def collapse(
        self,
        keep: Sequence[str] | None = None,
        drop: str | Sequence[str] | None = None,
        op: Literal["add", "sub", "mul", "div", "and", "or"] = "add",
    ) -> RleBatch:
        """Combines the Rles whose keys agree on the remaining key columns, with
        `op` ("add", "sub", "mul", "div", "and" or "or"), padding with zero.
        Give the columns to `keep` or to `drop`, for example
        `batch.collapse(drop="Strand")` to add up the strands.

        Examples:
            >>> keys = {"chrom": ["chr1", "chr1", "chr2"], "strand": ["+", "-", "+"]}
            >>> batch = rl.coverage_by_key(keys, [0, 2, 5], [3, 4, 7])
            >>> batch.collapse(drop="strand")
            RleBatch(keys=2, runs=5, dtype=Int64)
              ('chr1',): length=4, runs=[2, 1, 1], values=[1, 2, 1]
              ('chr2',): length=7, runs=[5, 2], values=[0, 1]
        """
    def slice_agg(
        self,
        keys: KeysLike,
        starts: ArrayLike,
        ends: ArrayLike,
        agg: SliceAgg = "mean",
        ddof: int = 1,
    ) -> ArrowArray:
        """Reduces many slices without copying them, in parallel: entry k reduces
        positions starts[k]:ends[k] of the Rle for row k of `keys` (a table
        with the batch's key columns, or one array if there is one key
        column). A key not in the batch gives an empty slice. `agg` and `ddof`
        are as in `Rle.slice_agg`.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
            >>> batch.slice_agg(["chr1", "chr2"], [0, 4], [4, 7], agg="max").to_pylist()
            [2, 1]
        """
    def shift(self, distance: int, fill: Any = ...) -> RleBatch:
        """Shifts every Rle; see `Rle.shift`.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
            >>> batch.shift(1)
            RleBatch(keys=2, runs=4, dtype=Int64)
              ('chr1',): length=3, runs=[1, 2], values=[0, 1]
              ('chr2',): length=7, runs=[6, 1], values=[0, 1]
        """
    def reverse(self) -> RleBatch:
        """Reverses every Rle.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
            >>> batch.reverse()
            RleBatch(keys=2, runs=3, dtype=Int64)
              ('chr1',): length=3, runs=[3], values=[1]
              ('chr2',): length=7, runs=[2, 5], values=[1, 0]
        """
    def sum(self) -> ArrowTable:
        """Sum per key, as a table of the key columns plus `sum`.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
            >>> pl.DataFrame(batch.sum())
            shape: (2, 2)
            ┌───────┬─────┐
            │ chrom ┆ sum │
            │ ---   ┆ --- │
            │ str   ┆ i64 │
            ╞═══════╪═════╡
            │ chr1  ┆ 5   │
            │ chr2  ┆ 2   │
            └───────┴─────┘
        """
    def mean(self) -> ArrowTable:
        """Mean per key, as a table of the key columns plus `mean`.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
            >>> pl.DataFrame(batch.mean())
            shape: (2, 2)
            ┌───────┬──────────┐
            │ chrom ┆ mean     │
            │ ---   ┆ ---      │
            │ str   ┆ f64      │
            ╞═══════╪══════════╡
            │ chr1  ┆ 1.25     │
            │ chr2  ┆ 0.285714 │
            └───────┴──────────┘
        """
    def var(self, ddof: int = 1) -> ArrowTable:
        """Variance per key, as a table of the key columns plus `var`.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
            >>> pl.DataFrame(batch.var())
            shape: (2, 2)
            ┌───────┬──────────┐
            │ chrom ┆ var      │
            │ ---   ┆ ---      │
            │ str   ┆ f64      │
            ╞═══════╪══════════╡
            │ chr1  ┆ 0.25     │
            │ chr2  ┆ 0.238095 │
            └───────┴──────────┘
        """
    def std(self, ddof: int = 1) -> ArrowTable:
        """Standard deviation per key, as a table of the key columns plus `std`.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
            >>> pl.DataFrame(batch.std(ddof=0))
            shape: (2, 2)
            ┌───────┬──────────┐
            │ chrom ┆ std      │
            │ ---   ┆ ---      │
            │ str   ┆ f64      │
            ╞═══════╪══════════╡
            │ chr1  ┆ 0.433013 │
            │ chr2  ┆ 0.451754 │
            └───────┴──────────┘
        """
    def min(self) -> ArrowTable:
        """Minimum per key, as a table of the key columns plus `min`.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
            >>> pl.DataFrame(batch.min())
            shape: (2, 2)
            ┌───────┬─────┐
            │ chrom ┆ min │
            │ ---   ┆ --- │
            │ str   ┆ i64 │
            ╞═══════╪═════╡
            │ chr1  ┆ 1   │
            │ chr2  ┆ 0   │
            └───────┴─────┘
        """
    def max(self) -> ArrowTable:
        """Maximum per key, as a table of the key columns plus `max`.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr1", "chr2"]}, [0, 2, 5], [3, 4, 7])
            >>> pl.DataFrame(batch.max())
            shape: (2, 2)
            ┌───────┬─────┐
            │ chrom ┆ max │
            │ ---   ┆ --- │
            │ str   ┆ i64 │
            ╞═══════╪═════╡
            │ chr1  ┆ 2   │
            │ chr2  ┆ 1   │
            └───────┴─────┘
        """
    def to_arrow(self) -> Any:
        """The runs of all keys as a pyarrow Table: the key columns, then start,
        end and value.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
            >>> batch.to_arrow()
            pyarrow.Table
            chrom: string not null
            start: int64 not null
            end: int64 not null
            value: int64
            ----
            chrom: [["chr1","chr2","chr2"]]
            start: [[0,0,5]]
            end: [[3,5,7]]
            value: [[1,0,1]]
        """
    def to_polars(self) -> Any:
        """The runs of all keys as a polars DataFrame: the key columns, then
        start, end and value.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
            >>> batch.to_polars()
            shape: (3, 4)
            ┌───────┬───────┬─────┬───────┐
            │ chrom ┆ start ┆ end ┆ value │
            │ ---   ┆ ---   ┆ --- ┆ ---   │
            │ str   ┆ i64   ┆ i64 ┆ i64   │
            ╞═══════╪═══════╪═════╪═══════╡
            │ chr1  ┆ 0     ┆ 3   ┆ 1     │
            │ chr2  ┆ 0     ┆ 5   ┆ 0     │
            │ chr2  ┆ 5     ┆ 7   ┆ 1     │
            └───────┴───────┴─────┴───────┘
        """
    def to_pandas(self) -> Any:
        """The runs of all keys as a pandas DataFrame: the key columns, then
        start, end and value.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
            >>> batch.to_pandas()
              chrom  start  end  value
            0  chr1      0    3      1
            1  chr2      0    5      0
            2  chr2      5    7      1
        """
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...
    def map(self, func: Callable[[Any], ArrayLike], backend: Backend = "arrow") -> RleBatch:
        """Applies `func` to the run values of every Rle and merges each result;
        see `Rle.map`. Keys are processed in parallel.

        Examples:
            >>> batch = rl.coverage_by_key({"chrom": ["chr1", "chr2"]}, [0, 5], [3, 7])
            >>> batch.map(lambda v: v > 0, backend="numpy")
            RleBatch(keys=2, runs=3, dtype=Boolean)
              ('chr1',): length=3, runs=[3], values=[true]
              ('chr2',): length=7, runs=[5, 2], values=[false, true]
        """
    def __add__(self, other: RleBatch | Any) -> RleBatch: ...
    def __radd__(self, other: Any) -> RleBatch: ...
    def __sub__(self, other: RleBatch | Any) -> RleBatch: ...
    def __rsub__(self, other: Any) -> RleBatch: ...
    def __mul__(self, other: RleBatch | Any) -> RleBatch: ...
    def __rmul__(self, other: Any) -> RleBatch: ...
    def __truediv__(self, other: RleBatch | Any) -> RleBatch: ...
    def __rtruediv__(self, other: Any) -> RleBatch: ...
    def __and__(self, other: RleBatch | Any) -> RleBatch: ...
    def __rand__(self, other: Any) -> RleBatch: ...
    def __or__(self, other: RleBatch | Any) -> RleBatch: ...
    def __ror__(self, other: Any) -> RleBatch: ...
    def __lt__(self, other: RleBatch | Any) -> RleBatch: ...
    def __le__(self, other: RleBatch | Any) -> RleBatch: ...
    def __eq__(self, other: RleBatch | Any) -> RleBatch: ...  # type: ignore[override]
    def __ne__(self, other: RleBatch | Any) -> RleBatch: ...  # type: ignore[override]
    def __gt__(self, other: RleBatch | Any) -> RleBatch: ...
    def __ge__(self, other: RleBatch | Any) -> RleBatch: ...

Agg = Literal["sum", "count", "max", "min"]

def coverage(
    starts: ArrayLike,
    ends: ArrayLike,
    values: ArrayLike | None = None,
    length: int | None = None,
    agg: Agg = "sum",
    fill: Any = 0,
) -> Rle:
    """Coverage of half-open intervals [start, end) as an Rle.

    `agg` combines the intervals covering each position: "sum" of `values`
    (the number of intervals if no values are given), "count", "max" or
    "min" (numbers or strings). Uncovered positions hold `fill`: by default
    0 for numbers and null for strings; `fill=None` leaves them null. Counts are never missing, so with
    "count" uncovered positions are always 0. The result covers 0..max(end),
    or 0..length if given.

    Examples:
        >>> rl.coverage([0, 2], [3, 4])
        Rle(length=4, runs=3, dtype=Int64)
          runs:   [2, 1, 1]
          values: [1, 2, 1]
        >>> rl.coverage([0, 2], [3, 4], values=[1.5, 2.0])
        Rle(length=4, runs=3, dtype=Float64)
          runs:   [2, 1, 1]
          values: [1.5, 3.5, 2.0]
        >>> rl.coverage([0, 2], [3, 4], values=[1.5, 2.0], agg="max", fill=None, length=6)
        Rle(length=6, runs=3, dtype=Float64)
          runs:   [2, 2, 2]
          values: [1.5, 2.0, null]
    """
def coverage_by_key(
    keys: KeysLike,
    starts: ArrayLike,
    ends: ArrayLike,
    values: ArrayLike | None = None,
    agg: Agg = "sum",
    fill: Any = 0,
) -> RleBatch:
    """Coverage per key, computed in parallel, as an RleBatch.

    `keys` has one row per interval: a polars or pandas DataFrame, a pyarrow
    Table, or a dict of columns (for example Chromosome and Strand). Keys
    appear in order of first appearance; each Rle covers 0..max(end) of its
    own intervals. `agg` and `fill` are as in `coverage`.

    Examples:
        >>> keys = pl.DataFrame({"chrom": ["chr1", "chr1", "chr2"]})
        >>> rl.coverage_by_key(keys, [0, 2, 5], [3, 4, 7])
        RleBatch(keys=2, runs=5, dtype=Int64)
          ('chr1',): length=4, runs=[2, 1, 1], values=[1, 2, 1]
          ('chr2',): length=7, runs=[5, 2], values=[0, 1]
    """
_RleOrBatch = TypeVar("_RleOrBatch", Rle, RleBatch)

def concat(items: Sequence[_RleOrBatch]) -> _RleOrBatch:
    """Joins Rles end to end, merging identical neighbours. Given RleBatches,
    joins them key by key: each key's Rles are joined in order, and a batch
    without the key adds nothing.

    Examples:
        >>> rl.concat([rl.Rle([2], [1]), rl.Rle([1, 2], [1, 3])])
        Rle(length=5, runs=2, dtype=Int64)
          runs:   [3, 2]
          values: [1, 3]
    """
def plan_many(rles: Sequence[Rle]) -> tuple[ArrowArray, list[ArrowArray]]:
    """The plan for combining several Rles: (ends, [idx per Rle]). An index
    equal to that Rle's number of runs means the segment is past its end.

    Examples:
        >>> ends, (i, j, k) = rl.plan_many([rl.Rle([2, 2], [1, 2]), rl.Rle([3], [5]), rl.Rle([1, 3], [7, 8])])
        >>> ends.to_pylist(), i.to_pylist(), j.to_pylist(), k.to_pylist()
        ([1, 2, 3, 4], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 1])
    """
def zip_many(
    rles: Sequence[Rle], func: Callable[..., ArrayLike], fill: Any = None, backend: Backend = "arrow"
) -> Rle:
    """Combines several Rles position by position: `func(*arrays)` receives one
    array per Rle (their values on each segment) and returns one value per
    segment. Shorter Rles are padded with `fill` (one value for all; by
    default unequal lengths are an error). `backend` is as in `Rle.zip_with`.

    Examples:
        >>> rles = [rl.Rle([2, 2], [1, 2]), rl.Rle([3, 1], [5, 6]), rl.Rle([4], [100])]
        >>> rl.zip_many(rles, lambda a, b, c: a + b + c, backend="numpy")
        Rle(length=4, runs=3, dtype=Int64)
          runs:   [2, 1, 1]
          values: [106, 107, 108]
    """
def overlap_plan(
    starts: ArrayLike, ends: ArrayLike, length: int | None = None
) -> tuple[ArrowArray, ArrowArray, ArrowArray]:
    """Which intervals cover each segment: (ends, offsets, ids), where segment s
    ends at ends[s] and is covered by intervals ids[offsets[s]:offsets[s+1]].
    Aggregate interval values per segment in any way, for example with a
    polars group_by.

    Examples:
        >>> ends, offsets, ids = rl.overlap_plan([0, 2], [3, 4])
        >>> ends.to_pylist(), offsets.to_pylist(), ids.to_pylist()
        ([2, 3, 4], [0, 1, 3, 4], [0, 0, 1, 1])
    """
def overlap_plan_by_key(keys: KeysLike, starts: ArrayLike, ends: ArrayLike) -> ArrowTable:
    """`overlap_plan` per key, as one table: the key columns, then start and end
    of each segment and ids, a list of the intervals covering it (row numbers
    in the input). `keys` is as in `coverage_by_key`. Explode ids and join
    interval values onto them to aggregate per segment in any way.

    Examples:
        >>> keys = {"chrom": ["chr1", "chr1", "chr2"]}
        >>> pl.DataFrame(rl.overlap_plan_by_key(keys, [0, 2, 5], [3, 4, 7]))
        shape: (5, 4)
        ┌───────┬───────┬─────┬───────────┐
        │ chrom ┆ start ┆ end ┆ ids       │
        │ ---   ┆ ---   ┆ --- ┆ ---       │
        │ str   ┆ i64   ┆ i64 ┆ list[i64] │
        ╞═══════╪═══════╪═════╪═══════════╡
        │ chr1  ┆ 0     ┆ 2   ┆ [0]       │
        │ chr1  ┆ 2     ┆ 3   ┆ [0, 1]    │
        │ chr1  ┆ 3     ┆ 4   ┆ [1]       │
        │ chr2  ┆ 0     ┆ 5   ┆ []        │
        │ chr2  ┆ 5     ┆ 7   ┆ [2]       │
        └───────┴───────┴─────┴───────────┘
    """
