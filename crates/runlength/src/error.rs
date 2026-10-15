use std::fmt;

/// Errors raised when input is invalid or an operation cannot be carried out.
///
/// Input is validated once, when an [`Rle`](crate::Rle) or
/// [`RleBatch`](crate::RleBatch) is constructed. Operations on valid data only
/// fail for reasons that depend on the operation, such as unequal lengths with
/// no fill value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A run length was negative.
    NegativeRun {
        /// The position of the run.
        index: usize,
        /// Its length.
        run: i64,
    },
    /// Two inputs that must have the same number of elements did not.
    LengthMismatch {
        /// What the two inputs were.
        what: &'static str,
        /// The number of elements in the first.
        left: usize,
        /// The number of elements in the second.
        right: usize,
    },
    /// Run ends were not strictly increasing and positive.
    InvalidEnds {
        /// The position of the first invalid end.
        index: usize,
    },
    /// An interval had a negative start or ended before it started.
    InvalidInterval {
        /// The position of the interval.
        index: usize,
        /// Its start.
        start: i64,
        /// Its end.
        end: i64,
    },
    /// Batch offsets did not start at 0, decreased, or did not end at the
    /// number of runs.
    InvalidOffsets,
    /// The total length did not fit in an `i64`.
    Overflow,
    /// Two sequences had different lengths and the shorter had no fill value.
    UnequalLength {
        /// The length of the left sequence.
        left: i64,
        /// The length of the right sequence.
        right: i64,
    },
    /// A requested length was shorter than the sequence it applies to.
    LengthTooShort {
        /// The length asked for.
        requested: i64,
        /// The length of the sequence.
        required: i64,
    },
    /// A key occurred more than once in a batch.
    DuplicateKey {
        /// The position of the second occurrence.
        position: usize,
    },
    /// An operation failed for the key at `position` in the output.
    InKey {
        /// The position of the key in the output.
        position: usize,
        /// What went wrong.
        source: Box<Error>,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NegativeRun { index, run } => {
                write!(f, "run {index} has negative length {run}")
            }
            Error::LengthMismatch { what, left, right } => {
                write!(f, "{what} have different lengths ({left} and {right})")
            }
            Error::InvalidEnds { index } => write!(
                f,
                "run end {index} is not greater than the previous end (or 0)"
            ),
            Error::InvalidInterval { index, start, end } => write!(
                f,
                "interval {index} [{start}, {end}) has a negative start or ends before it starts"
            ),
            Error::InvalidOffsets => write!(
                f,
                "offsets must start at 0, never decrease and end at the number of runs"
            ),
            Error::Overflow => write!(f, "total length does not fit in an i64"),
            Error::UnequalLength { left, right } => write!(
                f,
                "lengths differ ({left} and {right}) and the shorter side has no fill value"
            ),
            Error::LengthTooShort {
                requested,
                required,
            } => write!(
                f,
                "requested length {requested} is shorter than the existing length {required}"
            ),
            Error::DuplicateKey { position } => {
                write!(f, "key at position {position} already exists in the batch")
            }
            Error::InKey { position, source } => {
                write!(f, "key at output position {position}: {source}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::InKey { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// A `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
