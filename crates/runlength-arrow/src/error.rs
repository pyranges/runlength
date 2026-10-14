use std::fmt;

use arrow_schema::{ArrowError, DataType};

/// Errors from Arrow-backed sequences.
#[derive(Debug)]
pub enum Error {
    /// An error from the core crate, such as invalid runs or unequal lengths.
    Core(runlength::Error),
    /// An error from an Arrow kernel, such as mismatched types or overflow.
    Arrow(ArrowError),
    /// A fill value was not a single value of the right type.
    InvalidFill {
        expected: DataType,
        actual: DataType,
        len: usize,
    },
    /// An operation returned a different number of values than it was given.
    WrongLength { expected: usize, actual: usize },
    /// Two batches have keys with different column names or types.
    KeySchemaMismatch,
    /// The values have a type this operation does not support.
    UnsupportedType(DataType),
    /// The values contain nulls where they are not allowed.
    UnexpectedNulls,
    /// An aggregation needs values but none were given.
    MissingValues,
    /// An operation needs at least one input but was given none.
    NoInputs,
    /// An operation failed for the key at `position` in the output.
    InKey { position: usize, source: Box<Error> },
}

impl Error {
    pub(crate) fn in_key(self, position: usize) -> Self {
        Error::InKey {
            position,
            source: Box::new(self),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Core(e) => write!(f, "{e}"),
            Error::Arrow(e) => write!(f, "{e}"),
            Error::InvalidFill {
                expected,
                actual,
                len,
            } => write!(
                f,
                "fill must be one value of type {expected}, got {len} value(s) of type {actual}"
            ),
            Error::WrongLength { expected, actual } => write!(
                f,
                "operation returned {actual} values for {expected} segments"
            ),
            Error::KeySchemaMismatch => {
                write!(f, "the batches' key columns have different names or types")
            }
            Error::UnsupportedType(t) => write!(f, "unsupported value type {t}"),
            Error::UnexpectedNulls => write!(f, "values must not contain nulls"),
            Error::MissingValues => write!(f, "this aggregation needs values"),
            Error::NoInputs => write!(f, "this operation needs at least one input"),
            Error::InKey { position, source } => {
                write!(f, "key at output position {position}: {source}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Core(e) => Some(e),
            Error::Arrow(e) => Some(e),
            Error::InKey { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<runlength::Error> for Error {
    fn from(e: runlength::Error) -> Self {
        Error::Core(e)
    }
}

impl From<ArrowError> for Error {
    fn from(e: ArrowError) -> Self {
        Error::Arrow(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
