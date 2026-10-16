//! Dispatch from an Arrow data type to the core crate's numeric types.

use arrow_array::{Array, ArrayRef};
use arrow_cast::{CastOptions, cast_with_options};
use arrow_schema::DataType;

use crate::error::Result;

/// Runs `$body` with `$T` bound to the Arrow type of `$values`, for the
/// numeric types coverage supports.
macro_rules! with_numeric_type {
    ($values:expr, |$T:ident| $body:expr) => {
        match $values.data_type() {
            ::arrow_schema::DataType::Int8 => {
                type $T = ::arrow_array::types::Int8Type;
                $body
            }
            ::arrow_schema::DataType::Int16 => {
                type $T = ::arrow_array::types::Int16Type;
                $body
            }
            ::arrow_schema::DataType::Int32 => {
                type $T = ::arrow_array::types::Int32Type;
                $body
            }
            ::arrow_schema::DataType::Int64 => {
                type $T = ::arrow_array::types::Int64Type;
                $body
            }
            ::arrow_schema::DataType::UInt8 => {
                type $T = ::arrow_array::types::UInt8Type;
                $body
            }
            ::arrow_schema::DataType::UInt16 => {
                type $T = ::arrow_array::types::UInt16Type;
                $body
            }
            ::arrow_schema::DataType::UInt32 => {
                type $T = ::arrow_array::types::UInt32Type;
                $body
            }
            ::arrow_schema::DataType::UInt64 => {
                type $T = ::arrow_array::types::UInt64Type;
                $body
            }
            ::arrow_schema::DataType::Float32 => {
                type $T = ::arrow_array::types::Float32Type;
                $body
            }
            ::arrow_schema::DataType::Float64 => {
                type $T = ::arrow_array::types::Float64Type;
                $body
            }
            other => Err($crate::Error::UnsupportedType(other.clone())),
        }
    };
}

pub(crate) use with_numeric_type;

/// The type two numeric types are converted to before combining them:
/// themselves if equal, `Float64` if either is a float, `UInt64` if both are
/// unsigned, `Int64` otherwise. `None` if either is not numeric.
///
/// Converting to `Int64` fails for unsigned values above `i64::MAX`; see
/// [`cast_checked`].
pub fn common_numeric_type(a: &DataType, b: &DataType) -> Option<DataType> {
    if !a.is_numeric() || !b.is_numeric() {
        return None;
    }
    if a == b {
        return Some(a.clone());
    }
    if a.is_floating() || b.is_floating() {
        Some(DataType::Float64)
    } else if a.is_unsigned_integer() && b.is_unsigned_integer() {
        Some(DataType::UInt64)
    } else {
        Some(DataType::Int64)
    }
}

/// Casts `array` to `data_type`, failing instead of turning values that
/// don't fit (such as 300 as an `Int8`) into nulls.
pub fn cast_checked(array: &dyn Array, data_type: &DataType) -> Result<ArrayRef> {
    let options = CastOptions {
        safe: false,
        ..Default::default()
    };
    Ok(cast_with_options(array, data_type, &options)?)
}
