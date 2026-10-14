//! Dispatch from an Arrow data type to the core crate's numeric types.

use arrow_schema::DataType;

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
/// themselves if equal, `Float64` if either is a float, `Int64` otherwise.
/// `None` if either is not numeric.
pub fn common_numeric_type(a: &DataType, b: &DataType) -> Option<DataType> {
    if !a.is_numeric() || !b.is_numeric() {
        return None;
    }
    if a == b {
        return Some(a.clone());
    }
    if a.is_floating() || b.is_floating() {
        Some(DataType::Float64)
    } else {
        Some(DataType::Int64)
    }
}
