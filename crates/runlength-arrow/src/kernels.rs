//! Element-wise kernels in the shape `zip_with` expects.
//!
//! Each takes two arrays of equal length and returns one value per position.
//! They wrap `arrow-arith` and `arrow-ord`, so integer overflow and integer
//! division by zero are errors.

use std::sync::Arc;

use arrow_arith::{boolean, numeric};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray};
use arrow_ord::cmp;
use arrow_schema::ArrowError;

/// A binary element-wise kernel.
pub type Kernel = fn(&dyn Array, &dyn Array) -> Result<ArrayRef, ArrowError>;

pub fn add(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    numeric::add(&a, &b)
}

pub fn sub(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    numeric::sub(&a, &b)
}

pub fn mul(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    numeric::mul(&a, &b)
}

pub fn div(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    numeric::div(&a, &b)
}

pub fn eq(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::eq(&a, &b)?))
}

pub fn ne(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::neq(&a, &b)?))
}

pub fn lt(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::lt(&a, &b)?))
}

pub fn le(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::lt_eq(&a, &b)?))
}

pub fn gt(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::gt(&a, &b)?))
}

pub fn ge(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::gt_eq(&a, &b)?))
}

/// Logical and, with SQL semantics for nulls.
pub fn and(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    logical(a, b, boolean::and_kleene)
}

/// Logical or, with SQL semantics for nulls.
pub fn or(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    logical(a, b, boolean::or_kleene)
}

fn logical(
    a: &dyn Array,
    b: &dyn Array,
    f: fn(&BooleanArray, &BooleanArray) -> Result<BooleanArray, ArrowError>,
) -> Result<ArrayRef, ArrowError> {
    match (a.as_boolean_opt(), b.as_boolean_opt()) {
        (Some(a), Some(b)) => Ok(Arc::new(f(a, b)?)),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "logical operations need boolean values, got {} and {}",
            a.data_type(),
            b.data_type()
        ))),
    }
}

/// Defines one method per kernel on a type with an
/// `op(&self, &Self, Kernel) -> Result<Self>` method.
macro_rules! kernel_methods {
    ($Type:ident, $note:literal) => {
        #[allow(clippy::should_implement_trait)]
        impl $Type {
            #[doc = concat!("Element-wise addition (integer overflow is an error).", $note)]
            pub fn add(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::add)
            }
            #[doc = concat!("Element-wise subtraction (integer overflow is an error).", $note)]
            pub fn sub(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::sub)
            }
            #[doc = concat!("Element-wise multiplication (integer overflow is an error).", $note)]
            pub fn mul(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::mul)
            }
            #[doc = concat!("Element-wise division (integer division by zero is an error).", $note)]
            pub fn div(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::div)
            }
            #[doc = concat!("Element-wise `==`.", $note)]
            pub fn elem_eq(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::eq)
            }
            #[doc = concat!("Element-wise `!=`.", $note)]
            pub fn elem_ne(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::ne)
            }
            #[doc = concat!("Element-wise `<`.", $note)]
            pub fn elem_lt(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::lt)
            }
            #[doc = concat!("Element-wise `<=`.", $note)]
            pub fn elem_le(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::le)
            }
            #[doc = concat!("Element-wise `>`.", $note)]
            pub fn elem_gt(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::gt)
            }
            #[doc = concat!("Element-wise `>=`.", $note)]
            pub fn elem_ge(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::ge)
            }
            #[doc = concat!("Element-wise logical and, with SQL semantics for nulls.", $note)]
            pub fn and(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::and)
            }
            #[doc = concat!("Element-wise logical or, with SQL semantics for nulls.", $note)]
            pub fn or(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::or)
            }
        }
    };
}

pub(crate) use kernel_methods;
