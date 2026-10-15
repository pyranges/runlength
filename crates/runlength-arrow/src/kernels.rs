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

/// A built-in element-wise operation, as the operators use them.
///
/// Unlike an arbitrary function, a `Kernel` tells the sequence types which
/// operation it is, so they can combine numbers without nulls in a single
/// pass instead of through intermediate arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// [`add`]
    Add,
    /// [`sub`]
    Sub,
    /// [`mul`]
    Mul,
    /// [`div`]
    Div,
    /// [`eq`]
    Eq,
    /// [`ne`]
    Ne,
    /// [`lt`]
    Lt,
    /// [`le`]
    Le,
    /// [`gt`]
    Gt,
    /// [`ge`]
    Ge,
    /// [`and`]
    And,
    /// [`or`]
    Or,
}

impl Kernel {
    /// Applies the operation to two arrays of equal length.
    pub fn apply(self, a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
        match self {
            Kernel::Add => add(a, b),
            Kernel::Sub => sub(a, b),
            Kernel::Mul => mul(a, b),
            Kernel::Div => div(a, b),
            Kernel::Eq => eq(a, b),
            Kernel::Ne => ne(a, b),
            Kernel::Lt => lt(a, b),
            Kernel::Le => le(a, b),
            Kernel::Gt => gt(a, b),
            Kernel::Ge => ge(a, b),
            Kernel::And => and(a, b),
            Kernel::Or => or(a, b),
        }
    }
}

/// `a + b`.
pub fn add(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    numeric::add(&a, &b)
}

/// `a - b`.
pub fn sub(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    numeric::sub(&a, &b)
}

/// `a * b`.
pub fn mul(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    numeric::mul(&a, &b)
}

/// `a / b`; integer division by zero is an error.
pub fn div(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    numeric::div(&a, &b)
}

/// `a == b`, as booleans (null where either side is null).
pub fn eq(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::eq(&a, &b)?))
}

/// `a != b`, as booleans.
pub fn ne(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::neq(&a, &b)?))
}

/// `a < b`, as booleans.
pub fn lt(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::lt(&a, &b)?))
}

/// `a <= b`, as booleans.
pub fn le(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::lt_eq(&a, &b)?))
}

/// `a > b`, as booleans.
pub fn gt(a: &dyn Array, b: &dyn Array) -> Result<ArrayRef, ArrowError> {
    Ok(Arc::new(cmp::gt(&a, &b)?))
}

/// `a >= b`, as booleans.
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
                self.op(other, $crate::kernels::Kernel::Add)
            }
            #[doc = concat!("Element-wise subtraction (integer overflow is an error).", $note)]
            pub fn sub(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Sub)
            }
            #[doc = concat!("Element-wise multiplication (integer overflow is an error).", $note)]
            pub fn mul(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Mul)
            }
            #[doc = concat!("Element-wise division (integer division by zero is an error).", $note)]
            pub fn div(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Div)
            }
            #[doc = concat!("Element-wise `==`.", $note)]
            pub fn elem_eq(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Eq)
            }
            #[doc = concat!("Element-wise `!=`.", $note)]
            pub fn elem_ne(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Ne)
            }
            #[doc = concat!("Element-wise `<`.", $note)]
            pub fn elem_lt(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Lt)
            }
            #[doc = concat!("Element-wise `<=`.", $note)]
            pub fn elem_le(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Le)
            }
            #[doc = concat!("Element-wise `>`.", $note)]
            pub fn elem_gt(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Gt)
            }
            #[doc = concat!("Element-wise `>=`.", $note)]
            pub fn elem_ge(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Ge)
            }
            #[doc = concat!("Element-wise logical and, with SQL semantics for nulls.", $note)]
            pub fn and(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::And)
            }
            #[doc = concat!("Element-wise logical or, with SQL semantics for nulls.", $note)]
            pub fn or(&self, other: &$Type) -> Result<$Type> {
                self.op(other, $crate::kernels::Kernel::Or)
            }
        }
    };
}

pub(crate) use kernel_methods;
