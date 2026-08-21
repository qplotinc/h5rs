//! Mapping between Rust primitive types and HDF5 scalar datatypes.

use crate::error::{H5Error, H5Result};
use crate::format::object::{DatatypeMessage, TypeDescriptor};

/// Trait for Rust types that correspond to HDF5 scalar datatypes.
///
/// Implementors must ensure that the type's in-memory layout exactly matches
/// the HDF5 on-disk layout when `check_dtype` succeeds.  The `bytemuck::Pod`
/// bound already enforces the memory-safety invariant; `check_dtype` verifies
/// the size/signedness/class at runtime.
pub trait H5Type: bytemuck::Pod {
    /// Return [`H5Error::TypeMismatch`] if the HDF5 datatype does not exactly
    /// match this Rust type's layout.
    fn check_dtype(dtype: &DatatypeMessage) -> H5Result<()>;
}

/// Describe an on-disk datatype for error messages.
fn describe(desc: &TypeDescriptor) -> String {
    match desc {
        TypeDescriptor::FixedPoint(fp) => format!(
            "a {}-byte {} integer",
            fp.size(),
            if fp.signed() != 0 {
                "signed"
            } else {
                "unsigned"
            }
        ),
        TypeDescriptor::FloatingPoint(fp) => format!("a {}-byte float", fp.size()),
        TypeDescriptor::String(s) => format!("a {}-byte string", s.size()),
        TypeDescriptor::Variable(_) => "a variable-length type".to_string(),
        TypeDescriptor::UnimplementedTypeClass => "an unsupported type class".to_string(),
    }
}

macro_rules! impl_fixed_point {
    ($ty:ty, $signed:expr) => {
        impl H5Type for $ty {
            fn check_dtype(dtype: &DatatypeMessage) -> H5Result<()> {
                let TypeDescriptor::FixedPoint(ref fp) = dtype.type_desc else {
                    return Err(H5Error::type_mismatch::<$ty>(describe(&dtype.type_desc)));
                };
                if fp.signed() != $signed || fp.size() as usize != std::mem::size_of::<$ty>() {
                    return Err(H5Error::type_mismatch::<$ty>(describe(&dtype.type_desc)));
                }
                Ok(())
            }
        }
    };
}

macro_rules! impl_float {
    ($ty:ty) => {
        impl H5Type for $ty {
            fn check_dtype(dtype: &DatatypeMessage) -> H5Result<()> {
                let TypeDescriptor::FloatingPoint(ref fp) = dtype.type_desc else {
                    return Err(H5Error::type_mismatch::<$ty>(describe(&dtype.type_desc)));
                };
                if fp.size() as usize != std::mem::size_of::<$ty>() {
                    return Err(H5Error::type_mismatch::<$ty>(describe(&dtype.type_desc)));
                }
                Ok(())
            }
        }
    };
}

// Unsigned integers
impl_fixed_point!(u8, 0);
impl_fixed_point!(u16, 0);
impl_fixed_point!(u32, 0);
impl_fixed_point!(u64, 0);

// Signed integers
impl_fixed_point!(i8, 1);
impl_fixed_point!(i16, 1);
impl_fixed_point!(i32, 1);
impl_fixed_point!(i64, 1);

// Floating point
impl_float!(f32);
impl_float!(f64);
