use crate::format::object::{DatatypeMessage, TypeDescriptor};

/// Trait for Rust types that correspond to HDF5 scalar datatypes.
///
/// Implementors must ensure that the type's in-memory layout exactly matches
/// the HDF5 on-disk layout when `check_dtype` succeeds.  The `bytemuck::Pod`
/// bound already enforces the memory-safety invariant; `check_dtype` verifies
/// the size/signedness/class at runtime.
pub trait H5Type: bytemuck::Pod {
    /// Panic if the HDF5 datatype does not exactly match this Rust type's layout.
    fn check_dtype(dtype: &DatatypeMessage);
}

macro_rules! impl_fixed_point {
    ($ty:ty, $signed:expr) => {
        impl H5Type for $ty {
            fn check_dtype(dtype: &DatatypeMessage) {
                let TypeDescriptor::FixedPoint(ref fp) = dtype.type_desc else {
                    panic!(
                        "expected FixedPoint for {}, got {:?}",
                        stringify!($ty),
                        dtype.type_desc
                    );
                };
                assert_eq!(
                    fp.signed(),
                    $signed,
                    "signedness mismatch for {}",
                    stringify!($ty)
                );
                assert_eq!(
                    fp.size() as usize,
                    std::mem::size_of::<$ty>(),
                    "size mismatch for {}: HDF5 has {} bytes",
                    stringify!($ty),
                    fp.size()
                );
            }
        }
    };
}

macro_rules! impl_float {
    ($ty:ty) => {
        impl H5Type for $ty {
            fn check_dtype(dtype: &DatatypeMessage) {
                let TypeDescriptor::FloatingPoint(ref fp) = dtype.type_desc else {
                    panic!(
                        "expected FloatingPoint for {}, got {:?}",
                        stringify!($ty),
                        dtype.type_desc
                    );
                };
                assert_eq!(
                    fp.size() as usize,
                    std::mem::size_of::<$ty>(),
                    "size mismatch for {}: HDF5 has {} bytes",
                    stringify!($ty),
                    fp.size()
                );
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
