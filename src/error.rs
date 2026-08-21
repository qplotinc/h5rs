//! Error type returned by all fallible h5rs operations.

use std::fmt;

/// Anything that can go wrong while reading an HDF5 file.
///
/// h5rs reads files it did not write, often over a network, so every failure
/// mode here is reachable from untrusted input: none of them panic.
#[derive(Debug)]
pub enum H5Error {
    /// The bytes read did not parse as the expected HDF5 structure.
    Parse(binrw::Error),
    /// The underlying object store failed to serve a byte range.
    Store(object_store::Error),
    /// The file uses a valid HDF5 feature that h5rs does not implement yet.
    /// See the crate docs for the supported subset of the format.
    Unsupported(String),
    /// The Rust element type requested does not match the dataset's on-disk
    /// datatype. Reading the data as that type would reinterpret the bytes.
    TypeMismatch {
        /// The Rust type the caller asked for.
        rust_type: &'static str,
        /// A description of what the file actually holds.
        hdf5_type: String,
    },
    /// The caller's selection does not describe a valid region of the dataset.
    InvalidSelection(String),
    /// The file parsed, but its contents are internally inconsistent — for
    /// example a chunk whose recorded size disagrees with the chunk layout.
    Corrupt(String),
}

impl H5Error {
    /// Build a [`H5Error::TypeMismatch`] for the Rust type `T`.
    pub(crate) fn type_mismatch<T>(hdf5_type: impl Into<String>) -> Self {
        H5Error::TypeMismatch {
            rust_type: std::any::type_name::<T>(),
            hdf5_type: hdf5_type.into(),
        }
    }

    /// Build a [`H5Error::Unsupported`].
    pub(crate) fn unsupported(what: impl Into<String>) -> Self {
        H5Error::Unsupported(what.into())
    }

    /// Build a [`H5Error::Corrupt`].
    pub(crate) fn corrupt(what: impl Into<String>) -> Self {
        H5Error::Corrupt(what.into())
    }
}

impl fmt::Display for H5Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            H5Error::Parse(e) => write!(f, "HDF5 parse error: {e}"),
            H5Error::Store(e) => write!(f, "Object store error: {e}"),
            H5Error::Unsupported(what) => write!(f, "Unsupported HDF5 feature: {what}"),
            H5Error::TypeMismatch {
                rust_type,
                hdf5_type,
            } => write!(
                f,
                "Datatype mismatch: requested {rust_type}, but the file holds {hdf5_type}"
            ),
            H5Error::InvalidSelection(what) => write!(f, "Invalid selection: {what}"),
            H5Error::Corrupt(what) => write!(f, "Malformed HDF5 file: {what}"),
        }
    }
}

impl std::error::Error for H5Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            H5Error::Parse(e) => Some(e),
            H5Error::Store(e) => Some(e),
            H5Error::Unsupported(_)
            | H5Error::TypeMismatch { .. }
            | H5Error::InvalidSelection(_)
            | H5Error::Corrupt(_) => None,
        }
    }
}

impl From<binrw::Error> for H5Error {
    fn from(e: binrw::Error) -> Self {
        H5Error::Parse(e)
    }
}

impl From<object_store::Error> for H5Error {
    fn from(e: object_store::Error) -> Self {
        H5Error::Store(e)
    }
}

/// `Result` alias used throughout the crate.
pub type H5Result<T> = Result<T, H5Error>;
