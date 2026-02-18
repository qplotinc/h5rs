use std::fmt;

#[derive(Debug)]
pub enum H5Error {
    Parse(binrw::Error),
    Store(object_store::Error),
}

impl fmt::Display for H5Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            H5Error::Parse(e) => write!(f, "HDF5 parse error: {e}"),
            H5Error::Store(e) => write!(f, "Object store error: {e}"),
        }
    }
}

impl std::error::Error for H5Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            H5Error::Parse(e) => Some(e),
            H5Error::Store(e) => Some(e),
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

pub type H5Result<T> = Result<T, H5Error>;
