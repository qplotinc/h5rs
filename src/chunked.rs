//! Reading chunked datasets: chunk B-tree traversal, filter decoding, and
//! sub-region assembly.

use std::ops::Range;

use crate::error::{H5Error, H5Result};
use crate::format::{
    chunk_index::{ChunkRecord, enumerate_chunks},
    object::{ChunkedLayout, DataspaceMessage, DatatypeMessage, FilterMessage, FilterType},
};
use crate::h5type::H5Type;
use crate::object_store::{ObjectStoreFile, fetch_exact};
use std::io::Read;

/// N-dimensional array wrapper that pairs a flat data vector with its logical shape.
/// Data is in row-major (C) order.
#[derive(Debug, Clone)]
pub struct NdArray<T> {
    /// Elements in row-major (C) order. Length is the product of `shape`.
    pub data: Vec<T>,
    /// Length of each dimension, outermost first.
    pub shape: Vec<usize>,
}

/// Compute row-major strides for a given shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Copy a rectangular sub-region from raw bytes (with element layout of size
/// `elem_size`) into a destination byte buffer.  Both buffers are N-dimensional
/// arrays stored in row-major order.  Strides, offsets, and sizes are all
/// expressed in *elements*, not bytes.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn copy_region_inner(
    src: &[u8],
    elem_size: usize,
    src_strides: &[usize],
    src_offset: &[usize],
    dst: &mut [u8],
    dst_strides: &[usize],
    dst_offset: &[usize],
    size: &[usize],
    dim: usize,
) {
    let ndim = size.len();
    if dim == ndim - 1 {
        // Innermost dimension: contiguous byte copy
        let src_idx: usize = src_offset
            .iter()
            .zip(src_strides.iter())
            .map(|(&o, &s)| o * s)
            .sum();
        let dst_idx: usize = dst_offset
            .iter()
            .zip(dst_strides.iter())
            .map(|(&o, &s)| o * s)
            .sum();
        let sb = src_idx * elem_size;
        let db = dst_idx * elem_size;
        let len = size[dim] * elem_size;
        dst[db..db + len].copy_from_slice(&src[sb..sb + len]);
    } else {
        let mut src_off: Vec<usize> = src_offset.to_vec();
        let mut dst_off: Vec<usize> = dst_offset.to_vec();
        for i in 0..size[dim] {
            src_off[dim] = src_offset[dim] + i;
            dst_off[dim] = dst_offset[dim] + i;
            copy_region_inner(
                src,
                elem_size,
                src_strides,
                &src_off,
                dst,
                dst_strides,
                &dst_off,
                size,
                dim + 1,
            );
        }
    }
}

/// A chunked HDF5 dataset, opened and ready to read.
///
/// Holds only the dataset's metadata — the dataspace, datatype, chunk layout
/// and filter pipeline. No bulk data is fetched until you call
/// [`read_full`](Self::read_full) or [`read_range`](Self::read_range), and a
/// range read fetches only the chunks that overlap the selection.
pub struct ChunkedDataset {
    pub(crate) name: String,
    pub(crate) dataspace: DataspaceMessage,
    pub(crate) datatype: DatatypeMessage,
    pub(crate) chunks_layout: ChunkedLayout,
    pub(crate) filter: Option<FilterMessage>,
}

impl ChunkedDataset {
    /// The dataset's name (its last path segment).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns human-readable names of the HDF5 filters (compression, shuffle, etc.)
    pub fn filter_names(&self) -> Vec<String> {
        match &self.filter {
            Some(fm) => fm
                .filters
                .iter()
                .map(|fd| format!("{:?}", fd.filter_type))
                .collect(),
            None => vec![],
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dataspace.dimensionality as usize
    }

    /// Dataset extent along each dimension.
    fn dataset_dims(&self) -> &[u64] {
        &self.dataspace.dimension[..self.ndim()]
    }

    /// Length of each dataset dimension, outermost first.
    pub fn shape(&self) -> Vec<u64> {
        self.dataspace.dimension[..self.ndim()].to_vec()
    }

    /// Length of each chunk dimension, outermost first.
    pub fn chunk_shape(&self) -> Vec<u64> {
        self.chunks_layout.chunk_dims.clone()
    }

    /// Name of the structure used to locate this dataset's chunks, for
    /// diagnostics: `"v1 btree"`, `"single chunk"`, `"implicit"`,
    /// `"fixed array"`, `"extensible array"` or `"v2 btree"`.
    pub fn chunk_index_name(&self) -> &'static str {
        use crate::format::object::ChunkIndex::*;
        match self.chunks_layout.index {
            BTreeV1 { .. } => "v1 btree",
            SingleChunk { .. } => "single chunk",
            Implicit { .. } => "implicit",
            FixedArray { .. } => "fixed array",
            ExtensibleArray { .. } => "extensible array",
            BTreeV2 { .. } => "v2 btree",
        }
    }

    /// Returns (is_float, is_signed, byte_size) for the dataset's scalar type.
    pub fn dtype_info(&self) -> (bool, bool, usize) {
        use crate::format::object::TypeDescriptor;
        match &self.datatype.type_desc {
            TypeDescriptor::FloatingPoint(fp) => (true, true, fp.size() as usize),
            TypeDescriptor::FixedPoint(fp) => (false, fp.signed() != 0, fp.size() as usize),
            _ => (false, false, 0),
        }
    }

    /// Reject filter pipelines h5rs cannot decode, before any chunk is fetched.
    ///
    /// Only deflate (gzip) and shuffle are implemented. Silently returning
    /// garbage for szip or a checksum filter would be worse than an error.
    fn check_filters_supported(&self) -> H5Result<()> {
        let Some(fm) = &self.filter else {
            return Ok(());
        };
        for fd in &fm.filters {
            match &fd.filter_type {
                FilterType::None | FilterType::Deflate | FilterType::Shuffle => {}
                other => {
                    return Err(H5Error::unsupported(format!(
                        "{other:?} filter (only Deflate and Shuffle are implemented)"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Undo the filter pipeline for one chunk.
    ///
    /// Filters are applied in order on write, so they are undone in reverse.
    /// A chunk whose data a filter would have grown is stored with that filter
    /// skipped and its bit set in the chunk's filter mask, so the mask has to be
    /// honoured or such a chunk decodes to garbage.
    ///
    /// Returns `None` when no filter actually ran, letting the caller use the
    /// fetched bytes without copying them.
    fn decode_filters(
        &self,
        c: &ChunkRecord,
        stored: &[u8],
        uncompressed_bytes: usize,
    ) -> H5Result<Option<Vec<u8>>> {
        let Some(pipeline) = &self.filter else {
            return Ok(None);
        };

        let mut decoded: Option<Vec<u8>> = None;
        for (i, fd) in pipeline.filters.iter().enumerate().rev() {
            if c.filter_mask & (1u32 << i) != 0 {
                continue;
            }
            let input: &[u8] = decoded.as_deref().unwrap_or(stored);
            decoded = match fd.filter_type {
                FilterType::None => continue,
                FilterType::Deflate => {
                    let mut out = Vec::with_capacity(uncompressed_bytes);
                    let mut decoder = flate2::read::ZlibDecoder::new(input);
                    decoder
                        .read_to_end(&mut out)
                        .map_err(|e| binrw::Error::Custom {
                            pos: c.address,
                            err: Box::new(e),
                        })?;
                    Some(out)
                }
                FilterType::Shuffle => Some(self.unshuffle(input)?),
                // check_filters_supported rejects everything else up front.
                ref other => {
                    return Err(H5Error::unsupported(format!("{other:?} filter")));
                }
            };
        }
        Ok(decoded)
    }

    /// Reverse the shuffle filter, which stores all the first bytes of each
    /// element, then all the second bytes, and so on.
    fn unshuffle(&self, data: &[u8]) -> H5Result<Vec<u8>> {
        let element_size = self.chunks_layout.element_size as usize;
        if element_size == 0 {
            return Err(H5Error::corrupt(
                "chunk layout has no element size for the shuffle filter",
            ));
        }
        let num_elements = data.len() / element_size;
        let mut out = vec![0u8; data.len()];
        for i in 0..num_elements {
            for b in 0..element_size {
                out[i * element_size + b] = data[b * num_elements + i];
            }
        }
        Ok(out)
    }

    /// Decode a chunk and copy the specified sub-region directly into `dst`.
    /// For unfiltered data, copies directly from the fetched bytes into `dst`
    /// with no intermediate buffer.
    #[allow(clippy::too_many_arguments)]
    async fn read_chunk_into<T: H5Type>(
        &self,
        c: &ChunkRecord,
        file: &ObjectStoreFile,
        chunk_shape: &[usize],
        src_start: &[usize],
        dst: &mut [T],
        dst_shape: &[usize],
        dst_start: &[usize],
        size: &[usize],
    ) -> H5Result<()> {
        let elem_size = std::mem::size_of::<T>();

        if self.filter.is_none() {
            // Unfiltered: fetch bytes and copy sub-region directly into dst.
            // No intermediate Vec<T> needed.
            let uncompressed_bytes = self.chunks_layout.chunk_bytes() as usize;
            if uncompressed_bytes != c.size as usize {
                return Err(H5Error::corrupt(format!(
                    "unfiltered chunk at {} records {} bytes but its layout implies {}",
                    c.address, c.size, uncompressed_bytes
                )));
            }
            let bytes = fetch_exact(file, c.address, uncompressed_bytes as u64).await?;

            let src_strides = row_major_strides(chunk_shape);
            let dst_strides = row_major_strides(dst_shape);
            copy_region_inner(
                &bytes,
                elem_size,
                &src_strides,
                src_start,
                bytemuck::cast_slice_mut(dst),
                &dst_strides,
                dst_start,
                size,
                0,
            );
        } else {
            // Filtered: run the pipeline backwards into a temp buffer, then copy
            // the sub-region out of it.
            let uncompressed_bytes = self.chunks_layout.chunk_bytes() as usize;
            let stored = fetch_exact(file, c.address, c.size).await?;
            let decoded = self.decode_filters(c, &stored, uncompressed_bytes)?;
            let data: &[u8] = decoded.as_deref().unwrap_or(&stored);

            if data.len() != uncompressed_bytes {
                return Err(H5Error::corrupt(format!(
                    "chunk at {} decoded to {} bytes, expected {}",
                    c.address,
                    data.len(),
                    uncompressed_bytes
                )));
            }

            // Copy sub-region from decompressed bytes directly into dst
            let src_strides = row_major_strides(chunk_shape);
            let dst_strides = row_major_strides(dst_shape);
            copy_region_inner(
                data,
                elem_size,
                &src_strides,
                src_start,
                bytemuck::cast_slice_mut(dst),
                &dst_strides,
                dst_start,
                size,
                0,
            );
        }

        Ok(())
    }

    /// Read the entire dataset, returning an NdArray with the dataset's shape.
    pub async fn read_full<T: H5Type>(&self, file: &ObjectStoreFile) -> H5Result<NdArray<T>> {
        let ndim = self.dataspace.dimensionality as usize;
        let sel: Vec<std::ops::Range<u64>> = self.dataspace.dimension[..ndim]
            .iter()
            .map(|&d| 0..d)
            .collect();
        self.read_range(&sel, file).await
    }

    /// Read a rectangular sub-region of the dataset, only fetching and
    /// decompressing the chunks that overlap with `selection`.
    ///
    /// `selection` must contain one `Range<u64>` per dataset dimension.
    /// Ranges are clamped to the dataset extent.
    pub async fn read_range<T: H5Type>(
        &self,
        selection: &[Range<u64>],
        file: &ObjectStoreFile,
    ) -> H5Result<NdArray<T>> {
        let ndim = self.dataspace.dimensionality as usize;
        if selection.len() != ndim {
            return Err(H5Error::InvalidSelection(format!(
                "selection has {} range(s) but the dataset has {ndim} dimension(s)",
                selection.len()
            )));
        }
        self.check_filters_supported()?;
        T::check_dtype(&self.datatype)?;

        let dataset_dims: &[u64] = &self.dataspace.dimension[..ndim];
        let chunk_dims: Vec<usize> = self
            .chunks_layout
            .chunk_dims
            .iter()
            .map(|&d| d as usize)
            .collect();

        // Clamp selection to dataset extent
        let sel: Vec<Range<u64>> = (0..ndim)
            .map(|d| {
                let start = selection[d].start.min(dataset_dims[d]);
                let end = selection[d].end.min(dataset_dims[d]);
                start..end
            })
            .collect();

        let output_shape: Vec<usize> = sel.iter().map(|r| (r.end - r.start) as usize).collect();
        let total_elements: usize = output_shape.iter().product();

        if total_elements == 0 {
            return Ok(NdArray {
                data: vec![],
                shape: output_shape,
            });
        }

        let mut output = vec![T::zeroed(); total_elements];
        let all_chunks = self.collect_chunks(file).await?;

        for chunk_ptr in &all_chunks {
            let chunk_offset: &[u64] = &chunk_ptr.offsets;

            // Compute intersection of chunk region with selection (in global coords),
            // clamping chunk extent to the dataset boundary for edge chunks.
            let mut intersects = true;
            let mut global_start = vec![0u64; ndim];
            let mut global_end = vec![0u64; ndim];
            for d in 0..ndim {
                let chunk_end = (chunk_offset[d] + chunk_dims[d] as u64).min(dataset_dims[d]);
                global_start[d] = sel[d].start.max(chunk_offset[d]);
                global_end[d] = sel[d].end.min(chunk_end);
                if global_start[d] >= global_end[d] {
                    intersects = false;
                    break;
                }
            }
            if !intersects {
                continue;
            }

            let chunk_local_start: Vec<usize> = (0..ndim)
                .map(|d| (global_start[d] - chunk_offset[d]) as usize)
                .collect();
            let output_start: Vec<usize> = (0..ndim)
                .map(|d| (global_start[d] - sel[d].start) as usize)
                .collect();
            let inter_size: Vec<usize> = (0..ndim)
                .map(|d| (global_end[d] - global_start[d]) as usize)
                .collect();

            self.read_chunk_into(
                chunk_ptr,
                file,
                &chunk_dims,
                &chunk_local_start,
                &mut output,
                &output_shape,
                &output_start,
                &inter_size,
            )
            .await?;
        }

        Ok(NdArray {
            data: output,
            shape: output_shape,
        })
    }

    /// Walk the chunk B-tree and collect a pointer to every chunk.
    pub(crate) async fn collect_chunks(
        &self,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<ChunkRecord>> {
        enumerate_chunks(file, &self.chunks_layout, self.dataset_dims()).await
    }
}
