#![allow(dead_code)]

use std::ops::Range;

use arrayvec::ArrayVec;

use crate::error::H5Result;
use crate::format::{
    btree::collect_btree_leaves_args,
    metadata::{ChunkBTreeV1, ChunkPointerV1},
    object::{
        DataLayoutChunked, DataLayoutMessage, DataspaceMessage, DatatypeMessage, FilterMessage,
        FilterType,
    },
};
use crate::h5type::H5Type;
use crate::object_store::{ObjectStoreFile, fetch_exact, read_metadata_args};
use std::io::Read;

/// N-dimensional array wrapper that pairs a flat data vector with its logical shape.
/// Data is in row-major (C) order.
#[derive(Debug, Clone)]
pub struct NdArray<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

type Dims = ArrayVec<usize, 4>;

/// Compute row-major strides for a given shape.
fn row_major_strides(shape: &[usize]) -> Dims {
    let ndim = shape.len();
    let mut strides = Dims::new();
    strides.try_extend_from_slice(&[1; 4][..ndim]).unwrap();
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Copy a rectangular sub-region from `src` into `dst`, both N-dimensional
/// arrays stored in row-major order.
fn copy_region<T: bytemuck::Pod>(
    src: &[T],
    src_shape: &[usize],
    src_start: &[usize],
    dst: &mut [T],
    dst_shape: &[usize],
    dst_start: &[usize],
    size: &[usize],
) {
    let elem_size = std::mem::size_of::<T>();
    let src_strides = row_major_strides(src_shape);
    let dst_strides = row_major_strides(dst_shape);
    copy_region_inner(
        bytemuck::cast_slice(src),
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

/// Copy a rectangular sub-region from raw bytes (with element layout of size
/// `elem_size`) into a destination byte buffer.  Both buffers are N-dimensional
/// arrays stored in row-major order.  Strides, offsets, and sizes are all
/// expressed in *elements*, not bytes.
#[inline(never)]
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
        let mut src_off: Dims = Dims::try_from(src_offset).unwrap();
        let mut dst_off: Dims = Dims::try_from(dst_offset).unwrap();
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

pub struct ChunkedDataset {
    pub(crate) name: String,
    pub(crate) dataspace: DataspaceMessage,
    pub(crate) datatype: DatatypeMessage,
    pub(crate) layout: DataLayoutMessage,
    pub(crate) chunks_layout: DataLayoutChunked,
    pub(crate) filter: Option<FilterMessage>,
}

impl ChunkedDataset {
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

    pub fn ndim(&self) -> usize {
        self.dataspace.dimensionality as usize
    }

    pub fn shape(&self) -> Vec<u64> {
        self.dataspace.dimension[..self.ndim()].to_vec()
    }

    pub fn chunk_shape(&self) -> Vec<u64> {
        self.chunks_layout.dimension_sizes[..self.ndim()]
            .iter()
            .map(|&d| d as u64)
            .collect()
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

    /// Decode a chunk and copy the specified sub-region directly into `dst`.
    /// For unfiltered data, copies directly from the fetched bytes into `dst`
    /// with no intermediate buffer.
    async fn read_chunk_into<T: H5Type>(
        &self,
        c: &ChunkPointerV1,
        file: &ObjectStoreFile,
        chunk_shape: &[usize],
        src_start: &[usize],
        dst: &mut [T],
        dst_shape: &[usize],
        dst_start: &[usize],
        size: &[usize],
    ) -> H5Result<()> {
        T::check_dtype(&self.datatype);
        let elem_size = std::mem::size_of::<T>();

        if self.filter.is_none() {
            // Unfiltered: fetch bytes and copy sub-region directly into dst.
            // No intermediate Vec<T> needed.
            let uncompressed_bytes: usize = self
                .chunks_layout
                .dimension_sizes
                .iter()
                .map(|&d| d as usize)
                .product();
            assert_eq!(uncompressed_bytes, c.key.chunk_size as usize);
            let bytes = fetch_exact(file, c.child_pointer, uncompressed_bytes as u64).await?;

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
            // Filtered: decompress into a temp buffer, then copy sub-region
            let uncompressed_bytes: usize = self
                .chunks_layout
                .dimension_sizes
                .iter()
                .map(|&d| d as usize)
                .product();

            let compressed = fetch_exact(file, c.child_pointer, c.key.chunk_size as u64).await?;

            // Inflate
            let mut data = Vec::with_capacity(uncompressed_bytes);
            let mut decoder = flate2::read::ZlibDecoder::new(&compressed[..]);
            decoder
                .read_to_end(&mut data)
                .map_err(|e| binrw::Error::Custom {
                    pos: c.child_pointer,
                    err: Box::new(e),
                })?;
            assert_eq!(data.len(), uncompressed_bytes);

            // Un-shuffle if needed
            if self.filter.as_ref().is_some_and(|f| {
                f.filters
                    .iter()
                    .any(|fd| matches!(fd.filter_type, FilterType::Shuffle))
            }) {
                let element_size = *self.chunks_layout.dimension_sizes.last().unwrap() as usize;
                let num_elements = uncompressed_bytes / element_size;
                let mut unshuffled = vec![0u8; uncompressed_bytes];
                for i in 0..num_elements {
                    for b in 0..element_size {
                        unshuffled[i * element_size + b] = data[b * num_elements + i];
                    }
                }
                data = unshuffled;
            }

            // Copy sub-region from decompressed bytes directly into dst
            let src_strides = row_major_strides(chunk_shape);
            let dst_strides = row_major_strides(dst_shape);
            copy_region_inner(
                &data,
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
        assert_eq!(
            selection.len(),
            ndim,
            "selection must have one range per dimension"
        );

        let dataset_dims: &[u64] = &self.dataspace.dimension[..ndim];
        let chunk_dims: Vec<usize> = self.chunks_layout.dimension_sizes[..ndim]
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
            let chunk_offset: &[u64] = &chunk_ptr.key.offsets[..ndim];

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

            let chunk_local_start: Dims = (0..ndim)
                .map(|d| (global_start[d] - chunk_offset[d]) as usize)
                .collect();
            let output_start: Dims = (0..ndim)
                .map(|d| (global_start[d] - sel[d].start) as usize)
                .collect();
            let inter_size: Dims = (0..ndim)
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

    pub async fn collect_chunks(&self, file: &ObjectStoreFile) -> H5Result<Vec<ChunkPointerV1>> {
        let btree: ChunkBTreeV1 = read_metadata_args(
            file,
            self.chunks_layout.address,
            (self.dataspace.dimensionality,),
        )
        .await?;

        collect_btree_leaves_args(file, btree).await
    }
}
