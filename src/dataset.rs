//! Reading dataset raw data: chunk traversal and filter decoding for chunked
//! datasets, and byte-range extraction for contiguous and compact ones.

use std::io::Read;
use std::ops::Range;

use crate::error::{H5Error, H5Result};
use crate::format::{
    chunk_index::{ChunkRecord, enumerate_chunks},
    object::{
        AttributeMessage, ChunkedLayout, DataLayoutMessage, DataspaceMessage, DatatypeMessage,
        FilterMessage, FilterType, LayoutInner, is_undefined_address,
    },
};
use crate::h5type::H5Type;
use crate::object_store::{ObjectStoreFile, fetch_exact};

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

/// An HDF5 dataset, opened and ready to read.
///
/// Holds only the dataset's metadata — its dataspace, datatype, storage layout
/// and filter pipeline. No bulk data is fetched until you call
/// [`read_full`](Self::read_full) or [`read_range`](Self::read_range), and a
/// range read fetches only the bytes it needs: the overlapping chunks of a
/// chunked dataset, or the enclosing byte span of a contiguous one.
pub struct Dataset {
    pub(crate) name: String,
    pub(crate) dataspace: DataspaceMessage,
    pub(crate) datatype: DatatypeMessage,
    pub(crate) layout: DataLayoutMessage,
    pub(crate) filter: Option<FilterMessage>,
    /// Read by the differential tests, which compare every attribute against
    /// the HDF5 C library. There is no public accessor yet.
    #[allow(dead_code)]
    pub(crate) attributes: Vec<AttributeMessage>,
}

impl Dataset {
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

    /// Number of dimensions. Zero for a scalar or null dataspace.
    pub fn ndim(&self) -> usize {
        self.dataspace.dimensionality as usize
    }

    /// Dataset extent along each dimension.
    fn dataset_dims(&self) -> &[u64] {
        &self.dataspace.dimension[..self.ndim()]
    }

    /// Length of each dataset dimension, outermost first.
    pub fn shape(&self) -> Vec<u64> {
        self.dataset_dims().to_vec()
    }

    /// How the raw data is stored: `"chunked"`, `"contiguous"`, `"compact"`,
    /// or `"virtual"`.
    pub fn layout_name(&self) -> &'static str {
        match &self.layout.inner {
            LayoutInner::Chunked(_) => "chunked",
            LayoutInner::Contiguous { .. } => "contiguous",
            LayoutInner::Compact { .. } => "compact",
            LayoutInner::Unsupported(_) => "virtual",
        }
    }

    /// Length of each chunk dimension, outermost first, or `None` if the
    /// dataset is not chunked.
    pub fn chunk_shape(&self) -> Option<Vec<u64>> {
        self.layout.chunked().map(|c| c.chunk_dims.clone())
    }

    /// Name of the structure used to locate this dataset's chunks, for
    /// diagnostics: `"v1 btree"`, `"single chunk"`, `"implicit"`,
    /// `"fixed array"`, `"extensible array"` or `"v2 btree"`. `None` if the
    /// dataset is not chunked.
    pub fn chunk_index_name(&self) -> Option<&'static str> {
        use crate::format::object::ChunkIndex::*;
        self.layout.chunked().map(|c| match c.index {
            BTreeV1 { .. } => "v1 btree",
            SingleChunk { .. } => "single chunk",
            Implicit { .. } => "implicit",
            FixedArray { .. } => "fixed array",
            ExtensibleArray { .. } => "extensible array",
            BTreeV2 { .. } => "v2 btree",
        })
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

    /// The chunked layout, or an error naming the layout this dataset actually
    /// uses.
    #[cfg(test)]
    fn chunked_layout(&self) -> H5Result<&ChunkedLayout> {
        self.layout.chunked().ok_or_else(|| {
            H5Error::corrupt(format!("dataset is {}, not chunked", self.layout_name()))
        })
    }

    /// Reject filter pipelines h5rs cannot decode, before any data is fetched.
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

        // HDF5 only applies the filter pipeline to chunked datasets. A pipeline
        // on any other layout would mean the bytes on disk are not what the
        // dataspace and datatype describe.
        let has_filters = fm
            .filters
            .iter()
            .any(|fd| fd.filter_type != FilterType::None);
        if has_filters && self.layout.chunked().is_none() {
            return Err(H5Error::unsupported(format!(
                "filter pipeline on a {} dataset",
                self.layout_name()
            )));
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
        layout: &ChunkedLayout,
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
                FilterType::Shuffle => Some(unshuffle(input, layout.element_size as usize)?),
                // check_filters_supported rejects everything else up front.
                ref other => {
                    return Err(H5Error::unsupported(format!("{other:?} filter")));
                }
            };
        }
        Ok(decoded)
    }

    /// Decode a chunk and copy the specified sub-region directly into `dst`.
    /// For unfiltered data, copies directly from the fetched bytes into `dst`
    /// with no intermediate buffer.
    #[allow(clippy::too_many_arguments)]
    async fn read_chunk_into<T: H5Type>(
        &self,
        layout: &ChunkedLayout,
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
        let uncompressed_bytes = layout.chunk_bytes() as usize;

        if self.filter.is_none() {
            // Unfiltered: fetch bytes and copy sub-region directly into dst.
            // No intermediate Vec<T> needed.
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
            let stored = fetch_exact(file, c.address, c.size).await?;
            let decoded = self.decode_filters(layout, c, &stored, uncompressed_bytes)?;
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
        let sel: Vec<Range<u64>> = self.dataset_dims().iter().map(|&d| 0..d).collect();
        self.read_range(&sel, file).await
    }

    /// Read a rectangular sub-region of the dataset.
    ///
    /// `selection` must contain one `Range<u64>` per dataset dimension; a
    /// scalar dataset takes an empty selection. Ranges are clamped to the
    /// dataset extent.
    ///
    /// Only the bytes needed are fetched. For a chunked dataset that means the
    /// chunks the selection overlaps; for a contiguous one, the single byte
    /// span running from the first selected element to the last — which is
    /// exact for a one-dimensional range, and for higher-rank selections spans
    /// the rows the selection touches.
    pub async fn read_range<T: H5Type>(
        &self,
        selection: &[Range<u64>],
        file: &ObjectStoreFile,
    ) -> H5Result<NdArray<T>> {
        let ndim = self.ndim();
        if selection.len() != ndim {
            return Err(H5Error::InvalidSelection(format!(
                "selection has {} range(s) but the dataset has {ndim} dimension(s)",
                selection.len()
            )));
        }
        self.check_filters_supported()?;
        T::check_dtype(&self.datatype)?;

        let dataset_dims = self.dataset_dims();

        // Clamp selection to dataset extent
        let sel: Vec<Range<u64>> = (0..ndim)
            .map(|d| {
                let start = selection[d].start.min(dataset_dims[d]);
                let end = selection[d].end.min(dataset_dims[d]);
                start..end
            })
            .collect();

        let output_shape: Vec<usize> = sel.iter().map(|r| (r.end - r.start) as usize).collect();
        // A scalar dataspace has no dimensions but one element; a null
        // dataspace has none.
        let total_elements: usize = if self.dataspace.is_null() {
            0
        } else {
            output_shape.iter().product()
        };

        if total_elements == 0 {
            return Ok(NdArray {
                data: vec![],
                shape: output_shape,
            });
        }

        let mut output = vec![T::zeroed(); total_elements];
        match &self.layout.inner {
            LayoutInner::Chunked(layout) => {
                self.read_chunked_into(layout, &sel, &output_shape, &mut output, file)
                    .await?;
            }
            LayoutInner::Contiguous { address, size } => {
                self.read_contiguous_into(*address, *size, &sel, &output_shape, &mut output, file)
                    .await?;
            }
            LayoutInner::Compact { data } => {
                self.read_compact_into(data, &sel, &output_shape, &mut output)?;
            }
            LayoutInner::Unsupported(class) => {
                return Err(H5Error::unsupported(format!(
                    "data layout class {class} (virtual datasets are not implemented)"
                )));
            }
        }

        Ok(NdArray {
            data: output,
            shape: output_shape,
        })
    }

    /// Fill `output` from the chunks that overlap `sel`.
    async fn read_chunked_into<T: H5Type>(
        &self,
        layout: &ChunkedLayout,
        sel: &[Range<u64>],
        output_shape: &[usize],
        output: &mut [T],
        file: &ObjectStoreFile,
    ) -> H5Result<()> {
        let ndim = self.ndim();
        if ndim == 0 {
            return Err(H5Error::corrupt("a scalar dataset cannot be chunked"));
        }
        if layout.ndim() != ndim {
            return Err(H5Error::corrupt(format!(
                "chunk layout has {} dimensions but the dataspace has {ndim}",
                layout.ndim()
            )));
        }

        let dataset_dims = self.dataset_dims();
        let chunk_dims: Vec<usize> = layout.chunk_dims.iter().map(|&d| d as usize).collect();
        let all_chunks = enumerate_chunks(file, layout, dataset_dims).await?;

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
                layout,
                chunk_ptr,
                file,
                &chunk_dims,
                &chunk_local_start,
                output,
                output_shape,
                &output_start,
                &inter_size,
            )
            .await?;
        }
        Ok(())
    }

    /// Fill `output` from a contiguous run of raw data.
    ///
    /// The whole dataset is one row-major array on disk, so the selection maps
    /// to a single byte span: from the first selected element to the last. That
    /// span is fetched in one request and the sub-region copied out of it.
    async fn read_contiguous_into<T: H5Type>(
        &self,
        address: u64,
        allocated_size: u64,
        sel: &[Range<u64>],
        output_shape: &[usize],
        output: &mut [T],
        file: &ObjectStoreFile,
    ) -> H5Result<()> {
        // No storage allocated yet: the dataset reads as its fill value.
        if is_undefined_address(address) {
            return Ok(());
        }

        let elem_size = std::mem::size_of::<T>();
        let dst: &mut [u8] = bytemuck::cast_slice_mut(output);

        if self.ndim() == 0 {
            let bytes = fetch_exact(file, address, elem_size as u64).await?;
            dst[..elem_size].copy_from_slice(&bytes[..elem_size]);
            return Ok(());
        }

        let strides = row_major_strides(&dims_as_usize(self.dataset_dims()));

        // Offset of the first selected element, and how far past it the last
        // selected element lies.
        let first: usize = (0..sel.len())
            .map(|d| sel[d].start as usize * strides[d])
            .sum();
        let span: usize = (0..output_shape.len())
            .map(|d| (output_shape[d] - 1) * strides[d])
            .sum::<usize>()
            + 1;

        let end = (first + span) as u64 * elem_size as u64;
        if allocated_size > 0 && end > allocated_size {
            return Err(H5Error::corrupt(format!(
                "contiguous dataset needs {end} bytes but only {allocated_size} are allocated"
            )));
        }

        let bytes = fetch_exact(
            file,
            address + (first * elem_size) as u64,
            (span * elem_size) as u64,
        )
        .await?;

        // The fetched buffer begins at the first selected element, so within it
        // the selection starts at the origin while keeping the dataset's strides.
        let origin = vec![0usize; sel.len()];
        copy_region_inner(
            &bytes,
            elem_size,
            &strides,
            &origin,
            dst,
            &row_major_strides(output_shape),
            &origin,
            output_shape,
            0,
        );
        Ok(())
    }

    /// Fill `output` from raw data stored inline in the object header.
    fn read_compact_into<T: H5Type>(
        &self,
        data: &[u8],
        sel: &[Range<u64>],
        output_shape: &[usize],
        output: &mut [T],
    ) -> H5Result<()> {
        let elem_size = std::mem::size_of::<T>();
        let needed = self.dataspace.num_elements() * elem_size;
        if data.len() < needed {
            return Err(H5Error::corrupt(format!(
                "compact dataset holds {} bytes but its dataspace needs {needed}",
                data.len()
            )));
        }

        let dst: &mut [u8] = bytemuck::cast_slice_mut(output);
        if self.ndim() == 0 {
            dst[..elem_size].copy_from_slice(&data[..elem_size]);
            return Ok(());
        }

        let src_offset: Vec<usize> = sel.iter().map(|r| r.start as usize).collect();
        copy_region_inner(
            data,
            elem_size,
            &row_major_strides(&dims_as_usize(self.dataset_dims())),
            &src_offset,
            dst,
            &row_major_strides(output_shape),
            &vec![0usize; sel.len()],
            output_shape,
            0,
        );
        Ok(())
    }

    /// Walk the chunk index and collect a record for every allocated chunk.
    #[cfg(test)]
    pub(crate) async fn collect_chunks(
        &self,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<ChunkRecord>> {
        let layout = self.chunked_layout()?;
        enumerate_chunks(file, layout, self.dataset_dims()).await
    }
}

fn dims_as_usize(dims: &[u64]) -> Vec<usize> {
    dims.iter().map(|&d| d as usize).collect()
}

/// Reverse the shuffle filter, which stores all the first bytes of each
/// element, then all the second bytes, and so on.
fn unshuffle(data: &[u8], element_size: usize) -> H5Result<Vec<u8>> {
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
