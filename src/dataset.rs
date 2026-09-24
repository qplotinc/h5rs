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
use crate::object_store::{ObjectStoreFile, fetch_data};

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

/// Requests h5rs will keep in flight regardless of the memory ceiling, so that
/// there is always something downloading while something else decodes.
const MIN_IN_FLIGHT: usize = 2;

/// Where one chunk's contribution to the output lives, in elements.
struct ChunkCopy {
    /// Offset of the intersection within the chunk.
    src_start: Vec<usize>,
    /// Offset of the intersection within the output array.
    dst_start: Vec<usize>,
    /// Extent of the intersection.
    size: Vec<usize>,
}

/// An HDF5 dataset, opened and ready to read.
///
/// Holds only the dataset's metadata — its dataspace, datatype, storage layout
/// and filter pipeline. No bulk data is fetched until you call
/// [`read_full`](Self::read_full) or [`read_range`](Self::read_range), and a
/// range read fetches only the bytes it needs: the overlapping chunks of a
/// chunked dataset, or the enclosing byte span of a contiguous one.
pub struct Dataset {
    /// The chunk index, once walked. Reading several ranges out of one dataset
    /// is a common pattern, and re-walking the index for each of them would
    /// cost round trips for a list that cannot change.
    chunks: std::sync::Mutex<Option<std::sync::Arc<Vec<ChunkRecord>>>>,
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
    /// Assemble a dataset from the messages of its object header.
    pub(crate) fn new(
        name: String,
        dataspace: DataspaceMessage,
        datatype: DatatypeMessage,
        layout: DataLayoutMessage,
        filter: Option<FilterMessage>,
        attributes: Vec<AttributeMessage>,
    ) -> Dataset {
        Dataset {
            chunks: Default::default(),
            name,
            dataspace,
            datatype,
            layout,
            filter,
            attributes,
        }
    }

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
    /// An enumeration reads as its unsigned base integer.
    pub fn dtype_info(&self) -> (bool, bool, usize) {
        use crate::format::object::TypeDescriptor;
        match &self.datatype.type_desc {
            TypeDescriptor::FloatingPoint(fp) => (true, true, fp.size() as usize),
            TypeDescriptor::FixedPoint(fp) => (false, fp.signed() != 0, fp.size() as usize),
            TypeDescriptor::Enumeration(e) => (false, false, e.size() as usize),
            _ => (false, false, 0),
        }
    }

    /// The datatype's on-disk element size.
    pub fn element_size(&self) -> H5Result<usize> {
        self.datatype.element_size()
    }

    /// Whether the datatype is an enumeration (h5py's
    /// booleans), so a reader can tell a flag column from a `uint8`.
    pub fn is_enumeration(&self) -> bool {
        use crate::format::object::TypeDescriptor;
        matches!(self.datatype.type_desc, TypeDescriptor::Enumeration(_))
    }

    /// This object's attributes.
    pub fn attributes(&self) -> &[crate::format::object::AttributeMessage] {
        &self.attributes
    }

    /// A rectangular sub-region as raw element bytes —
    /// `element_size()` bytes per element, row-major — for datatypes with no
    /// `H5Type` (fixed-length strings, variable-length references, enums).
    /// Returns the bytes and the region's shape in elements.
    ///
    /// Implemented as a *byte view*: the same storage re-described with a
    /// one-byte datatype and its innermost dimension (and chunk dimension,
    /// and chunk offsets) multiplied by the element size, then read through
    /// the ordinary typed path as `u8`. A scalar dataset views as one row of
    /// `element_size` bytes. The shuffle filter needs the true element size,
    /// so a shuffled dataset is refused here rather than un-shuffled wrongly.
    pub async fn read_range_bytes(
        &self,
        selection: &[Range<u64>],
        file: &ObjectStoreFile,
    ) -> H5Result<(Vec<u8>, Vec<usize>)> {
        let elem = self.element_size()?;
        let ndim = self.ndim();
        if selection.len() != ndim {
            return Err(H5Error::InvalidSelection(format!(
                "selection has {} range(s) but the dataset has {ndim} dimension(s)",
                selection.len()
            )));
        }
        if let Some(fm) = &self.filter
            && fm
                .filters
                .iter()
                .any(|fd| fd.filter_type == FilterType::Shuffle)
        {
            return Err(H5Error::unsupported(
                "raw reads of a shuffled dataset (the byte view cannot un-shuffle)",
            ));
        }
        let elem64 = elem as u64;

        // The view's dataspace: innermost dimension in bytes; a scalar
        // becomes one dimension of `elem` bytes.
        let mut dataspace = self.dataspace.clone();
        let mut sel: Vec<Range<u64>> = selection.to_vec();
        if ndim == 0 {
            dataspace.dimensionality = 1;
            dataspace.dataspace_type = 1;
            dataspace.dimension = vec![elem64];
            dataspace.dimension_max = vec![elem64];
            sel = std::iter::once(0..elem64).collect();
        } else {
            let last = ndim - 1;
            dataspace.dimension[last] *= elem64;
            if dataspace.dimension_max.len() > last && dataspace.dimension_max[last] != u64::MAX {
                dataspace.dimension_max[last] *= elem64;
            }
            sel[last] = sel[last].start * elem64..sel[last].end * elem64;
        }

        // The view's layout: chunk dims in bytes, element size 1. Chunk
        // records are re-derived with byte offsets so the read never walks
        // the index in the wrong units.
        let mut layout = self.layout.clone();
        let mut chunks_in_bytes = None;
        if let LayoutInner::Chunked(c) = &mut layout.inner {
            let last = c.chunk_dims.len().saturating_sub(1);
            if ndim > 0 {
                c.chunk_dims[last] *= elem64;
            }
            c.element_size = 1;
            let records = self.collect_chunks(file).await?;
            let scaled: Vec<ChunkRecord> = records
                .iter()
                .map(|r| {
                    let mut r = r.clone();
                    if let Some(o) = r.offsets.last_mut() {
                        *o *= elem64;
                    }
                    r
                })
                .collect();
            chunks_in_bytes = Some(std::sync::Arc::new(scaled));
        }

        let view = Dataset {
            chunks: std::sync::Mutex::new(chunks_in_bytes),
            name: self.name.clone(),
            dataspace,
            datatype: DatatypeMessage::unsigned_byte(),
            layout,
            filter: self.filter.clone(),
            attributes: Vec::new(),
        };
        let arr = view.read_range::<u8>(&sel, file).await?;
        let mut shape = arr.shape;
        if ndim == 0 {
            shape = vec![];
        } else if let Some(last) = shape.last_mut() {
            *last /= elem;
        }
        Ok((arr.data, shape))
    }

    /// The chunked layout, or an error naming the layout this dataset actually
    /// uses.
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

    /// Whether any filter actually applies to this chunk, and so whether its
    /// bytes have to pass through the compute pool at all.
    fn needs_decode(&self, c: &ChunkRecord) -> bool {
        let Some(pipeline) = &self.filter else {
            return false;
        };
        pipeline
            .filters
            .iter()
            .enumerate()
            .any(|(i, fd)| fd.filter_type != FilterType::None && c.filter_mask & (1u32 << i) == 0)
    }

    /// Copy an already-decoded chunk's sub-region into the output.
    #[allow(clippy::too_many_arguments)]
    fn place_chunk<T: H5Type>(
        &self,
        c: &ChunkRecord,
        data: &[u8],
        uncompressed_bytes: usize,
        chunk_shape: &[usize],
        src_start: &[usize],
        dst: &mut [T],
        dst_shape: &[usize],
        dst_start: &[usize],
        size: &[usize],
    ) -> H5Result<()> {
        if data.len() != uncompressed_bytes {
            return Err(H5Error::corrupt(format!(
                "chunk at {} decoded to {} bytes, expected {}",
                c.address,
                data.len(),
                uncompressed_bytes
            )));
        }
        copy_region_inner(
            data,
            std::mem::size_of::<T>(),
            &row_major_strides(chunk_shape),
            src_start,
            bytemuck::cast_slice_mut(dst),
            &row_major_strides(dst_shape),
            dst_start,
            size,
            0,
        );
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
        let all_chunks = self.collect_chunks(file).await?;

        // Work out which chunks contribute, and where, before fetching any of
        // them: knowing the whole list up front is what lets the reads be
        // batched into a few requests instead of one per chunk.
        let mut wanted: Vec<(&ChunkRecord, ChunkCopy)> = vec![];
        for chunk in all_chunks.iter() {
            let chunk_offset: &[u64] = &chunk.offsets;

            // Intersect the chunk with the selection in global coordinates,
            // clamping the chunk extent to the dataset boundary for edge chunks.
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

            wanted.push((
                chunk,
                ChunkCopy {
                    src_start: (0..ndim)
                        .map(|d| (global_start[d] - chunk_offset[d]) as usize)
                        .collect(),
                    dst_start: (0..ndim)
                        .map(|d| (global_start[d] - sel[d].start) as usize)
                        .collect(),
                    size: (0..ndim)
                        .map(|d| (global_end[d] - global_start[d]) as usize)
                        .collect(),
                },
            ));
        }

        self.stream_chunks_into(layout, &wanted, &chunk_dims, output, output_shape, file)
            .await
    }

    /// Fetch and decode the wanted chunks as a pipeline.
    ///
    /// Requests are kept in flight up to `io_concurrency`; each one that lands
    /// is handed straight to the compute pool, and each decoded chunk is copied
    /// into the output as it comes back. So downloading, decompressing and
    /// copying all proceed at once, and whichever is slowest sets the pace.
    async fn stream_chunks_into<T: H5Type>(
        &self,
        layout: &ChunkedLayout,
        wanted: &[(&ChunkRecord, ChunkCopy)],
        chunk_dims: &[usize],
        output: &mut [T],
        output_shape: &[usize],
        file: &ObjectStoreFile,
    ) -> H5Result<()> {
        use futures_util::stream::FuturesUnordered;

        if wanted.is_empty() {
            return Ok(());
        }

        let options = file.options();
        let groups = group_requests(wanted, options.coalesce_gap, options.max_coalesced_bytes);
        let pool = file.compute().clone();
        let uncompressed = layout.chunk_bytes() as usize;

        let mut fetches = FuturesUnordered::new();
        let mut decodes = FuturesUnordered::new();
        let mut next_group = 0usize;
        // Bytes h5rs is holding for this read: fetched-but-not-yet-copied, plus
        // whatever is still in flight.
        let mut held_bytes = 0u64;
        // A pool with more workers than the concurrency setting would otherwise
        // sit idle waiting for data.
        let concurrency = options.io_concurrency.max(pool.parallelism()).max(1);

        loop {
            // Keep requests in flight up to the concurrency limit, but never let
            // the bytes h5rs is holding run past the ceiling. That accounting
            // spans both stages: a fast link feeding a slow decoder would
            // otherwise buffer the whole dataset, since bytes that have arrived
            // but not yet been decoded are just as resident as ones still in
            // flight.
            //
            // `MIN_IN_FLIGHT` requests are always allowed through, whatever the
            // ceiling says. A dataset whose chunks are each larger than the
            // ceiling would otherwise be read one chunk at a time, with nothing
            // downloading while a chunk decodes and nothing decoding while one
            // downloads — no pipeline at all.
            while fetches.len() < concurrency && next_group < groups.len() {
                let group = groups[next_group].clone();
                let size = group.range.end - group.range.start;
                if fetches.len() >= MIN_IN_FLIGHT && held_bytes + size > options.max_inflight_bytes
                {
                    break;
                }
                next_group += 1;
                held_bytes += size;
                let file = file.clone();
                fetches.push(async move {
                    let bytes = file.get_range(group.range.clone()).await?;
                    H5Result::Ok((group, bytes))
                });
            }

            if fetches.is_empty() && decodes.is_empty() {
                if next_group >= groups.len() {
                    break;
                }
                // Backpressure emptied both queues; take the next request.
                continue;
            }

            match next_event(&mut fetches, &mut decodes).await {
                // A decoded chunk: copy it into the output and free the buffer.
                Event::Decoded(result) => {
                    let (index, decoded): (usize, Vec<u8>) = result?;
                    let (chunk, copy) = &wanted[index];
                    held_bytes = held_bytes.saturating_sub(chunk.size);
                    self.place_chunk(
                        chunk,
                        &decoded,
                        uncompressed,
                        chunk_dims,
                        &copy.src_start,
                        output,
                        output_shape,
                        &copy.dst_start,
                        &copy.size,
                    )?;
                }
                // A fetched request: split it up and queue each chunk's decode.
                Event::Fetched(result) => {
                    let (group, bytes) = result?;
                    // The group's bytes stay charged until each chunk cut from
                    // them has been decoded and copied out.
                    held_bytes = held_bytes
                        .saturating_sub(group.range.end - group.range.start)
                        .saturating_add(group.members.iter().map(|&i| wanted[i].0.size).sum());
                    for &index in &group.members {
                        let (chunk, copy) = &wanted[index];
                        let start = (chunk.address - group.range.start) as usize;
                        let end = (start + chunk.size as usize).min(bytes.len());
                        let stored = bytes.slice(start.min(bytes.len())..end);

                        // Nothing to undo: copy straight out of the fetched
                        // bytes rather than paying for a job and a buffer.
                        if !self.needs_decode(chunk) {
                            held_bytes = held_bytes.saturating_sub(chunk.size);
                            self.place_chunk(
                                chunk,
                                &stored,
                                uncompressed,
                                chunk_dims,
                                &copy.src_start,
                                output,
                                output_shape,
                                &copy.dst_start,
                                &copy.size,
                            )?;
                            continue;
                        }

                        let filter = self.filter.clone();
                        let element_size = layout.element_size as usize;
                        let chunk = (*chunk).clone();
                        let job: crate::compute::ComputeJob = Box::new(move || {
                            decode_chunk(&filter, element_size, &chunk, &stored, uncompressed)
                        });
                        let run = pool.run(job);
                        decodes.push(async move { run.await.map(|decoded| (index, decoded)) });
                    }
                }
            }
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
            let bytes = fetch_data(file, address, elem_size as u64).await?;
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

        let bytes = fetch_data(
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

    /// Walk the chunk index and collect a record for every allocated chunk,
    /// reusing the result of any earlier walk.
    pub(crate) async fn collect_chunks(
        &self,
        file: &ObjectStoreFile,
    ) -> H5Result<std::sync::Arc<Vec<ChunkRecord>>> {
        if let Some(chunks) = self.chunks.lock().expect("chunk cache poisoned").as_ref() {
            return Ok(chunks.clone());
        }

        let layout = self.chunked_layout()?;
        let chunks =
            std::sync::Arc::new(enumerate_chunks(file, layout, self.dataset_dims()).await?);
        // A concurrent walk may have finished first; either result is the same.
        *self.chunks.lock().expect("chunk cache poisoned") = Some(chunks.clone());
        Ok(chunks)
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

/// A single bulk request, and which of the wanted chunks it carries.
#[derive(Clone, Debug)]
struct RequestGroup {
    range: Range<u64>,
    members: Vec<usize>,
}

/// Group the wanted chunks into requests.
///
/// Chunks that sit next to each other on disk are fetched together, since the
/// gap between them costs less than another round trip — but only while the
/// merged span stays under `max_coalesced`, because nothing in a request can be
/// decoded until all of it has landed.
///
/// A chunk always lands in exactly one request, at its full stored size, even
/// when that is larger than `max_coalesced`: the ceiling gates merging a
/// neighbour in, never splitting a chunk up.
fn group_requests(
    wanted: &[(&ChunkRecord, ChunkCopy)],
    gap: u64,
    max_coalesced: u64,
) -> Vec<RequestGroup> {
    let mut order: Vec<usize> = (0..wanted.len()).collect();
    order.sort_unstable_by_key(|&i| wanted[i].0.address);

    let max_coalesced = max_coalesced.max(1);
    let mut groups: Vec<RequestGroup> = vec![];
    for index in order {
        let chunk = wanted[index].0;
        let range = chunk.address..chunk.address + chunk.size;
        match groups.last_mut() {
            Some(last)
                if range.start <= last.range.end.saturating_add(gap)
                    && range.end.saturating_sub(last.range.start) <= max_coalesced =>
            {
                last.range.end = last.range.end.max(range.end);
                last.members.push(index);
            }
            _ => groups.push(RequestGroup {
                range,
                members: vec![index],
            }),
        }
    }
    groups
}

/// Undo the filter pipeline for one chunk.
///
/// Filters are applied in order on write, so they are undone in reverse. A
/// chunk whose data a filter would have grown is stored with that filter
/// skipped and its bit set in the chunk's filter mask, so the mask has to be
/// honoured or such a chunk decodes to garbage.
///
/// This runs on the compute pool, so it takes only owned values.
fn decode_chunk(
    filter: &Option<FilterMessage>,
    element_size: usize,
    c: &ChunkRecord,
    stored: &[u8],
    uncompressed_bytes: usize,
) -> H5Result<Vec<u8>> {
    let Some(pipeline) = filter else {
        return Ok(stored.to_vec());
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
                out.reserve(uncompressed_bytes);
                decoder
                    .read_to_end(&mut out)
                    .map_err(|e| binrw::Error::Custom {
                        pos: c.address,
                        err: Box::new(e),
                    })?;
                Some(out)
            }
            FilterType::Shuffle => Some(unshuffle(input, element_size)?),
            // check_filters_supported rejects everything else up front.
            ref other => {
                return Err(H5Error::unsupported(format!("{other:?} filter")));
            }
        };
    }
    Ok(decoded.unwrap_or_else(|| stored.to_vec()))
}

/// Which pipeline stage finished first.
enum Event<F, D> {
    Fetched(F),
    Decoded(D),
}

/// Wait for the next fetch or decode to complete.
///
/// Decodes are checked first so that finished buffers are copied out and freed
/// before more are pulled in, which is what bounds the memory a read holds.
async fn next_event<F, D>(
    fetches: &mut futures_util::stream::FuturesUnordered<F>,
    decodes: &mut futures_util::stream::FuturesUnordered<D>,
) -> Event<F::Output, D::Output>
where
    F: std::future::Future,
    D: std::future::Future,
{
    use futures_util::StreamExt;
    use std::task::Poll;

    futures_util::future::poll_fn(|cx| {
        if let Poll::Ready(Some(decoded)) = decodes.poll_next_unpin(cx) {
            return Poll::Ready(Event::Decoded(decoded));
        }
        if let Poll::Ready(Some(fetched)) = fetches.poll_next_unpin(cx) {
            return Poll::Ready(Event::Fetched(fetched));
        }
        Poll::Pending
    })
    .await
}
