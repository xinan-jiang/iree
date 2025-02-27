// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/buffer_view.h"

#include <inttypes.h>

#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/detail.h"

struct iree_hal_buffer_view_s {
  iree_atomic_ref_count_t ref_count;
  iree_hal_buffer_t* buffer;
  iree_hal_element_type_t element_type;
  iree_device_size_t byte_length;
  iree_host_size_t shape_rank;
  iree_hal_dim_t shape[];
};

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer_view);

  *out_buffer_view = NULL;
  if (IREE_UNLIKELY(shape_rank > 0 && !shape)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no shape dimensions specified");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(iree_hal_buffer_allocator(buffer));

  // Allocate and initialize the iree_hal_buffer_view_t struct.
  // Note that we have the dynamically-sized shape dimensions on the end.
  iree_hal_buffer_view_t* buffer_view = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      sizeof(*buffer_view) + sizeof(iree_hal_dim_t) * shape_rank,
      (void**)&buffer_view);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&buffer_view->ref_count);
    buffer_view->buffer = buffer;
    iree_hal_buffer_retain(buffer_view->buffer);
    buffer_view->element_type = element_type;
    buffer_view->byte_length =
        iree_hal_element_byte_count(buffer_view->element_type);
    buffer_view->shape_rank = shape_rank;
    for (iree_host_size_t i = 0; i < shape_rank; ++i) {
      buffer_view->shape[i] = shape[i];
      buffer_view->byte_length *= shape[i];
    }
    *out_buffer_view = buffer_view;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_view_retain(iree_hal_buffer_view_t* buffer_view) {
  if (IREE_LIKELY(buffer_view)) {
    iree_atomic_ref_count_inc(&buffer_view->ref_count);
  }
}

IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_view_release(iree_hal_buffer_view_t* buffer_view) {
  if (IREE_LIKELY(buffer_view) &&
      iree_atomic_ref_count_dec(&buffer_view->ref_count) == 1) {
    iree_hal_buffer_view_destroy(buffer_view);
  }
}

IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_view_destroy(iree_hal_buffer_view_t* buffer_view) {
  iree_allocator_t host_allocator = iree_hal_allocator_host_allocator(
      iree_hal_buffer_allocator(buffer_view->buffer));
  iree_hal_buffer_release(buffer_view->buffer);
  iree_allocator_free(host_allocator, buffer_view);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_subview(
    const iree_hal_buffer_view_t* buffer_view,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(out_buffer_view);

  // NOTE: we rely on the compute range call to do parameter validation.
  iree_device_size_t start_offset = 0;
  iree_device_size_t subview_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_range(
      buffer_view, start_indices, indices_count, lengths, lengths_count,
      &start_offset, &subview_length));

  iree_hal_buffer_t* subview_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(
      buffer_view->buffer, start_offset, subview_length, &subview_buffer));

  iree_status_t status =
      iree_hal_buffer_view_create(subview_buffer, lengths, lengths_count,
                                  buffer_view->element_type, out_buffer_view);
  iree_hal_buffer_release(subview_buffer);
  return status;
}

IREE_API_EXPORT iree_hal_buffer_t* iree_hal_buffer_view_buffer(
    const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->buffer;
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_hal_buffer_view_shape_rank(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->shape_rank;
}

IREE_API_EXPORT const iree_hal_dim_t* IREE_API_CALL
iree_hal_buffer_view_shape_dims(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->shape;
}

IREE_API_EXPORT iree_hal_dim_t IREE_API_CALL iree_hal_buffer_view_shape_dim(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  if (IREE_UNLIKELY(index > buffer_view->shape_rank)) {
    return 0;
  }
  return buffer_view->shape[index];
}

IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_count(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  iree_host_size_t element_count = 1;
  for (iree_host_size_t i = 0; i < buffer_view->shape_rank; ++i) {
    element_count *= buffer_view->shape[i];
  }
  return element_count;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_shape(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t rank_capacity,
    iree_hal_dim_t* out_shape, iree_host_size_t* out_shape_rank) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  IREE_ASSERT_ARGUMENT(out_shape);
  if (out_shape_rank) {
    *out_shape_rank = 0;
  }

  if (out_shape_rank) {
    *out_shape_rank = buffer_view->shape_rank;
  }
  if (rank_capacity < buffer_view->shape_rank) {
    // Not an error; just a size query.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  for (iree_host_size_t i = 0; i < buffer_view->shape_rank; ++i) {
    out_shape[i] = buffer_view->shape[i];
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_reshape(
    iree_hal_buffer_view_t* buffer_view, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  IREE_ASSERT_ARGUMENT(shape);

  if (shape_rank != buffer_view->shape_rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer view reshapes must have the same rank; "
                            "target=%zu, existing=%zu",
                            shape_rank, buffer_view->shape_rank);
  }

  iree_device_size_t new_element_count = 1;
  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    new_element_count *= shape[i];
  }
  iree_device_size_t old_element_count =
      iree_hal_buffer_view_element_count(buffer_view);
  if (new_element_count != old_element_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer view reshapes must have the same element "
                            "count; target=%" PRIu64 ", existing=%" PRIu64,
                            new_element_count, old_element_count);
  }

  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    buffer_view->shape[i] = shape[i];
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_element_type_t IREE_API_CALL
iree_hal_buffer_view_element_type(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->element_type;
}

IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_size(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return iree_hal_element_byte_count(buffer_view->element_type);
}

IREE_API_EXPORT iree_device_size_t IREE_API_CALL
iree_hal_buffer_view_byte_length(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->byte_length;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_offset(
    const iree_hal_buffer_view_t* buffer_view, const iree_hal_dim_t* indices,
    iree_host_size_t indices_count, iree_device_size_t* out_offset) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return iree_hal_buffer_compute_view_offset(
      buffer_view->shape, buffer_view->shape_rank, buffer_view->element_type,
      indices, indices_count, out_offset);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_range(
    const iree_hal_buffer_view_t* buffer_view,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_device_size_t* out_start_offset, iree_device_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return iree_hal_buffer_compute_view_range(
      buffer_view->shape, buffer_view->shape_rank, buffer_view->element_type,
      start_indices, indices_count, lengths, lengths_count, out_start_offset,
      out_length);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_compute_view_size(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type,
    iree_device_size_t* out_allocation_size) {
  IREE_ASSERT_ARGUMENT(shape);
  IREE_ASSERT_ARGUMENT(out_allocation_size);
  *out_allocation_size = 0;
  iree_device_size_t byte_length = iree_hal_element_byte_count(element_type);
  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    byte_length *= shape[i];
  }
  *out_allocation_size = byte_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_compute_view_offset(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type, const iree_hal_dim_t* indices,
    iree_host_size_t indices_count, iree_device_size_t* out_offset) {
  IREE_ASSERT_ARGUMENT(shape);
  IREE_ASSERT_ARGUMENT(indices);
  IREE_ASSERT_ARGUMENT(out_offset);
  *out_offset = 0;
  if (IREE_UNLIKELY(shape_rank != indices_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shape rank/indices mismatch: %zu != %zu",
                            shape_rank, indices_count);
  }

  iree_device_size_t offset = 0;
  for (iree_host_size_t i = 0; i < indices_count; ++i) {
    if (IREE_UNLIKELY(indices[i] >= shape[i])) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "index[%zu] out of bounds: %d >= %d", i,
                              indices[i], shape[i]);
    }
    iree_device_size_t axis_offset = indices[i];
    for (iree_host_size_t j = i + 1; j < shape_rank; ++j) {
      axis_offset *= shape[j];
    }
    offset += axis_offset;
  }
  offset *= iree_hal_element_byte_count(element_type);

  *out_offset = offset;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_compute_view_range(
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type, const iree_hal_dim_t* start_indices,
    iree_host_size_t indices_count, const iree_hal_dim_t* lengths,
    iree_host_size_t lengths_count, iree_device_size_t* out_start_offset,
    iree_device_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(shape);
  IREE_ASSERT_ARGUMENT(start_indices);
  IREE_ASSERT_ARGUMENT(lengths);
  IREE_ASSERT_ARGUMENT(out_start_offset);
  IREE_ASSERT_ARGUMENT(out_length);
  *out_start_offset = 0;
  *out_length = 0;
  if (IREE_UNLIKELY(indices_count != lengths_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indices/lengths mismatch: %zu != %zu",
                            indices_count, lengths_count);
  }
  if (IREE_UNLIKELY(shape_rank != indices_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shape rank/indices mismatch: %zu != %zu",
                            shape_rank, indices_count);
  }

  iree_hal_dim_t* end_indices =
      iree_alloca(shape_rank * sizeof(iree_hal_dim_t));
  iree_device_size_t element_size = iree_hal_element_byte_count(element_type);
  iree_device_size_t subspan_length = element_size;
  for (iree_host_size_t i = 0; i < lengths_count; ++i) {
    subspan_length *= lengths[i];
    end_indices[i] = start_indices[i] + lengths[i] - 1;
  }

  iree_device_size_t start_byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_compute_view_offset(
      shape, shape_rank, element_type, start_indices, indices_count,
      &start_byte_offset));
  iree_device_size_t end_byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_compute_view_offset(
      shape, shape_rank, element_type, end_indices, shape_rank,
      &end_byte_offset));

  // Non-contiguous regions not yet implemented. Will be easier to detect when
  // we have strides.
  iree_device_size_t offset_length =
      end_byte_offset - start_byte_offset + element_size;
  if (subspan_length != offset_length) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-contiguous range region computation not implemented");
  }

  *out_start_offset = start_byte_offset;
  *out_length = subspan_length;
  return iree_ok_status();
}
