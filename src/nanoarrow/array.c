// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow.h"

static void ArrowArrayRelease(struct ArrowArray* array) {
  // Release buffers held by this array
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  if (private_data != NULL) {
    ArrowBitmapReset(&private_data->bitmap);
    ArrowBufferReset(&private_data->buffers[0]);
    ArrowBufferReset(&private_data->buffers[1]);
    ArrowFree(private_data);
  }

  // This object owns the memory for all the children, but those
  // children may have been generated elsewhere and might have
  // their own release() callback.
  if (array->children != NULL) {
    for (int64_t i = 0; i < array->n_children; i++) {
      if (array->children[i] != NULL) {
        if (array->children[i]->release != NULL) {
          array->children[i]->release(array->children[i]);
        }

        ArrowFree(array->children[i]);
      }
    }

    ArrowFree(array->children);
  }

  // This object owns the memory for the dictionary but it
  // may have been generated somewhere else and have its own
  // release() callback.
  if (array->dictionary != NULL) {
    if (array->dictionary->release != NULL) {
      array->dictionary->release(array->dictionary);
    }

    ArrowFree(array->dictionary);
  }

  // Mark released
  array->release = NULL;
}

static ArrowErrorCode ArrowArraySetStorageType(struct ArrowArray* array,
                                               enum ArrowType storage_type) {
  switch (storage_type) {
    case NANOARROW_TYPE_UNINITIALIZED:
    case NANOARROW_TYPE_NA:
      array->n_buffers = 0;
      break;

    case NANOARROW_TYPE_FIXED_SIZE_LIST:
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_SPARSE_UNION:
      array->n_buffers = 1;
      break;

    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_MAP:
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
    case NANOARROW_TYPE_INTERVAL_MONTHS:
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
    case NANOARROW_TYPE_DENSE_UNION:
      array->n_buffers = 2;
      break;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      array->n_buffers = 3;
      break;

    default:
      return EINVAL;

      return NANOARROW_OK;
  }

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  private_data->storage_type = storage_type;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayInitFromType(struct ArrowArray* array,
                                      enum ArrowType storage_type) {
  array->length = 0;
  array->null_count = 0;
  array->offset = 0;
  array->n_buffers = 0;
  array->n_children = 0;
  array->buffers = NULL;
  array->children = NULL;
  array->dictionary = NULL;
  array->release = &ArrowArrayRelease;
  array->private_data = NULL;

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)ArrowMalloc(sizeof(struct ArrowArrayPrivateData));
  if (private_data == NULL) {
    array->release = NULL;
    return ENOMEM;
  }

  ArrowBitmapInit(&private_data->bitmap);
  ArrowBufferInit(&private_data->buffers[0]);
  ArrowBufferInit(&private_data->buffers[1]);
  private_data->buffer_data[0] = NULL;
  private_data->buffer_data[1] = NULL;
  private_data->buffer_data[2] = NULL;

  array->private_data = private_data;
  array->buffers = (const void**)(&private_data->buffer_data);

  int result = ArrowArraySetStorageType(array, storage_type);
  if (result != NANOARROW_OK) {
    array->release(array);
    return result;
  }

  ArrowLayoutInit(&private_data->layout, storage_type);
  // We can only know this not to be true when initializing based on a schema
  // so assume this to be true.
  private_data->union_type_id_is_child_index = 1;
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayInitFromArrayView(struct ArrowArray* array,
                                                  struct ArrowArrayView* array_view,
                                                  struct ArrowError* error) {
  ArrowArrayInitFromType(array, array_view->storage_type);
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  int result = ArrowArrayAllocateChildren(array, array_view->n_children);
  if (result != NANOARROW_OK) {
    array->release(array);
    return result;
  }

  private_data->layout = array_view->layout;

  for (int64_t i = 0; i < array_view->n_children; i++) {
    int result =
        ArrowArrayInitFromArrayView(array->children[i], array_view->children[i], error);
    if (result != NANOARROW_OK) {
      array->release(array);
      return result;
    }
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayInitFromSchema(struct ArrowArray* array,
                                        struct ArrowSchema* schema,
                                        struct ArrowError* error) {
  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromSchema(&array_view, schema, error));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromArrayView(array, &array_view, error));
  if (array_view.storage_type == NANOARROW_TYPE_DENSE_UNION ||
      array_view.storage_type == NANOARROW_TYPE_SPARSE_UNION) {
    struct ArrowArrayPrivateData* private_data =
        (struct ArrowArrayPrivateData*)array->private_data;
    // We can still build arrays if this isn't true; however, the append
    // functions won't work. Instead, we store this value and error only
    // when StartAppending is called.
    private_data->union_type_id_is_child_index =
        _ArrowUnionTypeIdsWillEqualChildIndices(schema->format + 4, schema->n_children);
  }

  ArrowArrayViewReset(&array_view);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayAllocateChildren(struct ArrowArray* array, int64_t n_children) {
  if (array->children != NULL) {
    return EINVAL;
  }

  if (n_children == 0) {
    return NANOARROW_OK;
  }

  array->children =
      (struct ArrowArray**)ArrowMalloc(n_children * sizeof(struct ArrowArray*));
  if (array->children == NULL) {
    return ENOMEM;
  }

  for (int64_t i = 0; i < n_children; i++) {
    array->children[i] = NULL;
  }

  for (int64_t i = 0; i < n_children; i++) {
    array->children[i] = (struct ArrowArray*)ArrowMalloc(sizeof(struct ArrowArray));
    if (array->children[i] == NULL) {
      return ENOMEM;
    }
    array->children[i]->release = NULL;
  }

  array->n_children = n_children;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayAllocateDictionary(struct ArrowArray* array) {
  if (array->dictionary != NULL) {
    return EINVAL;
  }

  array->dictionary = (struct ArrowArray*)ArrowMalloc(sizeof(struct ArrowArray));
  if (array->dictionary == NULL) {
    return ENOMEM;
  }

  array->dictionary->release = NULL;
  return NANOARROW_OK;
}

void ArrowArraySetValidityBitmap(struct ArrowArray* array, struct ArrowBitmap* bitmap) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  ArrowBufferMove(&bitmap->buffer, &private_data->bitmap.buffer);
  private_data->bitmap.size_bits = bitmap->size_bits;
  bitmap->size_bits = 0;
  private_data->buffer_data[0] = private_data->bitmap.buffer.data;
  array->null_count = -1;
}

ArrowErrorCode ArrowArraySetBuffer(struct ArrowArray* array, int64_t i,
                                   struct ArrowBuffer* buffer) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  switch (i) {
    case 0:
      ArrowBufferMove(buffer, &private_data->bitmap.buffer);
      private_data->buffer_data[i] = private_data->bitmap.buffer.data;
      break;
    case 1:
    case 2:
      ArrowBufferMove(buffer, &private_data->buffers[i - 1]);
      private_data->buffer_data[i] = private_data->buffers[i - 1].data;
      break;
    default:
      return EINVAL;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayViewInitFromArray(struct ArrowArrayView* array_view,
                                                  struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  ArrowArrayViewInitFromType(array_view, private_data->storage_type);
  array_view->layout = private_data->layout;
  array_view->array = array;

  int result = ArrowArrayViewAllocateChildren(array_view, array->n_children);
  if (result != NANOARROW_OK) {
    ArrowArrayViewReset(array_view);
    return result;
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    result = ArrowArrayViewInitFromArray(array_view->children[i], array->children[i]);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayReserveInternal(struct ArrowArray* array,
                                                struct ArrowArrayView* array_view) {
  // Loop through buffers and reserve the extra space that we know about
  for (int64_t i = 0; i < array->n_buffers; i++) {
    // Don't reserve on a validity buffer that hasn't been allocated yet
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_VALIDITY &&
        ArrowArrayBuffer(array, i)->data == NULL) {
      continue;
    }

    int64_t additional_size_bytes =
        array_view->buffer_views[i].n_bytes - ArrowArrayBuffer(array, i)->size_bytes;

    if (additional_size_bytes > 0) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferReserve(ArrowArrayBuffer(array, i), additional_size_bytes));
    }
  }

  // Recursively reserve children
  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayReserveInternal(array->children[i], array_view->children[i]));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayReserve(struct ArrowArray* array,
                                 int64_t additional_size_elements) {
  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromArray(&array_view, array));

  // Calculate theoretical buffer sizes (recursively)
  ArrowArrayViewSetLength(&array_view, array->length + additional_size_elements);

  // Walk the structure (recursively)
  int result = ArrowArrayReserveInternal(array, &array_view);
  ArrowArrayViewReset(&array_view);
  if (result != NANOARROW_OK) {
    return result;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayFinalizeBuffers(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  // The only buffer finalizing this currently does is make sure the data
  // buffer for (Large)String|Binary is never NULL
  switch (private_data->storage_type) {
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_LARGE_STRING:
      if (ArrowArrayBuffer(array, 2)->data == NULL) {
        ArrowBufferAppendUInt8(ArrowArrayBuffer(array, 2), 0);
      }
      break;
    default:
      break;
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayFinalizeBuffers(array->children[i]));
  }

  return NANOARROW_OK;
}

static void ArrowArrayFlushInternalPointers(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  for (int64_t i = 0; i < 3; i++) {
    private_data->buffer_data[i] = ArrowArrayBuffer(array, i)->data;
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    ArrowArrayFlushInternalPointers(array->children[i]);
  }
}

static ArrowErrorCode ArrowArrayCheckInternalBufferSizes(
    struct ArrowArray* array, struct ArrowArrayView* array_view, char set_length,
    struct ArrowError* error) {
  if (set_length) {
    ArrowArrayViewSetLength(array_view, array->offset + array->length);
  }

  for (int64_t i = 0; i < array->n_buffers; i++) {
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_VALIDITY &&
        array->null_count == 0 && array->buffers[i] == NULL) {
      continue;
    }

    int64_t expected_size = array_view->buffer_views[i].n_bytes;
    int64_t actual_size = ArrowArrayBuffer(array, i)->size_bytes;

    if (actual_size < expected_size) {
      ArrowErrorSet(
          error,
          "Expected buffer %d to size >= %ld bytes but found buffer with %ld bytes", i,
          (long)expected_size, (long)actual_size);
      return EINVAL;
    }
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayCheckInternalBufferSizes(
        array->children[i], array_view->children[i], set_length, error));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayFinishBuilding(struct ArrowArray* array,
                                        struct ArrowError* error) {
  // Even if the data buffer is size zero, the value needs to be non-null
  NANOARROW_RETURN_NOT_OK(ArrowArrayFinalizeBuffers(array));

  // Make sure the value we get with array->buffers[i] is set to the actual
  // pointer (which may have changed from the original due to reallocation)
  ArrowArrayFlushInternalPointers(array);

  // Check buffer sizes to make sure we are not sending an ArrowArray
  // into the wild that is going to segfault
  struct ArrowArrayView array_view;

  NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromArray(&array_view, array));

  // Check buffer sizes once without using internal buffer data since
  // ArrowArrayViewSetArray() assumes that all the buffers are long enough
  // and issues invalid reads on offset buffers if they are not
  int result = ArrowArrayCheckInternalBufferSizes(array, &array_view, 1, error);
  if (result != NANOARROW_OK) {
    ArrowArrayViewReset(&array_view);
    return result;
  }

  result = ArrowArrayViewSetArray(&array_view, array, error);
  if (result != NANOARROW_OK) {
    ArrowArrayViewReset(&array_view);
    return result;
  }

  result = ArrowArrayCheckInternalBufferSizes(array, &array_view, 0, error);
  ArrowArrayViewReset(&array_view);
  return result;
}

void ArrowArrayViewInitFromType(struct ArrowArrayView* array_view,
                                enum ArrowType storage_type) {
  memset(array_view, 0, sizeof(struct ArrowArrayView));
  array_view->storage_type = storage_type;
  ArrowLayoutInit(&array_view->layout, storage_type);
}

ArrowErrorCode ArrowArrayViewAllocateChildren(struct ArrowArrayView* array_view,
                                              int64_t n_children) {
  if (array_view->children != NULL) {
    return EINVAL;
  }

  array_view->children =
      (struct ArrowArrayView**)ArrowMalloc(n_children * sizeof(struct ArrowArrayView*));
  if (array_view->children == NULL) {
    return ENOMEM;
  }

  for (int64_t i = 0; i < n_children; i++) {
    array_view->children[i] = NULL;
  }

  array_view->n_children = n_children;

  for (int64_t i = 0; i < n_children; i++) {
    array_view->children[i] =
        (struct ArrowArrayView*)ArrowMalloc(sizeof(struct ArrowArrayView));
    if (array_view->children[i] == NULL) {
      return ENOMEM;
    }
    ArrowArrayViewInitFromType(array_view->children[i], NANOARROW_TYPE_UNINITIALIZED);
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewInitFromSchema(struct ArrowArrayView* array_view,
                                            struct ArrowSchema* schema,
                                            struct ArrowError* error) {
  struct ArrowSchemaView schema_view;
  int result = ArrowSchemaViewInit(&schema_view, schema, error);
  if (result != NANOARROW_OK) {
    return result;
  }

  ArrowArrayViewInitFromType(array_view, schema_view.storage_data_type);
  array_view->layout = schema_view.layout;

  result = ArrowArrayViewAllocateChildren(array_view, schema->n_children);
  if (result != NANOARROW_OK) {
    ArrowArrayViewReset(array_view);
    return result;
  }

  for (int64_t i = 0; i < schema->n_children; i++) {
    result =
        ArrowArrayViewInitFromSchema(array_view->children[i], schema->children[i], error);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_SPARSE_UNION ||
      array_view->storage_type == NANOARROW_TYPE_DENSE_UNION) {
    array_view->union_type_id_map = (int8_t*)ArrowMalloc(256 * sizeof(int8_t));
    if (array_view->union_type_id_map == NULL) {
      return ENOMEM;
    }

    memset(array_view->union_type_id_map, -1, 256);
    int8_t n_type_ids = _ArrowParseUnionTypeIds(schema_view.union_type_ids,
                                                array_view->union_type_id_map + 128);
    for (int8_t child_index = 0; child_index < n_type_ids; child_index++) {
      int8_t type_id = array_view->union_type_id_map[128 + child_index];
      array_view->union_type_id_map[type_id] = child_index;
    }
  }

  return NANOARROW_OK;
}

void ArrowArrayViewReset(struct ArrowArrayView* array_view) {
  if (array_view->children != NULL) {
    for (int64_t i = 0; i < array_view->n_children; i++) {
      if (array_view->children[i] != NULL) {
        ArrowArrayViewReset(array_view->children[i]);
        ArrowFree(array_view->children[i]);
      }
    }

    ArrowFree(array_view->children);
  }

  if (array_view->union_type_id_map != NULL) {
    ArrowFree(array_view->union_type_id_map);
  }

  ArrowArrayViewInitFromType(array_view, NANOARROW_TYPE_UNINITIALIZED);
}

void ArrowArrayViewSetLength(struct ArrowArrayView* array_view, int64_t length) {
  for (int i = 0; i < 3; i++) {
    int64_t element_size_bytes = array_view->layout.element_size_bits[i] / 8;
    array_view->buffer_views[i].data.data = NULL;

    switch (array_view->layout.buffer_type[i]) {
      case NANOARROW_BUFFER_TYPE_VALIDITY:
        array_view->buffer_views[i].n_bytes = _ArrowBytesForBits(length);
        continue;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        // Probably don't want/need to rely on the producer to have allocated an
        // offsets buffer of length 1 for a zero-size array
        array_view->buffer_views[i].n_bytes =
            (length != 0) * element_size_bytes * (length + 1);
        continue;
      case NANOARROW_BUFFER_TYPE_DATA:
        array_view->buffer_views[i].n_bytes =
            _ArrowRoundUpToMultipleOf8(array_view->layout.element_size_bits[i] * length) /
            8;
        continue;
      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        array_view->buffer_views[i].n_bytes = element_size_bytes * length;
        continue;
      case NANOARROW_BUFFER_TYPE_NONE:
        array_view->buffer_views[i].n_bytes = 0;
        continue;
    }
  }

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_SPARSE_UNION:
      for (int64_t i = 0; i < array_view->n_children; i++) {
        ArrowArrayViewSetLength(array_view->children[i], length);
      }
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      if (array_view->n_children >= 1) {
        ArrowArrayViewSetLength(array_view->children[0],
                                length * array_view->layout.child_size_elements);
      }
    default:
      break;
  }
}

ArrowErrorCode ArrowArrayViewSetArray(struct ArrowArrayView* array_view,
                                      struct ArrowArray* array,
                                      struct ArrowError* error) {
  array_view->array = array;
  ArrowArrayViewSetLength(array_view, array->offset + array->length);

  int64_t buffers_required = 0;
  for (int i = 0; i < 3; i++) {
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    buffers_required++;

    // If the null_count is 0, the validity buffer can be NULL
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_VALIDITY &&
        array->null_count == 0 && array->buffers[i] == NULL) {
      array_view->buffer_views[i].n_bytes = 0;
    }

    array_view->buffer_views[i].data.data = array->buffers[i];
  }

  if (buffers_required != array->n_buffers) {
    ArrowErrorSet(error, "Expected array with %d buffer(s) but found %d buffer(s)",
                  (int)buffers_required, (int)array->n_buffers);
    return EINVAL;
  }

  if (array_view->n_children != array->n_children) {
    return EINVAL;
  }

  // Check child sizes and calculate sizes that depend on data in the array buffers
  int64_t last_offset;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      if (array_view->buffer_views[1].n_bytes != 0) {
        last_offset =
            array_view->buffer_views[1].data.as_int32[array->offset + array->length];
        array_view->buffer_views[2].n_bytes = last_offset;
      }
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (array_view->buffer_views[1].n_bytes != 0) {
        last_offset =
            array_view->buffer_views[1].data.as_int64[array->offset + array->length];
        array_view->buffer_views[2].n_bytes = last_offset;
      }
      break;
    case NANOARROW_TYPE_STRUCT:
      for (int64_t i = 0; i < array_view->n_children; i++) {
        if (array->children[i]->length < (array->offset + array->length)) {
          ArrowErrorSet(
              error,
              "Expected struct child %d to have length >= %ld but found child with "
              "length %ld",
              (int)(i + 1), (long)(array->offset + array->length),
              (long)array->children[i]->length);
          return EINVAL;
        }
      }
      break;
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_MAP: {
      const char* type_name =
          array_view->storage_type == NANOARROW_TYPE_LIST ? "list" : "map";
      if (array->n_children != 1) {
        ArrowErrorSet(error, "Expected 1 child of %s array but found %d child arrays",
                      type_name, (int)array->n_children);
        return EINVAL;
      }

      if (array_view->buffer_views[1].n_bytes != 0) {
        last_offset =
            array_view->buffer_views[1].data.as_int32[array->offset + array->length];
        if (array->children[0]->length < last_offset) {
          ArrowErrorSet(
              error,
              "Expected child of %s array with length >= %ld but found array with "
              "length %ld",
              type_name, (long)last_offset, (long)array->children[0]->length);
          return EINVAL;
        }
      }
      break;
    }
    case NANOARROW_TYPE_LARGE_LIST:
      if (array->n_children != 1) {
        ArrowErrorSet(error,
                      "Expected 1 child of large list array but found %d child arrays",
                      (int)array->n_children);
        return EINVAL;
      }

      if (array_view->buffer_views[1].n_bytes != 0) {
        last_offset =
            array_view->buffer_views[1].data.as_int64[array->offset + array->length];
        if (array->children[0]->length < last_offset) {
          ArrowErrorSet(
              error,
              "Expected child of large list array with length >= %ld but found array "
              "with length %ld",
              (long)last_offset, (long)array->children[0]->length);
          return EINVAL;
        }
      }
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      if (array->n_children != 1) {
        ArrowErrorSet(error,
                      "Expected 1 child of fixed-size array but found %d child arrays",
                      (int)array->n_children);
        return EINVAL;
      }

      last_offset =
          (array->offset + array->length) * array_view->layout.child_size_elements;
      if (array->children[0]->length < last_offset) {
        ArrowErrorSet(
            error,
            "Expected child of fixed-size list array with length >= %ld but found array "
            "with length %ld",
            (long)last_offset, (long)array->children[0]->length);
        return EINVAL;
      }
      break;
    default:
      break;
  }

  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewSetArray(array_view->children[i], array->children[i], error));
  }

  return NANOARROW_OK;
}
