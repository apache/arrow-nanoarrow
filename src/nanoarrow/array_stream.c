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

#include "nanoarrow.h"

struct BasicArrayStreamPrivate {
  struct ArrowSchema schema;
  int64_t n_arrays;
  struct ArrowArray* arrays;
  int64_t arrays_i;
};

static int ArrowBasicArrayStreamGetSchema(struct ArrowArrayStream* array_stream,
                                          struct ArrowSchema* schema) {
  if (array_stream == NULL || array_stream->release == NULL) {
    return EINVAL;
  }

  struct BasicArrayStreamPrivate* private =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;
  return ArrowSchemaDeepCopy(&private->schema, schema);
}

static int ArrowBasicArrayStreamGetNext(struct ArrowArrayStream* array_stream,
                                        struct ArrowArray* array) {
  if (array_stream == NULL || array_stream->release == NULL) {
    return EINVAL;
  }

  struct BasicArrayStreamPrivate* private =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;

  if (private->arrays_i == private->n_arrays) {
    array->release = NULL;
    return NANOARROW_OK;
  }

  ArrowArrayMove(&private->arrays[private->arrays_i++], array);
  return NANOARROW_OK;
}

static const char* ArrowBasicArrayStreamGetLastError(
    struct ArrowArrayStream* array_stream) {
  return NULL;
}

static void ArrowBasicArrayStreamRelease(struct ArrowArrayStream* array_stream) {
  if (array_stream == NULL || array_stream->release == NULL) {
    return;
  }

  struct BasicArrayStreamPrivate* private =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;

  if (private->schema.release != NULL) {
    private->schema.release(&private->schema);
  }

  for (int64_t i = 0; i < private->n_arrays; i++) {
    if (private->arrays[i].release != NULL) {
      private->arrays[i].release(&private->arrays[i]);
    }
  }

  if (private->arrays != NULL) {
    ArrowFree(private->arrays);
  }

  ArrowFree(private);
  array_stream->release = NULL;
}

ArrowErrorCode ArrowBasicArrayStreamInit(struct ArrowArrayStream* array_stream,
                                         struct ArrowSchema* schema, int64_t n_arrays) {
  struct BasicArrayStreamPrivate* private = (struct BasicArrayStreamPrivate*)ArrowMalloc(
      sizeof(struct BasicArrayStreamPrivate));
  if (private == NULL) {
    return ENOMEM;
  }

  ArrowSchemaMove(schema, &private->schema);

  private->n_arrays = n_arrays;
  private->arrays = NULL;
  private->arrays_i = 0;

  if (n_arrays > 0) {
    private->arrays =
        (struct ArrowArray*)ArrowMalloc(n_arrays * sizeof(struct ArrowArray));
    if (private->arrays == NULL) {
      ArrowBasicArrayStreamRelease(array_stream);
      return ENOMEM;
    }
  }

  for (int64_t i = 0; i < private->n_arrays; i++) {
    private->arrays[i].release = NULL;
  }

  array_stream->get_schema = &ArrowBasicArrayStreamGetSchema;
  array_stream->get_next = &ArrowBasicArrayStreamGetNext;
  array_stream->get_last_error = ArrowBasicArrayStreamGetLastError;
  array_stream->release = ArrowBasicArrayStreamRelease;
  array_stream->private_data = private;
  return NANOARROW_OK;
}

void ArrowBasicArrayStreamSetArray(struct ArrowArrayStream* array_stream, int64_t i,
                                   struct ArrowArray* array) {
  struct BasicArrayStreamPrivate* private =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;
  ArrowArrayMove(array, &private->arrays[i]);
}

ArrowErrorCode ArrowBasicArrayStreamValidate(struct ArrowArrayStream* array_stream,
                                             struct ArrowError* error) {
  struct BasicArrayStreamPrivate* private =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;

  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(&array_view, &private->schema, error));

  for (int64_t i = 0; i < private->n_arrays; i++) {
    if (private->arrays[i].release != NULL) {
      int result = ArrowArrayViewSetArray(&array_view, &private->arrays[i], error);
      if (result != NANOARROW_OK) {
        ArrowArrayViewReset(&array_view);
        return result;
      }
    }
  }

  ArrowArrayViewReset(&array_view);
  return NANOARROW_OK;
}
