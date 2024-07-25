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
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "flatcc/flatcc_builder.h"
#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"

struct ArrowIpcEncoderPrivate {
  flatcc_builder_t builder;
  struct ArrowBuffer buffers;
  struct ArrowBuffer nodes;
};

ArrowErrorCode ArrowIpcEncoderInit(struct ArrowIpcEncoder* encoder) {
  NANOARROW_DCHECK(encoder != NULL);
  memset(encoder, 0, sizeof(struct ArrowIpcEncoder));
  encoder->encode_buffer = NULL;
  encoder->encode_buffer_state = NULL;
  encoder->codec = NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  encoder->private_data = ArrowMalloc(sizeof(struct ArrowIpcEncoderPrivate));
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;
  if (flatcc_builder_init(&private->builder) == -1) {
    ArrowFree(private);
    return ESPIPE;
  }
  ArrowBufferInit(&private->buffers);
  ArrowBufferInit(&private->nodes);
  return NANOARROW_OK;
}

void ArrowIpcEncoderReset(struct ArrowIpcEncoder* encoder) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL);
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;
  flatcc_builder_clear(&private->builder);
  ArrowBufferReset(&private->nodes);
  ArrowBufferReset(&private->buffers);
  ArrowFree(private);
  memset(encoder, 0, sizeof(struct ArrowIpcEncoder));
}

ArrowErrorCode ArrowIpcEncoderFinalizeBuffer(struct ArrowIpcEncoder* encoder,
                                             struct ArrowBuffer* out) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL && out != NULL);
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  int64_t size = (int64_t)flatcc_builder_get_buffer_size(&private->builder);
  if (size == 0) {
    // Finalizing an empty flatcc_builder_t triggers an assertion
    return NANOARROW_OK;
  }

  void* data = flatcc_builder_get_direct_buffer(&private->builder, NULL);
  if (data == NULL) {
    return ENOMEM;
  }

  NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(out, data, size));

  // don't deallocate yet, just wipe the builder's current Message
  flatcc_builder_reset(&private->builder);
  return NANOARROW_OK;
}
