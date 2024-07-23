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

// For thread safe shared buffers we need C11 + stdatomic.h
// Can compile with -DNANOARROW_IPC_USE_STDATOMIC=0 or 1 to override
// automatic detection
#if !defined(NANOARROW_IPC_USE_STDATOMIC)
#define NANOARROW_IPC_USE_STDATOMIC 0

// Check for C11
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L

// Check for GCC 4.8, which doesn't include stdatomic.h but does
// not define __STDC_NO_ATOMICS__
#if defined(__clang__) || !defined(__GNUC__) || __GNUC__ >= 5

#if !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
#undef NANOARROW_IPC_USE_STDATOMIC
#define NANOARROW_IPC_USE_STDATOMIC 1
#endif
#endif
#endif

#endif

#include "nanoarrow/ipc/flatcc_generated.h"
#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"

// R 3.6 / Windows builds on a very old toolchain that does not define ENODATA
#if defined(_WIN32) && !defined(_MSC_VER) && !defined(ENODATA)
#define ENODATA 120
#endif

#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(org_apache_arrow_flatbuf, x)

#define FLATCC_RETURN_UNLESS_0(x) \
  if (ns(x) != 0) return ENOMEM;

struct ArrowIpcEncoderPrivate {
  flatcc_builder_t builder;
  struct ArrowBuffer buffers, nodes;
};

ArrowErrorCode ArrowIpcEncoderInit(struct ArrowIpcEncoder* encoder) {
  NANOARROW_DCHECK(encoder);
  memset(encoder, 0, sizeof(struct ArrowIpcEncoder));
  encoder->encode_buffer = NULL;
  encoder->encode_buffer_state = NULL;
  encoder->codec = NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  encoder->private_data = ArrowMalloc(sizeof(struct ArrowIpcEncoderPrivate));
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;
  ArrowBufferInit(&private->buffers);
  ArrowBufferInit(&private->nodes);
  if (flatcc_builder_init(&private->builder) == -1) {
    return ESPIPE;
  }
  return NANOARROW_OK;
}

void ArrowIpcEncoderReset(struct ArrowIpcEncoder* encoder) {
  NANOARROW_DCHECK(encoder && encoder->private_data);
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
  NANOARROW_DCHECK(encoder && encoder->private_data && out);
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;
  ArrowBufferReset(out);
  size_t size = flatcc_builder_get_buffer_size(&private->builder);
  if (size == 0) {
    // Finalizing an empty flatcc_builder_t triggers an assertion
    return NANOARROW_OK;
  }

  out->size_bytes = out->capacity_bytes = (int64_t)size;
  out->data = (uint8_t*)flatcc_builder_finalize_buffer(&private->builder, &size);
  return out->data ? NANOARROW_OK : ENOMEM;
}
