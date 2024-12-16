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

#include "nanoarrow/nanoarrow_ipc.h"

static ArrowErrorCode ArrowIpcDefaultDecompressorAdd(
    struct ArrowIpcDecompressor* decompressor,
    enum ArrowIpcCompressionType compression_type, struct ArrowBufferView* src,
    uint8_t* dst, int64_t* dst_size) {
  NANOARROW_UNUSED(decompressor);
  NANOARROW_UNUSED(compression_type);
  NANOARROW_UNUSED(src);
  NANOARROW_UNUSED(dst);
  NANOARROW_UNUSED(dst_size);
  return ENOTSUP;
}

static ArrowErrorCode ArrowIpcDefaultDecompressorWait(
    struct ArrowIpcDecompressor* decompressor, int64_t timeout_ms,
    struct ArrowError* error) {
  NANOARROW_UNUSED(decompressor);
  NANOARROW_UNUSED(timeout_ms);
  ArrowErrorSet(error, "Decompression is not supported in this build of nanoarrow_ipc");
  return ENOTSUP;
}

static void ArrowIpcDefaultDecompressorRelease(
    struct ArrowIpcDecompressor* decompressor) {
  decompressor->release = NULL;
}

ArrowErrorCode ArrowIpcGetDefaultDecompressor(struct ArrowIpcDecompressor* decompressor) {
  decompressor->decompress_add = &ArrowIpcDefaultDecompressorAdd;
  decompressor->decompress_wait = &ArrowIpcDefaultDecompressorWait;
  decompressor->release = &ArrowIpcDefaultDecompressorRelease;
  decompressor->private_data = NULL;
  return NANOARROW_OK;
}
