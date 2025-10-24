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

#include <inttypes.h>

#include "nanoarrow/nanoarrow_ipc.h"

#if defined(NANOARROW_IPC_WITH_ZSTD)
#include <zstd.h>

static ArrowErrorCode ArrowIpcDecompressZstd(struct ArrowBufferView src, uint8_t* dst,
                                             int64_t dst_size, struct ArrowError* error) {
  size_t code =
      ZSTD_decompress((void*)dst, (size_t)dst_size, src.data.data, src.size_bytes);
  if (ZSTD_isError(code)) {
    ArrowErrorSet(error,
                  "ZSTD_decompress([buffer with %" PRId64
                  " bytes] -> [buffer with %" PRId64 " bytes]) failed with error '%s'",
                  src.size_bytes, dst_size, ZSTD_getErrorName(code));
    return EIO;
  }

  if (dst_size != (int64_t)code) {
    ArrowErrorSet(error,
                  "Expected decompressed size of %" PRId64 " bytes but got %" PRId64
                  " bytes",
                  dst_size, (int64_t)code);
    return EIO;
  }

  return NANOARROW_OK;
}
#endif

ArrowIpcDecompressFunction ArrowIpcGetZstdDecompressionFunction(void) {
#if defined(NANOARROW_IPC_WITH_ZSTD)
  return &ArrowIpcDecompressZstd;
#else
  return NULL;
#endif
}

#if defined(NANOARROW_IPC_WITH_LZ4)
#include <lz4.h>
#include <lz4frame.h>
#include <lz4hc.h>

static ArrowErrorCode ArrowIpcDecompressLZ4(struct ArrowBufferView src, uint8_t* dst,
                                            int64_t dst_size, struct ArrowError* error) {
  LZ4F_errorCode_t ret;
  LZ4F_decompressionContext_t ctx = NULL;

  ret = LZ4F_createDecompressionContext(&ctx, LZ4F_VERSION);
  if (LZ4F_isError(ret)) {
    ArrowErrorSet(error, "LZ4 init failed");
    return EIO;
  }

  size_t dst_capacity = dst_size;
  size_t src_size = src.size_bytes;
  ret = LZ4F_decompress(ctx, dst, &dst_capacity, src.data.data, &src_size,
                        NULL /* options */);
  if (LZ4F_isError(ret)) {
    NANOARROW_UNUSED(LZ4F_freeDecompressionContext(ctx));
    ArrowErrorSet(error,
                  "LZ4F_decompress([buffer with %" PRId64
                  " bytes] -> [buffer with %" PRId64 " bytes]) failed",
                  src.size_bytes, dst_size);
    return EIO;
  }

  if ((int64_t)dst_capacity != dst_size) {
    NANOARROW_UNUSED(LZ4F_freeDecompressionContext(ctx));
    ArrowErrorSet(error,
                  "Expected decompressed size of %" PRId64 " bytes but got %" PRId64
                  " bytes",
                  dst_size, (int64_t)dst_capacity);
    return EIO;
  }

  if (ret != 0) {
    NANOARROW_UNUSED(LZ4F_freeDecompressionContext(ctx));
    ArrowErrorSet(
        error, "Expected complete frame but found frame with %" PRId64 " bytes remaining",
        (int64_t)ret);
    return EIO;
  }

  NANOARROW_UNUSED(LZ4F_freeDecompressionContext(ctx));
  return NANOARROW_OK;
}
#endif

ArrowIpcDecompressFunction ArrowIpcGetLZ4DecompressionFunction(void) {
#if defined(NANOARROW_IPC_WITH_LZ4)
  return &ArrowIpcDecompressLZ4;
#else
  return NULL;
#endif
}

struct ArrowIpcSerialDecompressorPrivate {
  ArrowIpcDecompressFunction decompress_functions[3];
};

static ArrowErrorCode ArrowIpcSerialDecompressorAdd(
    struct ArrowIpcDecompressor* decompressor,
    enum ArrowIpcCompressionType compression_type, struct ArrowBufferView src,
    uint8_t* dst, int64_t dst_size, struct ArrowError* error) {
  struct ArrowIpcSerialDecompressorPrivate* private_data =
      (struct ArrowIpcSerialDecompressorPrivate*)decompressor->private_data;

  ArrowIpcDecompressFunction fn = NULL;
  switch (compression_type) {
    case NANOARROW_IPC_COMPRESSION_TYPE_ZSTD:
    case NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME:
      fn = private_data->decompress_functions[compression_type];
      break;
    default:
      ArrowErrorSet(error, "Unknown decompression type with value %d",
                    (int)compression_type);
      return EINVAL;
  }

  if (fn == NULL) {
    ArrowErrorSet(
        error, "Compression type with value %d not supported by this build of nanoarrow",
        (int)compression_type);
    return ENOTSUP;
  }

  NANOARROW_RETURN_NOT_OK(fn(src, dst, dst_size, error));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcSerialDecompressorWait(
    struct ArrowIpcDecompressor* decompressor, int64_t timeout_ms,
    struct ArrowError* error) {
  NANOARROW_UNUSED(decompressor);
  NANOARROW_UNUSED(timeout_ms);
  NANOARROW_UNUSED(error);
  return NANOARROW_OK;
}

static void ArrowIpcSerialDecompressorRelease(struct ArrowIpcDecompressor* decompressor) {
  ArrowFree(decompressor->private_data);
  decompressor->release = NULL;
}

ArrowErrorCode ArrowIpcSerialDecompressor(struct ArrowIpcDecompressor* decompressor) {
  decompressor->decompress_add = &ArrowIpcSerialDecompressorAdd;
  decompressor->decompress_wait = &ArrowIpcSerialDecompressorWait;
  decompressor->release = &ArrowIpcSerialDecompressorRelease;
  decompressor->private_data =
      ArrowMalloc(sizeof(struct ArrowIpcSerialDecompressorPrivate));
  if (decompressor->private_data == NULL) {
    return ENOMEM;
  }

  memset(decompressor->private_data, 0, sizeof(struct ArrowIpcSerialDecompressorPrivate));
  ArrowIpcSerialDecompressorSetFunction(decompressor, NANOARROW_IPC_COMPRESSION_TYPE_ZSTD,
                                        ArrowIpcGetZstdDecompressionFunction());
  ArrowIpcSerialDecompressorSetFunction(decompressor,
                                        NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
                                        ArrowIpcGetLZ4DecompressionFunction());
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcSerialDecompressorSetFunction(
    struct ArrowIpcDecompressor* decompressor,
    enum ArrowIpcCompressionType compression_type,
    ArrowIpcDecompressFunction decompress_function) {
  struct ArrowIpcSerialDecompressorPrivate* private_data =
      (struct ArrowIpcSerialDecompressorPrivate*)decompressor->private_data;

  switch (compression_type) {
    case NANOARROW_IPC_COMPRESSION_TYPE_ZSTD:
    case NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME:
      break;
    default:
      return EINVAL;
  }

  private_data->decompress_functions[compression_type] = decompress_function;
  return NANOARROW_OK;
}
