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
#include <limits.h>

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

static ArrowErrorCode ArrowIpcCompressZstd(struct ArrowBufferView src,
                                           int compression_level, struct ArrowBuffer* dst,
                                           struct ArrowError* error) {
  size_t dst_capacity = ZSTD_compressBound((size_t)src.size_bytes);
  if (ZSTD_isError(dst_capacity)) {
    ArrowErrorSet(error, "ZSTD_compressBound(%" PRId64 ") failed with error '%s'",
                  src.size_bytes, ZSTD_getErrorName(dst_capacity));
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferReserve(dst, (int64_t)dst_capacity),
                                     error);

  size_t code = ZSTD_compress((void*)(dst->data + dst->size_bytes), dst_capacity,
                              src.data.data, (size_t)src.size_bytes, compression_level);
  if (ZSTD_isError(code)) {
    ArrowErrorSet(error,
                  "ZSTD_compress([buffer with %" PRId64 " bytes]) failed with error '%s'",
                  src.size_bytes, ZSTD_getErrorName(code));
    return EIO;
  }

  dst->size_bytes += (int64_t)code;
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

ArrowIpcCompressFunction ArrowIpcGetZstdCompressionFunction(void) {
#if defined(NANOARROW_IPC_WITH_ZSTD)
  return &ArrowIpcCompressZstd;
#else
  return NULL;
#endif
}

#if defined(NANOARROW_IPC_WITH_LZ4)
#include <lz4.h>
#include <lz4frame.h>

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
    ArrowErrorSet(error,
                  "Expected complete LZ4 frame but found frame with %" PRId64
                  " bytes remaining",
                  (int64_t)ret);
    return EIO;
  }

  NANOARROW_UNUSED(LZ4F_freeDecompressionContext(ctx));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcCompressLZ4(struct ArrowBufferView src,
                                          int compression_level, struct ArrowBuffer* dst,
                                          struct ArrowError* error) {
  // LZ4 computes acceleration as -level + 1 before clamping it. Keep that
  // calculation representable even for the most negative int values.
  if (compression_level < 1 - INT_MAX) {
    compression_level = 1 - INT_MAX;
  }

  // Default preferences except for the compression level (no content size, no
  // checksums). This produces a single complete frame, which is what
  // ArrowIpcDecompressLZ4() and Arrow C++ expect.
  LZ4F_preferences_t prefs;
  memset(&prefs, 0, sizeof(prefs));
  prefs.compressionLevel = compression_level;

  size_t dst_capacity = LZ4F_compressFrameBound((size_t)src.size_bytes, &prefs);
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferReserve(dst, (int64_t)dst_capacity),
                                     error);

  size_t code = LZ4F_compressFrame((void*)(dst->data + dst->size_bytes), dst_capacity,
                                   src.data.data, (size_t)src.size_bytes, &prefs);
  if (LZ4F_isError(code)) {
    ArrowErrorSet(error,
                  "LZ4F_compressFrame([buffer with %" PRId64
                  " bytes]) failed with error '%s'",
                  src.size_bytes, LZ4F_getErrorName(code));
    return EIO;
  }

  dst->size_bytes += (int64_t)code;
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

ArrowIpcCompressFunction ArrowIpcGetLZ4CompressionFunction(void) {
#if defined(NANOARROW_IPC_WITH_LZ4)
  return &ArrowIpcCompressLZ4;
#else
  return NULL;
#endif
}

ArrowErrorCode ArrowIpcGetCompressionLevelRange(
    enum ArrowIpcCompressionType compression_type, int* min_level_out,
    int* max_level_out) {
  NANOARROW_DCHECK(min_level_out != NULL && max_level_out != NULL);
  // (unused when neither codec is built in)
  NANOARROW_UNUSED(min_level_out);
  NANOARROW_UNUSED(max_level_out);

  switch (compression_type) {
    case NANOARROW_IPC_COMPRESSION_TYPE_ZSTD:
#if defined(NANOARROW_IPC_WITH_ZSTD)
#if ZSTD_VERSION_NUMBER >= 10400
      *min_level_out = ZSTD_minCLevel();
#else
      // Negative (fast) levels can't be queried before zstd 1.4.0
      *min_level_out = NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT;
#endif
      *max_level_out = ZSTD_maxCLevel();
      return NANOARROW_OK;
#else
      return ENOTSUP;
#endif
    case NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME:
#if defined(NANOARROW_IPC_WITH_LZ4)
      // A negative level selects an acceleration of 1 - level, which lz4 caps at 65537
      // (LZ4_ACCELERATION_MAX, which is not part of its public headers)
      *min_level_out = 1 - 65537;
      *max_level_out = LZ4F_compressionLevel_max();
      return NANOARROW_OK;
#else
      return ENOTSUP;
#endif
    default:
      return EINVAL;
  }
}

// The serial decompressor and compressor keep one function per codec, indexed by
// enum ArrowIpcCompressionType (NONE is never a codec)
static int ArrowIpcCompressionTypeIsCodec(enum ArrowIpcCompressionType compression_type) {
  switch (compression_type) {
    case NANOARROW_IPC_COMPRESSION_TYPE_ZSTD:
    case NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME:
      return 1;
    default:
      return 0;
  }
}

const char* ArrowIpcCompressionTypeToString(
    enum ArrowIpcCompressionType compression_type) {
  switch (compression_type) {
    case NANOARROW_IPC_COMPRESSION_TYPE_NONE:
      return "none";
    case NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME:
      return "lz4";
    case NANOARROW_IPC_COMPRESSION_TYPE_ZSTD:
      return "zstd";
    default:
      return "<unknown compression type>";
  }
}

ArrowErrorCode ArrowIpcCompressionTypeFromString(
    const char* name, enum ArrowIpcCompressionType* compression_type_out,
    struct ArrowError* error) {
  NANOARROW_DCHECK(compression_type_out != NULL);
  static const enum ArrowIpcCompressionType types[] = {
      NANOARROW_IPC_COMPRESSION_TYPE_NONE, NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
      NANOARROW_IPC_COMPRESSION_TYPE_ZSTD};

  if (name != NULL) {
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
      if (strcmp(name, ArrowIpcCompressionTypeToString(types[i])) == 0) {
        *compression_type_out = types[i];
        return NANOARROW_OK;
      }
    }
  }

  ArrowErrorSet(error,
                "Unknown compression type name '%s' (expected 'none', 'lz4', or 'zstd')",
                name == NULL ? "" : name);
  return EINVAL;
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

  if (!ArrowIpcCompressionTypeIsCodec(compression_type)) {
    ArrowErrorSet(error, "Unknown decompression type with value %d",
                  (int)compression_type);
    return EINVAL;
  }

  ArrowIpcDecompressFunction fn = private_data->decompress_functions[compression_type];
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

  if (!ArrowIpcCompressionTypeIsCodec(compression_type)) {
    return EINVAL;
  }

  private_data->decompress_functions[compression_type] = decompress_function;
  return NANOARROW_OK;
}

struct ArrowIpcSerialCompressorPrivate {
  ArrowIpcCompressFunction compress_functions[3];
  int compression_level;
};

static ArrowErrorCode ArrowIpcSerialCompressorAdd(struct ArrowIpcCompressor* compressor,
                                                  struct ArrowBufferView src,
                                                  struct ArrowBuffer* dst,
                                                  struct ArrowError* error) {
  struct ArrowIpcSerialCompressorPrivate* private_data =
      (struct ArrowIpcSerialCompressorPrivate*)compressor->private_data;
  enum ArrowIpcCompressionType compression_type = compressor->compression_type;

  if (!ArrowIpcCompressionTypeIsCodec(compression_type)) {
    ArrowErrorSet(error, "Unknown compression type with value %d", (int)compression_type);
    return EINVAL;
  }

  ArrowIpcCompressFunction fn = private_data->compress_functions[compression_type];
  if (fn == NULL) {
    ArrowErrorSet(
        error, "Compression type with value %d not supported by this build of nanoarrow",
        (int)compression_type);
    return ENOTSUP;
  }

  // Compression happens synchronously, so there is never anything to wait for
  NANOARROW_RETURN_NOT_OK(fn(src, private_data->compression_level, dst, error));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcSerialCompressorWait(struct ArrowIpcCompressor* compressor,
                                                   int64_t timeout_ms,
                                                   struct ArrowError* error) {
  NANOARROW_UNUSED(compressor);
  NANOARROW_UNUSED(timeout_ms);
  NANOARROW_UNUSED(error);
  return NANOARROW_OK;
}

static void ArrowIpcSerialCompressorRelease(struct ArrowIpcCompressor* compressor) {
  ArrowFree(compressor->private_data);
  compressor->release = NULL;
}

ArrowErrorCode ArrowIpcSerialCompressor(struct ArrowIpcCompressor* compressor,
                                        enum ArrowIpcCompressionType compression_type,
                                        int compression_level) {
  compressor->release = NULL;
  if (compression_type != NANOARROW_IPC_COMPRESSION_TYPE_NONE &&
      !ArrowIpcCompressionTypeIsCodec(compression_type)) {
    return EINVAL;
  }

  struct ArrowIpcSerialCompressorPrivate* private_data =
      (struct ArrowIpcSerialCompressorPrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcSerialCompressorPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct ArrowIpcSerialCompressorPrivate));
  private_data->compression_level = compression_level;
  compressor->compression_type = compression_type;
  compressor->private_data = private_data;
  ArrowIpcSerialCompressorSetFunction(compressor, NANOARROW_IPC_COMPRESSION_TYPE_ZSTD,
                                      ArrowIpcGetZstdCompressionFunction());
  ArrowIpcSerialCompressorSetFunction(compressor,
                                      NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
                                      ArrowIpcGetLZ4CompressionFunction());
  compressor->compress_add = &ArrowIpcSerialCompressorAdd;
  compressor->compress_wait = &ArrowIpcSerialCompressorWait;
  compressor->release = &ArrowIpcSerialCompressorRelease;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcSerialCompressorSetFunction(
    struct ArrowIpcCompressor* compressor, enum ArrowIpcCompressionType compression_type,
    ArrowIpcCompressFunction compress_function) {
  struct ArrowIpcSerialCompressorPrivate* private_data =
      (struct ArrowIpcSerialCompressorPrivate*)compressor->private_data;

  if (!ArrowIpcCompressionTypeIsCodec(compression_type)) {
    return EINVAL;
  }

  private_data->compress_functions[compression_type] = compress_function;
  return NANOARROW_OK;
}
