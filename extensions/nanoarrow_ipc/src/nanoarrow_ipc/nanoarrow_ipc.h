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

#ifndef NANOARROW_NANOARROW_IPC_H_INCLUDED
#define NANOARROW_NANOARROW_IPC_H_INCLUDED

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Extra guard for versions of Arrow without the canonical guard
#ifndef ARROW_FLAG_DICTIONARY_ORDERED

/// \defgroup nanoarrow-arrow-cdata
///
/// The Arrow C Data (https://arrow.apache.org/docs/format/CDataInterface.html)
/// and Arrow C Stream (https://arrow.apache.org/docs/format/CStreamInterface.html)
/// interfaces are part of the
/// Arrow Columnar Format specification
/// (https://arrow.apache.org/docs/format/Columnar.html). See the Arrow documentation for
/// documentation of these structures.
///
/// @{

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  // Array type description
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;

  // Release callback
  void (*release)(struct ArrowSchema*);
  // Opaque producer-specific data
  void* private_data;
};

struct ArrowArray {
  // Array data description
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;

  // Release callback
  void (*release)(struct ArrowArray*);
  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
  // Callback to get the stream type
  // (will be the same for all arrays in the stream).
  //
  // Return value: 0 if successful, an `errno`-compatible error code otherwise.
  //
  // If successful, the ArrowSchema must be released independently from the stream.
  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);

  // Callback to get the next array
  // (if no error and the array is released, the stream has ended)
  //
  // Return value: 0 if successful, an `errno`-compatible error code otherwise.
  //
  // If successful, the ArrowArray must be released independently from the stream.
  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);

  // Callback to get optional detailed error information.
  // This must only be called if the last stream operation failed
  // with a non-0 return code.
  //
  // Return value: pointer to a null-terminated character array describing
  // the last error, or NULL if no description is available.
  //
  // The returned pointer is only valid until the next operation on this stream
  // (including release).
  const char* (*get_last_error)(struct ArrowArrayStream*);

  // Release callback: release the stream's own resources.
  // Note that arrays returned by `get_next` must be individually released.
  void (*release)(struct ArrowArrayStream*);

  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE
#endif  // ARROW_FLAG_DICTIONARY_ORDERED

#ifdef __cplusplus
}
#endif

typedef int ArrowIpcErrorCode;

struct ArrowIpcError {
  char message[1024];
};

struct ArrowIpcIO {
  ArrowIpcErrorCode (*read)(struct ArrowIpcIO* io, uint8_t* dst, int64_t dst_size,
                            int64_t* size_read_out);
  ArrowIpcErrorCode (*write)(struct ArrowIpcIO* io, const uint8_t* src, int64_t src_size);
  ArrowIpcErrorCode (*size)(struct ArrowIpcIO* io, int64_t* size_out);
  ArrowIpcErrorCode (*seek)(struct ArrowIpcIO* io, int64_t position);
  const char* (*get_last_error)(struct ArrowIpcIO* io);
  void (*release)(struct ArrowIpcIO* io);
  void* private_data;
};

ArrowIpcErrorCode ArrowIpcInitBufferViewReader(struct ArrowIpcIO* io, const void* data,
                                               int64_t size_bytes);

ArrowIpcErrorCode ArrowIpcInitStreamReader(struct ArrowArrayStream* stream_out,
                                           struct ArrowIpcIO* io,
                                           struct ArrowIpcError* error);

ArrowIpcErrorCode ArrowIpcWriteSchema(struct ArrowArrayStream* stream_in,
                                      struct ArrowIpcIO* io, struct ArrowIpcError* error);

ArrowIpcErrorCode ArrowIpcWriteBatches(struct ArrowArrayStream* stream_in,
                                       struct ArrowIpcIO* io, int64_t num_batches,
                                       struct ArrowIpcError* error);

ArrowIpcErrorCode ArrowIpcWriteEndOfStream(struct ArrowIpcIO* io,
                                           struct ArrowIpcError* error);

#endif
