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

#include <stdio.h>

#include "nanoarrow/nanoarrow_ipc.h"

#include "library.h"

static struct ArrowError global_error;

const char* my_library_last_error(void) { return ArrowErrorMessage(&global_error); }

int verify_ipc_message(const void* data, int64_t size_bytes) {
  struct ArrowBufferView buffer_view;
  buffer_view.data.data = data;
  buffer_view.size_bytes = size_bytes;

  struct ArrowIpcDecoder decoder;
  ArrowIpcDecoderInit(&decoder);
  int result = ArrowIpcDecoderVerify(&decoder, buffer_view, &global_error);
  ArrowIpcDecoderReset(&decoder);

  return result;
}
