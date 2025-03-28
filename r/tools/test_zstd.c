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

#include <stdint.h>
#include <string.h>
#include <zstd.h>

// Function that requires at least one symbol from zstd.h
size_t test_zstd(void) {
  uint8_t src[128];
  memset(src, 0, sizeof(src));
  uint8_t dst[128];
  return ZSTD_compress(dst, sizeof(dst), src, sizeof(src), ZSTD_CLEVEL_DEFAULT);
}
