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

#include "library.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
  int64_t in_size = 0;
  uint8_t data[8096];

  freopen(NULL, "rb", stdin);
  in_size = fread(data, 1, 8096, stdin);

  if (in_size == 8096) {
    fprintf(stderr, "This example can't read messages more than 8096 bytes\n");
    return -1;
  }

  int result = verify_ipc_message(data, in_size);
  if (result != 0) {
    fprintf(stderr, "%s\n", my_library_last_error());
  }

  return result;
}
