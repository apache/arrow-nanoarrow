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

#include <stdio.h>

int main(int argc, char* argv[]) {
  struct ArrowArray array;
  struct ArrowSchema schema;

  int result = my_library_int32_array_from_args(argc - 1, argv + 1, &array, &schema);
  if (result != 0) {
    printf("%s\n", my_library_last_error());
    return result;
  }

  printf("Parsed array with length %ld\n", (long)array.length);

  array.release(&array);
  schema.release(&schema);
  return 0;
}
