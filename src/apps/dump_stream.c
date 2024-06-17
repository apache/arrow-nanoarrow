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

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void dump_schema_to_stdout(struct ArrowSchema* schema, int level, char* buf,
                           int buf_size) {
  ArrowSchemaToString(schema, buf, buf_size, 0);

  for (int i = 0; i < level; i++) {
    fprintf(stdout, "  ");
  }

  if (schema->name == NULL) {
    fprintf(stdout, "%s\n", buf);
  } else {
    fprintf(stdout, "%s: %s\n", schema->name, buf);
  }

  for (int64_t i = 0; i < schema->n_children; i++) {
    dump_schema_to_stdout(schema->children[i], level + 1, buf, buf_size);
  }
}

int main(int argc, char* argv[]) {
  // Parse arguments
  if (argc != 2) {
    fprintf(stderr, "Usage: dump_stream FILENAME (or - for stdin)\n");
    return 1;
  }

  // Sort the input stream
  FILE* file_ptr;
  if (strcmp(argv[1], "-") == 0) {
    file_ptr = freopen(NULL, "rb", stdin);
  } else {
    file_ptr = fopen(argv[1], "rb");
  }

  if (file_ptr == NULL) {
    fprintf(stderr, "Failed to open input '%s'\n", argv[1]);
    return 1;
  }

  struct ArrowIpcInputStream input;
  int result = ArrowIpcInputStreamInitFile(&input, file_ptr, 0);
  if (result != NANOARROW_OK) {
    fprintf(stderr, "ArrowIpcInputStreamInitFile() failed\n");
    return 1;
  }

  struct ArrowArrayStream stream;
  result = ArrowIpcArrayStreamReaderInit(&stream, &input, NULL);
  if (result != NANOARROW_OK) {
    fprintf(stderr, "ArrowIpcArrayStreamReaderInit() failed\n");
    return 1;
  }

  clock_t begin = clock();

  struct ArrowSchema schema;
  result = ArrowArrayStreamGetSchema(&stream, &schema, NULL);
  if (result != NANOARROW_OK) {
    fprintf(stderr, "stream.get_schema() returned %d with error '%s'\n", result,
            ArrowArrayStreamGetLastError(&stream));
    ArrowArrayStreamRelease(&stream);
    return 1;
  }

  clock_t end = clock();
  double elapsed = (end - begin) / ((double)CLOCKS_PER_SEC);
  fprintf(stdout, "Read Schema <%.06f seconds>\n", elapsed);

  char schema_tmp[8096];
  memset(schema_tmp, 0, sizeof(schema_tmp));
  dump_schema_to_stdout(&schema, 0, schema_tmp, sizeof(schema_tmp));
  ArrowSchemaRelease(&schema);

  struct ArrowArray array;
  array.release = NULL;

  int64_t batch_count = 0;
  int64_t row_count = 0;
  begin = clock();

  while (1) {
    result = ArrowArrayStreamGetNext(&stream, &array, NULL);
    if (result != NANOARROW_OK) {
      fprintf(stderr, "stream.get_next() returned %d with error '%s'\n", result,
              ArrowArrayStreamGetLastError(&stream));
      ArrowArrayStreamRelease(&stream);
      return 1;
    }

    if (array.release != NULL) {
      row_count += array.length;
      batch_count++;
      ArrowArrayRelease(&array);
    } else {
      break;
    }
  }

  end = clock();
  elapsed = (end - begin) / ((double)CLOCKS_PER_SEC);
  fprintf(stdout, "Read %" PRId64 " rows in %" PRId64 " batch(es) <%.06f seconds>\n",
          row_count, batch_count, elapsed);

  ArrowArrayStreamRelease(&stream);
  fclose(file_ptr);
  return 0;
}
