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
#include <sqlite3.h>

#include "nanoarrow/nanoarrow.h"

#include "nanoarrow_sqlite3.h"

void ArrowSQLite3ResultInit(struct ArrowSQLite3Result* result, sqlite3_stmt* stmt) {
  result->stmt = stmt;
  result->array.release = NULL;
  result->schema.release = NULL;
  result->schema_explicit = 0;
}

int ArrowSQLite3ResultSetSchema(struct ArrowSQLite3Result* result,
                                struct ArrowSchema* schema) {
  if (schema == NULL || schema->release == NULL || result->schema.release != NULL) {
    return EINVAL;
  }

  if (schema->format == NULL || strcmp(schema->format, "+s") != 0) {
    return EINVAL;
  }

  memcpy(&result->schema, schema, sizeof(struct ArrowSchema));
  schema->release = NULL;
  return 0;
}

void ArrowSQLite3ResultReset(struct ArrowSQLite3Result* result) {
  result->stmt = NULL;

  if (result->array.release != NULL) {
    result->array.release(&result->array);
  }

  if (result->schema.release != NULL) {
    result->schema.release(&result->schema);
  }
}

void ArrowSQLite3ResultFinishSchema(struct ArrowSQLite3Result* result,
                                    struct ArrowSchema* schema_out) {
  memcpy(schema_out, &result->schema, sizeof(struct ArrowSchema));
  result->schema.release = NULL;
}

void ArrowSQLite3ResultFinishArray(struct ArrowSQLite3Result* result,
                                   struct ArrowArray* array_out) {
  memcpy(array_out, &result->array, sizeof(struct ArrowArray));
  result->array.release = NULL;
}

static int ArrowSQLite3ColumnSchema(const char* name, const char* declared_type,
                                    int first_value_type,
                                    struct ArrowSchema* schema_out) {
  int result;

  switch (first_value_type) {
    case SQLITE_INTEGER:
      result = ArrowSchemaInit(schema_out, NANOARROW_TYPE_INT64);
      break;
    case SQLITE_FLOAT:
      result = ArrowSchemaInit(schema_out, NANOARROW_TYPE_INT32);
      break;
    case SQLITE_BLOB:
      result = ArrowSchemaInit(schema_out, NANOARROW_TYPE_BINARY);
      break;
    default:
      result = ArrowSchemaInit(schema_out, NANOARROW_TYPE_STRING);
      break;
  }

  NANOARROW_RETURN_NOT_OK(result);

  // Add the column name
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema_out, name));

  // Add a nanoarrow_sqlite3.decltype metadata key so that consumers with access
  // to a higher-level runtime can transform values to a more appropriate column type
  struct ArrowBuffer buffer;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderInit(&buffer, NULL));
  result = ArrowMetadataBuilderAppend(&buffer, ArrowCharView("nanoarrow_sqlite3.decltype"),
                             ArrowCharView(declared_type));
  if (result != NANOARROW_OK) {
    ArrowBufferReset(&buffer);
    return result;
  }

  result = ArrowSchemaSetMetadata(schema_out, (const char*)buffer.data);
  ArrowBufferReset(&buffer);
  NANOARROW_RETURN_NOT_OK(result);

  return 0;
}

static int ArrowSQLite3GuessSchema(sqlite3_stmt* stmt, struct ArrowSchema* schema_out) {

}

struct ArrowSQLite3ErrorCode ArrowSQLite3ResultStep(struct ArrowSQLite3Result* result,
                                                    struct ArrowSQLite3Error* error) {
  struct ArrowSQLite3ErrorCode out;
  out.errno_code = 0;
  out.sqlite3_code = 0;

  // Run sqlite3_step()
  out.sqlite3_code = sqlite3_step(result->stmt);

  // If it was an actual error, return it here
  if (out.sqlite3_code != SQLITE_OK && out.sqlite3_code != SQLITE_DONE) {
    out.errno_code = EIO;
    return out;
  }

  // Make sure we have a schema
  if (result->schema.release == NULL) {
  }

  return out;
}