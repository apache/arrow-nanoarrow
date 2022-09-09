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
#include <string.h>

#include "nanoarrow/nanoarrow.h"

#include "nanoarrow_sqlite3.h"

struct ArrowSQLite3ResultPrivate {
  struct ArrowError error;
  struct ArrowBuffer sqlite_column_types;
};

int ArrowSQLite3ResultInit(struct ArrowSQLite3Result* result) {
  result->step_return_code = SQLITE_OK;
  result->array.release = NULL;
  result->schema.release = NULL;

  result->private_data = ArrowMalloc(sizeof(struct ArrowSQLite3ResultPrivate));
  if (result->private_data == NULL) {
    return ENOMEM;
  }

  struct ArrowSQLite3ResultPrivate* private_data =
      (struct ArrowSQLite3ResultPrivate*)result->private_data;
  private_data->error.message[0] = '\0';
  ArrowBufferInit(&private_data->sqlite_column_types);

  return 0;
}

void ArrowSQLite3ResultReset(struct ArrowSQLite3Result* result) {
  if (result->array.release != NULL) {
    result->array.release(&result->array);
  }

  if (result->schema.release != NULL) {
    result->schema.release(&result->schema);
  }

  if (result->private_data != NULL) {
    struct ArrowSQLite3ResultPrivate* private_data =
        (struct ArrowSQLite3ResultPrivate*)result->private_data;
    ArrowBufferReset(&private_data->sqlite_column_types);
    ArrowFree(private_data);
  }
}

const char* ArrowSQLite3ResultError(struct ArrowSQLite3Result* result) {
  struct ArrowSQLite3ResultPrivate* private_data =
      (struct ArrowSQLite3ResultPrivate*)result->private_data;
  return private_data->error.message;
}

static int ArrowSQLite3TypeFromArrowType(enum ArrowType storage_type, int* sqlite_type) {
  switch (storage_type) {
    case NANOARROW_TYPE_NA:
      *sqlite_type = SQLITE_NULL;
      return 0;

    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
      *sqlite_type = SQLITE_INTEGER;
      return 0;

    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
      *sqlite_type = SQLITE_FLOAT;
      return 0;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      *sqlite_type = SQLITE_TEXT;
      return 0;

    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      *sqlite_type = SQLITE_BLOB;
      return 0;

    default:
      return EINVAL;
  }
}

static int ArrowSQlite3ResolveSQLiteTypes(struct ArrowSQLite3Result* result,
                                          struct ArrowSchema* schema) {
  struct ArrowSQLite3ResultPrivate* private =
      (struct ArrowSQLite3ResultPrivate*)result->private_data;

  ArrowBufferReset(&private->sqlite_column_types);
  struct ArrowSchemaView schema_view;
  int sqlite_type;
  for (int64_t i = 0; i < schema->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaViewInit(&schema_view, schema->children[i], &private->error));
    if (ArrowSQLite3TypeFromArrowType(schema_view.storage_data_type, &sqlite_type) !=
        NANOARROW_OK) {
      ArrowErrorSet(&private->error,
                    "Arrow type for column %d is not supported by SQLite3", (int)i);
      return EINVAL;
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppendInt32(&private->sqlite_column_types, sqlite_type));
  }

  return 0;
}

int ArrowSQLite3ResultSetSchema(struct ArrowSQLite3Result* result,
                                struct ArrowSchema* schema) {
  struct ArrowSQLite3ResultPrivate* private =
      (struct ArrowSQLite3ResultPrivate*)result->private_data;

  if (schema == NULL || schema->release == NULL || result->schema.release != NULL) {
    ArrowErrorSet(&private->error, "schema is null or released");
    return EINVAL;
  }

  if (schema->format == NULL || strcmp(schema->format, "+s") != 0) {
    ArrowErrorSet(&private->error, "schema is not a struct");
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(ArrowSQlite3ResolveSQLiteTypes(result, schema));

  memcpy(&result->schema, schema, sizeof(struct ArrowSchema));
  schema->release = NULL;

  return 0;
}

int ArrowSQLite3ResultFinishSchema(struct ArrowSQLite3Result* result,
                                   struct ArrowSchema* schema_out) {
  if (result->array.release == NULL) {
    return EINVAL;
  }

  memcpy(schema_out, &result->schema, sizeof(struct ArrowSchema));
  result->schema.release = NULL;

  return 0;
}

int ArrowSQLite3ResultFinishArray(struct ArrowSQLite3Result* result,
                                  struct ArrowArray* array_out) {
  if (result->array.release == NULL) {
    return EINVAL;
  }

  struct ArrowSQLite3ResultPrivate* private =
      (struct ArrowSQLite3ResultPrivate*)result->private_data;
  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuilding(&result->array, &private->error));

  memcpy(array_out, &result->array, sizeof(struct ArrowArray));
  result->array.release = NULL;
  return 0;
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
  result = ArrowMetadataBuilderAppend(
      &buffer, ArrowCharView("nanoarrow_sqlite3.decltype"), ArrowCharView(declared_type));
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
  NANOARROW_RETURN_NOT_OK(ArrowSchemaInit(schema_out, NANOARROW_TYPE_STRUCT));

  int n_col = sqlite3_column_count(stmt);
  NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema_out, n_col));

  for (int i = 0; i < n_col; i++) {
    const char* name = sqlite3_column_name16(stmt, i);
    const char* declared_type = sqlite3_column_decltype(stmt, i);
    int first_value_type = sqlite3_column_type(stmt, i);
    NANOARROW_RETURN_NOT_OK(ArrowSQLite3ColumnSchema(
        name, declared_type, first_value_type, schema_out->children[i]));
  }

  return 0;
}

int ArrowSQLite3ResultStep(struct ArrowSQLite3Result* result, sqlite3_stmt* stmt) {
  struct ArrowSQLite3ResultPrivate* private_data =
      (struct ArrowSQLite3ResultPrivate*)result->private_data;
  private_data->error.message[0] = '\0';

  // Call sqlite3_step()
  result->step_return_code = sqlite3_step(stmt);
  if (result->step_return_code != SQLITE_ROW && result->step_return_code != SQLITE_DONE) {
    return EIO;
  }

  // Make sure we have a schema
  if (result->schema.release == NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowSQLite3GuessSchema(stmt, &result->schema));
    NANOARROW_RETURN_NOT_OK(ArrowSQlite3ResolveSQLiteTypes(result, &result->schema));
  }

  // Make sure we have an array
  if (result->array.release == NULL) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayInitFromSchema(&result->array, &result->schema, &private_data->error));
    NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(&result->array));
  }

  // Check the schema
  int n_col = sqlite3_column_count(stmt);
  if (n_col != result->schema.n_children) {
    ArrowErrorSet(&private_data->error,
                  "Expected result with %d column(s) but got result with %d column(s)",
                  (int)result->schema.n_children, (int)n_col);
    return EINVAL;
  }

  // If there are no rows left, we're done!
  if (result->step_return_code == SQLITE_DONE) {
    return 0;
  }

  // Loop through columns and append to each child array
  // Instead of using sqlite3_column_type(), we use the type most appropriate
  // for the column type specified by schema so we can leverage sqlite3's facilities
  // for type conversion.
  int* sqlite_types = (int*)private_data->sqlite_column_types.data;
  struct ArrowStringView string_view;
  struct ArrowBufferView buffer_view;
  int result_code;

  for (int i = 0; i < n_col; i++) {
    switch (sqlite_types[i]) {
      case SQLITE_NULL:
        result_code = ArrowArrayAppendNull(result->array.children[i], 1);
        break;

      case SQLITE_INTEGER:
        result_code =
            ArrowArrayAppendInt(result->array.children[i], sqlite3_column_int64(stmt, i));
        break;

      case SQLITE_FLOAT:
        result_code = ArrowArrayAppendDouble(result->array.children[i],
                                             sqlite3_column_double(stmt, i));
        break;

      case SQLITE_BLOB:
        buffer_view.n_bytes = sqlite3_column_bytes(stmt, i);
        buffer_view.data.data = sqlite3_column_blob(stmt, i);
        result_code = ArrowArrayAppendBytes(result->array.children[i], buffer_view);
        break;

      case SQLITE_TEXT:
        string_view.n_bytes = sqlite3_column_bytes(stmt, i);
        string_view.data = sqlite3_column_text16(stmt, i);
        result_code = ArrowArrayAppendString(result->array.children[i], string_view);
        break;

      default:
        result_code = EIO;
        break;
    }

    if (result_code != NANOARROW_OK) {
      // Attempt to leave the parent array in a consistent state with equal-length
      // columns even if there was an error appending the value
      ArrowArrayAppendNull(result->array.children[i], 1);
      return result_code;
    }
  }

  return ArrowArrayFinishElement(&result->array);
}
