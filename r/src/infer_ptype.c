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

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"

#include "array.h"
#include "materialize.h"
#include "schema.h"
#include "util.h"

// These conversions are the default R-native type guesses for
// an array that don't require extra information from the ptype (e.g.,
// factor with levels). Some of these guesses may result in a conversion
// that later warns for out-of-range values (e.g., int64 to double());
// however, a user can use the convert_array(x, ptype = something_safer())
// when this occurs.
static enum VectorType nanoarrow_infer_vector_type(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_BOOL:
      return VECTOR_TYPE_LGL;

    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
      return VECTOR_TYPE_INT;

    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_DECIMAL32:
    case NANOARROW_TYPE_DECIMAL64:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
      return VECTOR_TYPE_DBL;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_STRING_VIEW:
      return VECTOR_TYPE_CHR;

    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_STRUCT:
      return VECTOR_TYPE_DATA_FRAME;

    default:
      return VECTOR_TYPE_OTHER;
  }
}

// The same as the above, but from a nanoarrow_schema()
static enum VectorType nanoarrow_infer_vector_type_schema(SEXP schema_xptr) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);

  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  if (ArrowSchemaViewInit(&schema_view, schema, &error) != NANOARROW_OK) {
    Rf_error("nanoarrow_infer_vector_type_schema(): %s", ArrowErrorMessage(&error));
  }

  if (schema_view.extension_name.size_bytes > 0) {
    return VECTOR_TYPE_OTHER;
  } else {
    return nanoarrow_infer_vector_type(schema_view.type);
  }
}

// The same as the above, but from a nanoarrow_array()
enum VectorType nanoarrow_infer_vector_type_array(SEXP array_xptr) {
  return nanoarrow_infer_vector_type_schema(array_xptr_get_schema(array_xptr));
}
