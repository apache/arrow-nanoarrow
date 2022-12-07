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
#include "schema.h"
#include "util.h"

void finalize_schema_xptr(SEXP schema_xptr) {
  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);
  if (schema != NULL && schema->release != NULL) {
    schema->release(schema);
  }

  if (schema != NULL) {
    ArrowFree(schema);
  }
}

static SEXP schema_metadata_to_list(const char* metadata) {
  if (metadata == NULL) {
    return R_NilValue;
  }

  struct ArrowMetadataReader reader;
  ArrowMetadataReaderInit(&reader, metadata);
  SEXP names = PROTECT(Rf_allocVector(STRSXP, reader.remaining_keys));
  SEXP values = PROTECT(Rf_allocVector(VECSXP, reader.remaining_keys));

  struct ArrowStringView key;
  struct ArrowStringView value;
  R_xlen_t i = 0;
  while (reader.remaining_keys > 0) {
    ArrowMetadataReaderRead(&reader, &key, &value);
    SET_STRING_ELT(names, i, Rf_mkCharLenCE(key.data, key.n_bytes, CE_UTF8));
    SEXP value_raw = PROTECT(Rf_allocVector(RAWSXP, value.n_bytes));
    memcpy(RAW(value_raw), value.data, value.n_bytes);
    SET_VECTOR_ELT(values, i, value_raw);
    UNPROTECT(1);
    i++;
  }

  Rf_setAttrib(values, R_NamesSymbol, names);
  UNPROTECT(2);
  return values;
}

static SEXP borrow_schema_xptr(struct ArrowSchema* schema, SEXP shelter) {
  SEXP schema_xptr = PROTECT(R_MakeExternalPtr(schema, R_NilValue, shelter));
  Rf_setAttrib(schema_xptr, R_ClassSymbol, nanoarrow_cls_schema);
  UNPROTECT(1);
  return schema_xptr;
}

SEXP borrow_schema_child_xptr(SEXP schema_xptr, int64_t i) {
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);
  return borrow_schema_xptr(schema->children[i], schema_xptr);
}

SEXP nanoarrow_c_schema_to_list(SEXP schema_xptr) {
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);

  const char* names[] = {"format",   "name",       "metadata", "flags",
                         "children", "dictionary", ""};
  SEXP result = PROTECT(Rf_mkNamed(VECSXP, names));

  SEXP format_sexp = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(format_sexp, 0, Rf_mkCharCE(schema->format, CE_UTF8));
  SET_VECTOR_ELT(result, 0, format_sexp);
  UNPROTECT(1);

  if (schema->name != NULL) {
    SEXP name_sexp = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(name_sexp, 0, Rf_mkCharCE(schema->name, CE_UTF8));
    SET_VECTOR_ELT(result, 1, name_sexp);
    UNPROTECT(1);
  } else {
    SET_VECTOR_ELT(result, 1, R_NilValue);
  }

  SET_VECTOR_ELT(result, 2, schema_metadata_to_list(schema->metadata));
  SET_VECTOR_ELT(result, 3, Rf_ScalarInteger(schema->flags));

  if (schema->n_children > 0) {
    SEXP children_sexp = PROTECT(Rf_allocVector(VECSXP, schema->n_children));
    SEXP children_names_sexp = PROTECT(Rf_allocVector(STRSXP, schema->n_children));
    for (R_xlen_t i = 0; i < schema->n_children; i++) {
      SEXP child_xptr = PROTECT(borrow_schema_xptr(schema->children[i], schema_xptr));
      SET_VECTOR_ELT(children_sexp, i, child_xptr);
      if (schema->children[i]->name != NULL) {
        SET_STRING_ELT(children_names_sexp, i,
                       Rf_mkCharCE(schema->children[i]->name, CE_UTF8));
      } else {
        SET_STRING_ELT(children_names_sexp, i, Rf_mkCharCE("", CE_UTF8));
      }
      UNPROTECT(1);
    }
    Rf_setAttrib(children_sexp, R_NamesSymbol, children_names_sexp);
    SET_VECTOR_ELT(result, 4, children_sexp);
    UNPROTECT(2);
  } else {
    SET_VECTOR_ELT(result, 4, R_NilValue);
  }

  if (schema->dictionary != NULL) {
    SEXP dictionary_xptr = PROTECT(borrow_schema_xptr(schema->dictionary, schema_xptr));
    SET_VECTOR_ELT(result, 5, dictionary_xptr);
    UNPROTECT(1);
  } else {
    SET_VECTOR_ELT(result, 5, R_NilValue);
  }

  UNPROTECT(1);
  return result;
}

static SEXP mkStringView(struct ArrowStringView* view) {
  if (view->data == NULL) {
    return R_NilValue;
  }

  SEXP chr = PROTECT(Rf_mkCharLenCE(view->data, view->n_bytes, CE_UTF8));
  SEXP str = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(str, 0, chr);
  UNPROTECT(2);
  return str;
}

SEXP nanoarrow_c_schema_parse(SEXP schema_xptr) {
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);

  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  int status = ArrowSchemaViewInit(&schema_view, schema, &error);
  if (status != NANOARROW_OK) {
    Rf_error("ArrowSchemaViewInit(): %s", ArrowErrorMessage(&error));
  }

  const char* names[] = {
      "type",       "storage_type",     "extension_name",    "extension_metadata",
      "fixed_size", "decimal_bitwidth", "decimal_precision", "decimal_scale",
      "time_unit",  "timezone",         "union_type_ids",    ""};

  SEXP result = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(result, 0, Rf_mkString(ArrowTypeString((schema_view.data_type))));
  SET_VECTOR_ELT(result, 1,
                 Rf_mkString(ArrowTypeString((schema_view.storage_data_type))));

  if (schema_view.extension_name.data != NULL) {
    SET_VECTOR_ELT(result, 2, mkStringView(&schema_view.extension_name));
  }

  if (schema_view.extension_metadata.data != NULL) {
    SEXP metadata_sexp =
        PROTECT(Rf_allocVector(RAWSXP, schema_view.extension_metadata.n_bytes));
    memcpy(RAW(metadata_sexp), schema_view.extension_metadata.data,
           schema_view.extension_metadata.n_bytes);
    SET_VECTOR_ELT(result, 3, metadata_sexp);
    UNPROTECT(1);
  }

  if (schema_view.data_type == NANOARROW_TYPE_FIXED_SIZE_LIST ||
      schema_view.data_type == NANOARROW_TYPE_FIXED_SIZE_BINARY) {
    SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(schema_view.fixed_size));
  }

  if (schema_view.data_type == NANOARROW_TYPE_DECIMAL128 ||
      schema_view.data_type == NANOARROW_TYPE_DECIMAL256) {
    SET_VECTOR_ELT(result, 5, Rf_ScalarInteger(schema_view.decimal_bitwidth));
    SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(schema_view.decimal_precision));
    SET_VECTOR_ELT(result, 7, Rf_ScalarInteger(schema_view.decimal_scale));
  }

  if (schema_view.data_type == NANOARROW_TYPE_TIME32 ||
      schema_view.data_type == NANOARROW_TYPE_TIME64 ||
      schema_view.data_type == NANOARROW_TYPE_TIMESTAMP ||
      schema_view.data_type == NANOARROW_TYPE_DURATION) {
    SET_VECTOR_ELT(result, 8, Rf_mkString(ArrowTimeUnitString((schema_view.time_unit))));
  }

  if (schema_view.data_type == NANOARROW_TYPE_TIMESTAMP) {
    SET_VECTOR_ELT(result, 9, mkStringView(&schema_view.timezone));
  }

  if (schema_view.data_type == NANOARROW_TYPE_DENSE_UNION ||
      schema_view.data_type == NANOARROW_TYPE_SPARSE_UNION) {
    int8_t type_ids[128];
    int num_type_ids = _ArrowParseUnionTypeIds(schema_view.union_type_ids, type_ids);
    if (num_type_ids == -1) {
      Rf_error("Invalid type IDs in union type: '%s'", schema_view.union_type_ids);
    }

    SEXP union_type_ids = PROTECT(Rf_allocVector(INTSXP, num_type_ids));
    for (int i = 0; i < num_type_ids; i++) {
      INTEGER(union_type_ids)[i] = type_ids[i];
    }
    SET_VECTOR_ELT(result, 10, union_type_ids);
    UNPROTECT(1);
  }

  UNPROTECT(1);
  return result;
}

SEXP nanoarrow_c_schema_format(SEXP schema_xptr, SEXP recursive_sexp) {
  int recursive = LOGICAL(recursive_sexp)[0];

  // Be extra safe here (errors during formatting are hard to work around)
  if (!Rf_inherits(schema_xptr, "nanoarrow_schema")) {
    return Rf_mkString("[invalid: schema is not a nanoarrow_schema]");
  }

  if (TYPEOF(schema_xptr) != EXTPTRSXP) {
    return Rf_mkString("[invalid: schema is not an external pointer]");
  }

  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);

  int64_t size_needed = ArrowSchemaToString(schema, NULL, 0, recursive);
  // Using an SEXP because Rf_mkCharLenCE could jump
  SEXP formatted_sexp = PROTECT(Rf_allocVector(RAWSXP, size_needed + 1));
  ArrowSchemaToString(schema, (char*)RAW(formatted_sexp), size_needed + 1, recursive);
  SEXP result_sexp = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(result_sexp, 0,
                 Rf_mkCharLenCE((char*)RAW(formatted_sexp), size_needed, CE_UTF8));
  UNPROTECT(2);
  return result_sexp;
}
