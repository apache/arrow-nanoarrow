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

SEXP nanoarrow_c_schema_init(SEXP type_id_sexp, SEXP nullable_sexp) {
  int type_id = INTEGER(type_id_sexp)[0];
  SEXP schema_xptr = PROTECT(nanoarrow_schema_owning_xptr());
  struct ArrowSchema* schema = nanoarrow_output_schema_from_xptr(schema_xptr);

  int result = ArrowSchemaInitFromType(schema, type_id);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaInitFromType() failed");
  }

  result = ArrowSchemaSetName(schema, "");
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaSetName() failed");
  }

  if (!LOGICAL(nullable_sexp)[0]) {
    schema->flags &= ~ARROW_FLAG_NULLABLE;
  }

  UNPROTECT(1);
  return schema_xptr;
}

SEXP nanoarrow_c_schema_init_date_time(SEXP type_id_sexp, SEXP time_unit_sexp,
                                       SEXP timezone_sexp, SEXP nullable_sexp) {
  int type_id = INTEGER(type_id_sexp)[0];
  int time_unit = INTEGER(time_unit_sexp)[0];

  const char* timezone = NULL;
  if (timezone_sexp != R_NilValue) {
    timezone = Rf_translateCharUTF8(STRING_ELT(timezone_sexp, 0));
  } else {
    timezone = NULL;
  }

  SEXP schema_xptr = PROTECT(nanoarrow_schema_owning_xptr());
  struct ArrowSchema* schema = nanoarrow_output_schema_from_xptr(schema_xptr);

  ArrowSchemaInit(schema);
  int result = ArrowSchemaSetTypeDateTime(schema, type_id, time_unit, timezone);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaSetTypeDateTime() failed");
  }

  result = ArrowSchemaSetName(schema, "");
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaSetName() failed");
  }

  if (!LOGICAL(nullable_sexp)[0]) {
    schema->flags &= ~ARROW_FLAG_NULLABLE;
  }

  UNPROTECT(1);
  return schema_xptr;
}

SEXP nanoarrow_c_schema_init_decimal(SEXP type_id_sexp, SEXP precision_sexp,
                                     SEXP scale_sexp, SEXP nullable_sexp) {
  int type_id = INTEGER(type_id_sexp)[0];
  int precision = INTEGER(precision_sexp)[0];
  int scale = INTEGER(scale_sexp)[0];

  SEXP schema_xptr = PROTECT(nanoarrow_schema_owning_xptr());
  struct ArrowSchema* schema = nanoarrow_output_schema_from_xptr(schema_xptr);

  ArrowSchemaInit(schema);
  int result = ArrowSchemaSetTypeDecimal(schema, type_id, precision, scale);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaSetTypeDecimal() failed");
  }

  result = ArrowSchemaSetName(schema, "");
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaSetName() failed");
  }

  if (!LOGICAL(nullable_sexp)[0]) {
    schema->flags &= ~ARROW_FLAG_NULLABLE;
  }

  UNPROTECT(1);
  return schema_xptr;
}

SEXP nanoarrow_c_schema_init_fixed_size(SEXP type_id_sexp, SEXP fixed_size_sexp,
                                        SEXP nullable_sexp) {
  int type_id = INTEGER(type_id_sexp)[0];
  int fixed_size = INTEGER(fixed_size_sexp)[0];

  SEXP schema_xptr = PROTECT(nanoarrow_schema_owning_xptr());
  struct ArrowSchema* schema = nanoarrow_output_schema_from_xptr(schema_xptr);

  ArrowSchemaInit(schema);
  int result = ArrowSchemaSetTypeFixedSize(schema, type_id, fixed_size);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaSetTypeFixedSize() failed");
  }

  result = ArrowSchemaSetName(schema, "");
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaSetName() failed");
  }

  if (!LOGICAL(nullable_sexp)[0]) {
    schema->flags &= ~ARROW_FLAG_NULLABLE;
  }

  UNPROTECT(1);
  return schema_xptr;
}

static SEXP schema_metadata_to_list(const char* metadata) {
  if (metadata == NULL) {
    return R_NilValue;
  }

  struct ArrowMetadataReader reader;
  int result = ArrowMetadataReaderInit(&reader, metadata);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowMetadataReaderInit() failed");
  }

  SEXP names = PROTECT(Rf_allocVector(STRSXP, reader.remaining_keys));
  SEXP values = PROTECT(Rf_allocVector(VECSXP, reader.remaining_keys));

  struct ArrowStringView key;
  struct ArrowStringView value;
  R_xlen_t i = 0;
  while (reader.remaining_keys > 0) {
    result = ArrowMetadataReaderRead(&reader, &key, &value);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowMetadataReaderRead() failed");
    }

    SET_STRING_ELT(names, i, Rf_mkCharLenCE(key.data, (int)key.size_bytes, CE_UTF8));
    SEXP value_raw = PROTECT(Rf_allocVector(RAWSXP, value.size_bytes));
    memcpy(RAW(value_raw), value.data, value.size_bytes);
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
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);
  return borrow_schema_xptr(schema->children[i], schema_xptr);
}

SEXP nanoarrow_c_schema_to_list(SEXP schema_xptr) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);

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
  SET_VECTOR_ELT(result, 3, Rf_ScalarInteger((int)schema->flags));

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
    SET_VECTOR_ELT(result, 4, Rf_allocVector(VECSXP, schema->n_children));
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

  SEXP chr = PROTECT(Rf_mkCharLenCE(view->data, (int)view->size_bytes, CE_UTF8));
  SEXP str = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(str, 0, chr);
  UNPROTECT(2);
  return str;
}

SEXP nanoarrow_c_schema_parse(SEXP schema_xptr) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);

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
  SET_VECTOR_ELT(result, 0, Rf_mkString(ArrowTypeString((schema_view.type))));
  SET_VECTOR_ELT(result, 1, Rf_mkString(ArrowTypeString((schema_view.storage_type))));

  if (schema_view.extension_name.data != NULL) {
    SET_VECTOR_ELT(result, 2, mkStringView(&schema_view.extension_name));
  }

  if (schema_view.extension_metadata.data != NULL) {
    SEXP metadata_sexp =
        PROTECT(Rf_allocVector(RAWSXP, schema_view.extension_metadata.size_bytes));
    memcpy(RAW(metadata_sexp), schema_view.extension_metadata.data,
           schema_view.extension_metadata.size_bytes);
    SET_VECTOR_ELT(result, 3, metadata_sexp);
    UNPROTECT(1);
  }

  if (schema_view.type == NANOARROW_TYPE_FIXED_SIZE_LIST ||
      schema_view.type == NANOARROW_TYPE_FIXED_SIZE_BINARY) {
    SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(schema_view.fixed_size));
  }

  if (schema_view.type == NANOARROW_TYPE_DECIMAL32 ||
      schema_view.type == NANOARROW_TYPE_DECIMAL64 ||
      schema_view.type == NANOARROW_TYPE_DECIMAL128 ||
      schema_view.type == NANOARROW_TYPE_DECIMAL256) {
    SET_VECTOR_ELT(result, 5, Rf_ScalarInteger(schema_view.decimal_bitwidth));
    SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(schema_view.decimal_precision));
    SET_VECTOR_ELT(result, 7, Rf_ScalarInteger(schema_view.decimal_scale));
  }

  if (schema_view.type == NANOARROW_TYPE_TIME32 ||
      schema_view.type == NANOARROW_TYPE_TIME64 ||
      schema_view.type == NANOARROW_TYPE_TIMESTAMP ||
      schema_view.type == NANOARROW_TYPE_DURATION) {
    SET_VECTOR_ELT(result, 8, Rf_mkString(ArrowTimeUnitString((schema_view.time_unit))));
  }

  if (schema_view.type == NANOARROW_TYPE_TIMESTAMP) {
    SET_VECTOR_ELT(result, 9, Rf_mkString(schema_view.timezone));
  }

  if (schema_view.type == NANOARROW_TYPE_DENSE_UNION ||
      schema_view.type == NANOARROW_TYPE_SPARSE_UNION) {
    int8_t type_ids[128];
    int num_type_ids = _ArrowParseUnionTypeIds(schema_view.union_type_ids, type_ids);
    if (num_type_ids == -1 || num_type_ids > 127) {
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

  int64_t size_needed = ArrowSchemaToString(schema, NULL, 0, recursive != 0);
  if (size_needed >= INT_MAX) {
    size_needed = INT_MAX - 1;
  }

  // Using an SEXP because Rf_mkCharLenCE could jump
  SEXP formatted_sexp = PROTECT(Rf_allocVector(RAWSXP, size_needed + 1));
  ArrowSchemaToString(schema, (char*)RAW(formatted_sexp), size_needed + 1,
                      recursive != 0);
  SEXP result_sexp = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(result_sexp, 0,
                 Rf_mkCharLenCE((char*)RAW(formatted_sexp), (int)size_needed, CE_UTF8));
  UNPROTECT(2);
  return result_sexp;
}

SEXP nanoarrow_c_schema_set_format(SEXP schema_mut_xptr, SEXP format_sexp) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_mut_xptr);

  if (TYPEOF(format_sexp) != STRSXP || Rf_length(format_sexp) != 1) {
    Rf_error("schema$format must be character(1)");
  }

  const char* format = Rf_translateCharUTF8(STRING_ELT(format_sexp, 0));
  if (ArrowSchemaSetFormat(schema, format) != NANOARROW_OK) {
    Rf_error("Error setting schema$format");
  }

  return R_NilValue;
}

SEXP nanoarrow_c_schema_set_name(SEXP schema_mut_xptr, SEXP name_sexp) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_mut_xptr);
  int result;

  if (name_sexp == R_NilValue) {
    result = ArrowSchemaSetName(schema, NULL);
  } else {
    if (TYPEOF(name_sexp) != STRSXP || Rf_length(name_sexp) != 1) {
      Rf_error("schema$name must be NULL or character(1)");
    }

    const char* name = Rf_translateCharUTF8(STRING_ELT(name_sexp, 0));
    result = ArrowSchemaSetName(schema, name);
  }

  if (result != NANOARROW_OK) {
    Rf_error("Error setting schema$name");
  }

  return R_NilValue;
}

static void finalize_buffer_xptr(SEXP buffer_xptr) {
  struct ArrowBuffer* buffer = (struct ArrowBuffer*)R_ExternalPtrAddr(buffer_xptr);
  if (buffer != NULL) {
    ArrowBufferReset(buffer);
    ArrowFree(buffer);
  }
}

static SEXP buffer_owning_xptr(void) {
  struct ArrowBuffer* buffer =
      (struct ArrowBuffer*)ArrowMalloc(sizeof(struct ArrowBuffer));
  if (buffer == NULL) {
    Rf_error("Failed to allocate ArrowBuffer");
  }

  SEXP buffer_xptr = PROTECT(R_MakeExternalPtr(buffer, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(buffer_xptr, &finalize_buffer_xptr);
  UNPROTECT(1);
  return buffer_xptr;
}

SEXP nanoarrow_c_schema_set_metadata(SEXP schema_mut_xptr, SEXP metadata_sexp) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_mut_xptr);
  int result;

  if (Rf_xlength(metadata_sexp) == 0) {
    result = ArrowSchemaSetMetadata(schema, NULL);
    if (result != NANOARROW_OK) {
      Rf_error("Failed to set schema$metadata");
    }

    return R_NilValue;
  }

  // We need this to ensure buffer gets cleaned up amongst the potential longjmp
  // possibilities below.
  SEXP buffer_xptr = PROTECT(buffer_owning_xptr());
  struct ArrowBuffer* buffer = (struct ArrowBuffer*)R_ExternalPtrAddr(buffer_xptr);

  result = ArrowMetadataBuilderInit(buffer, NULL);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowMetadataBuilderInit() failed");
  }

  SEXP metadata_names = PROTECT(Rf_getAttrib(metadata_sexp, R_NamesSymbol));
  if (metadata_names == R_NilValue) {
    Rf_error("schema$metadata must be named");
  }

  struct ArrowStringView key;
  struct ArrowStringView value;

  for (R_xlen_t i = 0; i < Rf_xlength(metadata_sexp); i++) {
    SEXP name_sexp = STRING_ELT(metadata_names, i);
    if (name_sexp == NA_STRING) {
      Rf_error("schema$metadata[[%ld]] must be named", (long)i + 1);
    }

    const void* vmax = vmaxget();
    key = ArrowCharView(Rf_translateCharUTF8(name_sexp));
    if (key.size_bytes == 0) {
      Rf_error("schema$metadata[[%ld]] must be named", (long)i + 1);
    }

    SEXP value_sexp = VECTOR_ELT(metadata_sexp, i);
    if (TYPEOF(value_sexp) == STRSXP && Rf_xlength(value_sexp) == 1) {
      SEXP value_chr = STRING_ELT(value_sexp, 0);
      if (value_chr == NA_STRING) {
        Rf_error("schema$metadata[[%ld]] must not be NA_character_", (long)i + 1);
      }

      value = ArrowCharView(Rf_translateCharUTF8(value_chr));
    } else if (TYPEOF(value_sexp) == RAWSXP) {
      value.data = (const char*)RAW(value_sexp);
      value.size_bytes = Rf_xlength(value_sexp);
    } else {
      Rf_error("schema$metadata[[%ld]] must be character(1) or raw()", (long)i + 1);
    }

    result = ArrowMetadataBuilderAppend(buffer, key, value);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowMetadataBuilderAppend() failed");
    }

    vmaxset(vmax);
  }

  UNPROTECT(1);

  result = ArrowSchemaSetMetadata(schema, (const char*)buffer->data);
  ArrowBufferReset(buffer);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaSetMetadata() failed");
  }

  UNPROTECT(1);
  return R_NilValue;
}

SEXP nanoarrow_c_schema_set_flags(SEXP schema_mut_xptr, SEXP flags_sexp) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_mut_xptr);

  if (TYPEOF(flags_sexp) != INTSXP || Rf_length(flags_sexp) != 1) {
    Rf_error("schema$flags must be integer(1)");
  }

  int flags = INTEGER(flags_sexp)[0];
  schema->flags = flags;

  return R_NilValue;
}

static void release_all_children(struct ArrowSchema* schema) {
  for (int64_t i = 0; i < schema->n_children; i++) {
    if (schema->children[i]->release != NULL) {
      schema->children[i]->release(schema->children[i]);
    }
  }
}

static void free_all_children(struct ArrowSchema* schema) {
  for (int64_t i = 0; i < schema->n_children; i++) {
    if (schema->children[i] != NULL) {
      ArrowFree(schema->children[i]);
      schema->children[i] = NULL;
    }
  }

  if (schema->children != NULL) {
    ArrowFree(schema->children);
    schema->children = NULL;
  }

  schema->n_children = 0;
}

SEXP nanoarrow_c_schema_set_children(SEXP schema_mut_xptr, SEXP children_sexp) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_mut_xptr);

  release_all_children(schema);

  if (Rf_xlength(children_sexp) == 0) {
    free_all_children(schema);
    return R_NilValue;
  }

  int result;
  if (Rf_xlength(children_sexp) != schema->n_children) {
    free_all_children(schema);
    result = ArrowSchemaAllocateChildren(schema, Rf_xlength(children_sexp));
    if (result != NANOARROW_OK) {
      Rf_error("Error allocating schema$children of size %ld",
               (long)Rf_xlength(children_sexp));
    }
  }

  // Names come from names(children) so that we can do
  // names(schema$children)[3] <- "something else" or
  // schema$children[[3]] <- some_unrelated_schema. On the flip
  // side, this makes schema$children[[3]]$name <- "something else"
  // have no effect, which is possibly confusing.
  SEXP children_names = PROTECT(Rf_getAttrib(children_sexp, R_NamesSymbol));

  for (int64_t i = 0; i < schema->n_children; i++) {
    struct ArrowSchema* child = nanoarrow_schema_from_xptr(VECTOR_ELT(children_sexp, i));
    result = ArrowSchemaDeepCopy(child, schema->children[i]);
    if (result != NANOARROW_OK) {
      Rf_error("Error copying new_values$children[[%ld]]", (long)i);
    }

    if (children_names != R_NilValue) {
      SEXP name_sexp = STRING_ELT(children_names, i);

      if (name_sexp == NA_STRING) {
        result = ArrowSchemaSetName(schema->children[i], "");
      } else {
        const void* vmax = vmaxget();
        const char* name = Rf_translateCharUTF8(name_sexp);
        result = ArrowSchemaSetName(schema->children[i], name);
        vmaxset(vmax);
      }
    } else {
      result = ArrowSchemaSetName(schema->children[i], "");
    }

    if (result != NANOARROW_OK) {
      Rf_error("Error copying new_values$children[[%ld]]$name", (long)i);
    }
  }

  UNPROTECT(1);

  return R_NilValue;
}

SEXP nanoarrow_c_schema_set_dictionary(SEXP schema_mut_xptr, SEXP dictionary_xptr) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_mut_xptr);

  // If there's already a dictionary, make sure we release it
  if (schema->dictionary != NULL) {
    if (schema->dictionary->release != NULL) {
      schema->dictionary->release(schema->dictionary);
    }
  }

  if (dictionary_xptr == R_NilValue) {
    if (schema->dictionary != NULL) {
      ArrowFree(schema->dictionary);
      schema->dictionary = NULL;
    }
  } else {
    int result;

    if (schema->dictionary == NULL) {
      result = ArrowSchemaAllocateDictionary(schema);
      if (result != NANOARROW_OK) {
        Rf_error("Error allocating schema$dictionary");
      }
    }

    struct ArrowSchema* dictionary = nanoarrow_schema_from_xptr(dictionary_xptr);
    result = ArrowSchemaDeepCopy(dictionary, schema->dictionary);
    if (result != NANOARROW_OK) {
      Rf_error("Error copying schema$dictionary");
    }
  }

  return R_NilValue;
}
