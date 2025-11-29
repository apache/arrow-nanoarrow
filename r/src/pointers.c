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

#include "array.h"
#include "array_stream.h"
#include "preserve.h"
#include "schema.h"

SEXP nanoarrow_c_allocate_schema(void) { return nanoarrow_schema_owning_xptr(); }

SEXP nanoarrow_c_allocate_array(void) { return nanoarrow_array_owning_xptr(); }

SEXP nanoarrow_c_allocate_array_stream(void) {
  return nanoarrow_array_stream_owning_xptr();
}

SEXP nanoarrow_c_pointer(SEXP obj_sexp) {
  if (TYPEOF(obj_sexp) == EXTPTRSXP) {
    return obj_sexp;
  } else if (TYPEOF(obj_sexp) == REALSXP && Rf_length(obj_sexp) == 1) {
    // Note that this is not a good idea to actually do; however, is provided for
    // backward compatibility with early versions of the arrow R package.
    intptr_t ptr_int = (intptr_t)(REAL(obj_sexp)[0]);
    return R_MakeExternalPtr((void*)ptr_int, R_NilValue, R_NilValue);
  } else if (TYPEOF(obj_sexp) == STRSXP && Rf_length(obj_sexp) == 1) {
    const char* text = CHAR(STRING_ELT(obj_sexp, 0));
    char* end_ptr;
    intptr_t ptr_int = strtoll(text, &end_ptr, 10);
    if (end_ptr != (text + strlen(text))) {
      Rf_error("'%s' could not be interpreted as an unsigned 64-bit integer", text);
    }

    return R_MakeExternalPtr((void*)ptr_int, R_NilValue, R_NilValue);
  }

  Rf_error("Pointer must be chr[1], dbl[1], or external pointer");
  return R_NilValue;
}

SEXP nanoarrow_c_pointer_addr_dbl(SEXP ptr) {
  // Note that this is not a good idea to actually do; however, is provided for
  // backward compatibility with early versions of the arrow R package.
  uintptr_t ptr_int = (uintptr_t)R_ExternalPtrAddr(nanoarrow_c_pointer(ptr));
  return Rf_ScalarReal((double)ptr_int);
}

SEXP nanoarrow_c_pointer_addr_chr(SEXP ptr) {
  intptr_t ptr_int = (intptr_t)R_ExternalPtrAddr(nanoarrow_c_pointer(ptr));
  char addr_chars[100];
  memset(addr_chars, 0, 100);
  intptr_as_string(ptr_int, addr_chars);
  return Rf_mkString(addr_chars);
}

SEXP nanoarrow_c_pointer_addr_pretty(SEXP ptr) {
  char addr_chars[100];
  memset(addr_chars, 0, 100);
  snprintf(addr_chars, sizeof(addr_chars), "%p",
           R_ExternalPtrAddr(nanoarrow_c_pointer(ptr)));
  return Rf_mkString(addr_chars);
}

SEXP nanoarrow_c_pointer_is_valid(SEXP ptr) {
  if (Rf_inherits(ptr, "nanoarrow_schema")) {
    struct ArrowSchema* obj = (struct ArrowSchema*)R_ExternalPtrAddr(ptr);
    return Rf_ScalarLogical(obj != NULL && obj->release != NULL);
  } else if (Rf_inherits(ptr, "nanoarrow_array")) {
    struct ArrowArray* obj = (struct ArrowArray*)R_ExternalPtrAddr(ptr);
    return Rf_ScalarLogical(obj != NULL && obj->release != NULL);
  } else if (Rf_inherits(ptr, "nanoarrow_array_stream")) {
    struct ArrowArrayStream* obj = (struct ArrowArrayStream*)R_ExternalPtrAddr(ptr);
    return Rf_ScalarLogical(obj != NULL && obj->release != NULL);
  } else {
    Rf_error(
        "`ptr` must inherit from 'nanoarrow_schema', 'nanoarrow_array', or "
        "'nanoarrow_array_stream'");
  }

  return R_NilValue;
}

SEXP nanoarrow_c_pointer_release(SEXP ptr) {
  if (Rf_inherits(ptr, "nanoarrow_schema")) {
    struct ArrowSchema* obj = (struct ArrowSchema*)R_ExternalPtrAddr(ptr);
    if (obj != NULL && obj->release != NULL) {
      obj->release(obj);
      obj->release = NULL;
    }
  } else if (Rf_inherits(ptr, "nanoarrow_array")) {
    struct ArrowArray* obj = (struct ArrowArray*)R_ExternalPtrAddr(ptr);
    if (obj != NULL && obj->release != NULL) {
      obj->release(obj);
      obj->release = NULL;
    }
  } else if (Rf_inherits(ptr, "nanoarrow_array_stream")) {
    struct ArrowArrayStream* obj = (struct ArrowArrayStream*)R_ExternalPtrAddr(ptr);
    if (obj != NULL && obj->release != NULL) {
      obj->release(obj);
      obj->release = NULL;
    }
  } else {
    Rf_error(
        "`ptr` must inherit from 'nanoarrow_schema', 'nanoarrow_array', or "
        "'nanoarrow_array_stream'");
  }

  return R_NilValue;
}

SEXP nanoarrow_c_pointer_move(SEXP ptr_src, SEXP ptr_dst) {
  SEXP xptr_src = PROTECT(nanoarrow_c_pointer(ptr_src));

  if (Rf_inherits(ptr_dst, "nanoarrow_schema")) {
    struct ArrowSchema* obj_dst = (struct ArrowSchema*)R_ExternalPtrAddr(ptr_dst);
    if (obj_dst == NULL) {
      Rf_error("`ptr_dst` is a pointer to NULL");
    }

    if (obj_dst->release != NULL) {
      Rf_error("`ptr_dst` is a valid struct ArrowSchema");
    }

    struct ArrowSchema* obj_src = (struct ArrowSchema*)R_ExternalPtrAddr(xptr_src);
    if (obj_src == NULL || obj_src->release == NULL) {
      Rf_error("`ptr_src` is not a valid struct ArrowSchema");
    }

    ArrowSchemaMove(obj_src, obj_dst);
  } else if (Rf_inherits(ptr_dst, "nanoarrow_array")) {
    struct ArrowArray* obj_dst = (struct ArrowArray*)R_ExternalPtrAddr(ptr_dst);
    if (obj_dst == NULL) {
      Rf_error("`ptr_dst` is a pointer to NULL");
    }

    if (obj_dst->release != NULL) {
      Rf_error("`ptr_dst` is a valid struct ArrowArray");
    }

    struct ArrowArray* obj_src = (struct ArrowArray*)R_ExternalPtrAddr(xptr_src);
    if (obj_src == NULL || obj_src->release == NULL) {
      Rf_error("`ptr_src` is not a valid struct ArrowArray");
    }

    ArrowArrayMove(obj_src, obj_dst);
  } else if (Rf_inherits(ptr_dst, "nanoarrow_array_stream")) {
    struct ArrowArrayStream* obj_dst =
        (struct ArrowArrayStream*)R_ExternalPtrAddr(ptr_dst);
    if (obj_dst == NULL) {
      Rf_error("`ptr_dst` is a pointer to NULL");
    }

    if (obj_dst->release != NULL) {
      Rf_error("`ptr_dst` is a valid struct ArrowArrayStream");
    }

    struct ArrowArrayStream* obj_src =
        (struct ArrowArrayStream*)R_ExternalPtrAddr(xptr_src);
    if (obj_src == NULL || obj_src->release == NULL) {
      Rf_error("`ptr_src` is not a valid struct ArrowArrayStream");
    }

    ArrowArrayStreamMove(obj_src, obj_dst);
  } else {
    Rf_error(
        "`ptr_dst` must inherit from 'nanoarrow_schema', 'nanoarrow_array', or "
        "'nanoarrow_array_stream'");
  }

  // also move SEXP dependencies
  R_SetExternalPtrProtected(ptr_dst, R_ExternalPtrProtected(xptr_src));
  R_SetExternalPtrTag(ptr_dst, R_ExternalPtrTag(xptr_src));
  R_SetExternalPtrProtected(xptr_src, R_NilValue);
  R_SetExternalPtrTag(xptr_src, R_NilValue);

  UNPROTECT(1);
  return R_NilValue;
}

// The rest of this package operates under the assumption that references
// to a schema/array external pointer are kept by anything that needs
// the underlying memory to persist. When the reference count reaches 0,
// R calls the release callback (and nobody else).
// When exporting to something that is expecting to call the release callback
// itself (e.g., Arrow C++ via the arrow R package or pyarrow Python package),
// the structure and the release callback need to keep the information.

// schemas are less frequently iterated over and it's much simpler to
// (recursively) copy the whole object and export it rather than try to
// keep all the object dependencies alive and/or risk moving a dependency
// of some other R object.
SEXP nanoarrow_c_export_schema(SEXP schema_xptr, SEXP ptr_dst) {
  struct ArrowSchema* obj_src = nanoarrow_schema_from_xptr(schema_xptr);
  SEXP xptr_dst = PROTECT(nanoarrow_c_pointer(ptr_dst));

  struct ArrowSchema* obj_dst = (struct ArrowSchema*)R_ExternalPtrAddr(xptr_dst);
  if (obj_dst == NULL) {
    Rf_error("`ptr_dst` is a pointer to NULL");
  }

  if (obj_dst->release != NULL) {
    Rf_error("`ptr_dst` is a valid struct ArrowSchema");
  }

  int result = ArrowSchemaDeepCopy(obj_src, obj_dst);
  if (result != NANOARROW_OK) {
    Rf_error("Failed to deep copy struct ArrowSchema");
  }

  UNPROTECT(1);
  return R_NilValue;
}

SEXP nanoarrow_c_export_array(SEXP array_xptr, SEXP ptr_dst) {
  SEXP xptr_dst = PROTECT(nanoarrow_c_pointer(ptr_dst));

  struct ArrowArray* obj_dst = (struct ArrowArray*)R_ExternalPtrAddr(xptr_dst);
  if (obj_dst == NULL) {
    Rf_error("`ptr_dst` is a pointer to NULL");
  }

  if (obj_dst->release != NULL) {
    Rf_error("`ptr_dst` is a valid struct ArrowArray");
  }

  array_export(array_xptr, obj_dst);
  UNPROTECT(1);
  return R_NilValue;
}

SEXP nanoarrow_c_export_array_stream(SEXP array_stream_xptr, SEXP ptr_dst) {
  SEXP xptr_dst = PROTECT(nanoarrow_c_pointer(ptr_dst));

  struct ArrowArrayStream* obj_dst =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(xptr_dst);
  if (obj_dst == NULL) {
    Rf_error("`ptr_dst` is a pointer to NULL");
  }

  if (obj_dst->release != NULL) {
    Rf_error("`ptr_dst` is a valid struct ArrowArrayStream");
  }

  array_stream_export(array_stream_xptr, obj_dst);

  // Remove SEXP dependencies (if important they are kept alive by array_stream_export)
  R_SetExternalPtrProtected(array_stream_xptr, R_NilValue);
  R_SetExternalPtrTag(array_stream_xptr, R_NilValue);

  UNPROTECT(1);
  return R_NilValue;
}

SEXP nanoarrow_c_pointer_set_protected(SEXP ptr_src, SEXP protected_sexp) {
  if (R_ExternalPtrProtected(ptr_src) != R_NilValue) {
    Rf_error("External pointer protected value has already been set");
  }

  R_SetExternalPtrProtected(ptr_src, protected_sexp);
  return R_NilValue;
}
