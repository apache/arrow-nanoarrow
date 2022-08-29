#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "array.h"
#include "nanoarrow.h"

void finalize_array_xptr(SEXP array_xptr) {
  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  if (array != NULL && array->release != NULL) {
    array->release(array);
  }
}
