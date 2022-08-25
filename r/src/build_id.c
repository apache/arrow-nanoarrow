#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"

SEXP nanoarrow_c_build_id() {
  return Rf_mkString(NANOARROW_BUILD_ID);
}

SEXP nanoarrow_c_build_id_runtime() {
  return Rf_mkString(ArrowNanoarrowBuildId());
}
