#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#ifndef NANOARROW_VERSION_STR
#define NANOARROW_VERSION_STR "dev"
#endif

SEXP nanoarrow_c_version() {
  return Rf_mkString(NANOARROW_VERSION_STR);
}
