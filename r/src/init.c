#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

SEXP nanoarrow_c_build_id();
SEXP nanoarrow_c_build_id_runtime();

static const R_CallMethodDef CallEntries[] = {
  {"nanoarrow_c_build_id", (DL_FUNC) &nanoarrow_c_build_id, 0},
  {"nanoarrow_c_build_id_runtime", (DL_FUNC) &nanoarrow_c_build_id_runtime, 0},
  {NULL, NULL, 0}
};

void R_init_nanoarrow(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
