#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"
#include "schema.h"

void finalize_schema_xptr(SEXP schema_xptr) {
  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);
  if (schema != NULL && schema->release != NULL) {
    schema->release(schema);
  }
}
