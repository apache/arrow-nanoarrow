#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "array_stream.h"
#include "nanoarrow.h"

void finalize_array_stream_xptr(SEXP array_stream_xptr) {
  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(array_stream_xptr);
  if (array_stream != NULL && array_stream->release != NULL) {
    array_stream->release(array_stream);
  }
}
