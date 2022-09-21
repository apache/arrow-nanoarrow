
nanoarrow_altrep <- function(array, ptype = NULL) {
  stopifnot(inherits(array, "nanoarrow_array"))

  if (inherits(ptype, "character")) {
    schema <- infer_nanoarrow_schema(array)
    array_view <- .Call(nanoarrow_c_array_view, array, schema)
    .Call(nanoarrow_c_make_altstring, array_view)
  } else {
    stop(
      sprintf(
        "Can't make nanoarrow::altrep object from ptype '%s'",
        class(ptype)[1]
      )
    )
  }
}
