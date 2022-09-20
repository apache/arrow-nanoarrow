
nanoarrow_altrep <- function(array) {
  stopifnot(inherits(array, "nanoarrow_array"))
  schema <- infer_nanoarrow_schema(array)
  array_view <- .Call(nanoarrow_c_array_view, array, schema)
  .Call(nanoarrow_c_make_altrep, array_view)
}
