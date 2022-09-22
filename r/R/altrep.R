
nanoarrow_altrep <- function(array, ptype = NULL) {
  stopifnot(inherits(array, "nanoarrow_array"))

  if (inherits(ptype, "character")) {
    schema <- infer_nanoarrow_schema(array)
    array_view <- .Call(nanoarrow_c_array_view, array, schema)
    .Call(nanoarrow_c_make_altstring, array_view)
  } else {
    NULL
  }
}

is_nanoarrow_altrep <- function(x) {
  cls <- .Call(nanoarrow_c_altrep_class, x)
  if (is.null(cls)) FALSE else grepl("^nanoarrow::", cls)
}

nanoarrow_altrep_force_materialize <- function(x, recursive = FALSE) {
  invisible(.Call(nanoarrow_c_altrep_force_materialize, x, recursive))
}

is_nanoarrow_altrep_materialized <- function(x) {
  .Call(nanoarrow_c_altrep_is_materialized, x)
}
