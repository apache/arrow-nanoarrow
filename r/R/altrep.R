
# For testing the altrep chr conversion
nanoarrow_altrep_chr <- function(array) {
  .Call(nanoarrow_c_make_altrep_chr, array)
}

is_nanoarrow_altrep <- function(x) {
  .Call(nanoarrow_c_is_altrep, x)
}

nanoarrow_altrep_force_materialize <- function(x, recursive = FALSE) {
  invisible(.Call(nanoarrow_c_altrep_force_materialize, x, recursive))
}

is_nanoarrow_altrep_materialized <- function(x) {
  .Call(nanoarrow_c_altrep_is_materialized, x)
}
