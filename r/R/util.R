
xptr_addr <- function(x) {
  .Call(nanoarrow_c_xptr_addr, x);
}

`%||%` <- function(rhs, lhs) {
  if (is.null(rhs)) lhs else rhs
}

new_data_frame <- function(x, nrow) {
  structure(x, row.names = c(NA, nrow), class = "data.frame")
}
