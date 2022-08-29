
#' Danger zone: low-level pointer operations
#'
#' @param ptr,ptr_src,ptr_dst An external pointer to a [nanoarrow_schema()],
#'   a [nanoarrow_array()], or a [nanoarrow_array_stream()].
#' @param cls One of "nanoarrow_schema", "nanoarrow_array", or
#'   "nanoarrow_array_stream".
#'
#' @export
#'
nanoarrow_pointer_is_valid <- function(ptr) {
  .Call(nanoarrow_c_pointer_is_valid, ptr)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_pointer_release <- function(ptr) {
  invisible(.Call(nanoarrow_c_pointer_release, ptr))
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_pointer_move <- function(ptr_src, ptr_dst) {
  invisible(.Call(nanoarrow_c_pointer_move, ptr_src, ptr_dst))
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_allocate_schema <- function() {
  .Call(nanoarrow_c_allocate_schema)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_allocate_array_data <- function() {
  .Call(nanoarrow_c_allocate_array_data)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_allocate_array_stream <- function() {
  .Call(nanoarrow_c_allocate_array_stream)
}
