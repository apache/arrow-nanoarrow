#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib nanoarrow, .registration = TRUE
## usethis namespace: end
NULL

#' Underlying 'nanoarrow' C library version
#'
#' @return A string identifying the version of nanoarrow this package
#'   was compiled against.
#' @export
#'
#' @examples
#' nanoarrow_version()
#'
nanoarrow_version <- function() {
  .Call(nanoarrow_c_version)
}
