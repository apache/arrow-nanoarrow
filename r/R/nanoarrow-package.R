#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib nanoarrow, .registration = TRUE
## usethis namespace: end
NULL

#' Underlying 'nanoarrow' C library build
#'
#' @param runtime Compare TRUE and FALSE values to detect a
#'   possible ABI mismatch.
#'
#' @return A string identifying the version of nanoarrow this package
#'   was compiled against.
#' @export
#'
#' @examples
#' nanoarrow_build_id()
#'
nanoarrow_build_id <- function(runtime = TRUE) {
  if (runtime) {
    .Call(nanoarrow_c_build_id_runtime)
  } else {
    .Call(nanoarrow_c_build_id)
  }
}
