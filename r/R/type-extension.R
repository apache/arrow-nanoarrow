# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#' Register Arrow extension types
#'
#' @param extension_name An Arrow extension type name (e.g., arrow.r.vctrs)
#' @param extension_spec An extension specification inheriting from
#'   'nanoarrow_extension_spec'.
#' @param data Optional data to include in the extension type specification
#' @param subclass A subclass for the extension type specification. Extension
#'    methods will dispatch on this object.
#'
#' @return
#'   - `nanoarrow_extension_spec()` returns an object of class
#'     'nanoarrow_extension_spec'.
#'   - `register_nanoarrow_extension()` returns `extension_spec`, invisibly.
#'   - `unregister_nanoarrow_extension()` returns `extension_name`, invisibly.
#'   - `resolve_nanoarrow_extension()` returns an object of class
#'     'nanoarrow_extension_spec' or NULL if the extension type was not
#'     registered.
#' @export
#'
#' @examples
#' nanoarrow_extension_spec("mynamespace.mytype", subclass = "mypackage_mytype_spec")
nanoarrow_extension_spec <- function(extension_name, data = NULL,
                                     subclass = character()) {
  structure(
    list(extension_name = extension_name, data = data),
    class = union(subclass, "nanoarrow_extension_spec")
  )
}

#' @rdname nanoarrow_extension_spec
#' @export
register_nanoarrow_extension <- function(extension_spec) {
  extension_registry[[extension_spec$extension_name]] <- extension_spec
  invisible(extension_spec)
}

#' @rdname nanoarrow_extension_spec
#' @export
unregister_nanoarrow_extension <- function(extension_name) {
  extension_registry[[extension_name]] <- NULL
  invisible(extension_name)
}

#' @rdname nanoarrow_extension_spec
#' @export
resolve_nanoarrow_extension <- function(extension_name) {
  extension_registry[[extension_name]]
}


#' Implement Arrow extension types
#'
#' @inheritParams nanoarrow_extension_spec
#' @param x,array,to,schema,... Passed from [infer_nanoarrow_ptype()],
#'   [convert_array()], [as_nanoarrow_array()], and/or
#'   [as_nanoarrow_array_stream()].
#'
#' @return
#'   - `infer_nanoarrow_ptype_extension()`: The R vector prototype to be used
#'     as the default conversion target.
#'   - `convert_array_extension()`: An R vector of type `to`.
#'   - `as_nanoarrow_array_extension()`: A [nanoarrow_array][as_nanoarrow_array]
#'     of type `schema`.
#' @export
#'
infer_nanoarrow_ptype_extension <- function(extension_spec, x) {
  UseMethod("infer_nanoarrow_ptype_extension")
}

#' @export
infer_nanoarrow_ptype_extension.default <- function(extension_spec, x) {
  # Warn that we're about to ignore an extension type
  if (!is.null(x$name) && !identical(x$name, "")) {
    warning(
      sprintf(
        "%s: Converting unknown extension %s as storage type",
        x$name,
        nanoarrow_schema_formatted(x)
      )
    )
  } else {
    warning(
      sprintf(
        "Converting unknown extension %s as storage type",
        nanoarrow_schema_formatted(x)
      )
    )
  }


  x$metadata[["ARROW:extension:name"]] <- NULL
  infer_nanoarrow_ptype(x)
}

#' @rdname infer_nanoarrow_ptype_extension
#' @export
convert_array_extension <- function(extension_spec, array, to, ...) {
  UseMethod("convert_array_extension")
}

#' @export
convert_array_extension.default <- function(extension_spec, array, to, ...) {
  message("Fish")
  storage <- .Call(nanoarrow_c_infer_schema_array, array)
  storage$metadata[["ARROW:extension:name"]] <- NULL

  array <- array_shallow_copy(array, validate = FALSE)
  nanoarrow_array_set_schema(array, storage)
  convert_array(array, to, ...)
}

#' @rdname infer_nanoarrow_ptype_extension
#' @export
as_nanoarrow_array_extension <- function(extension_spec, x, ..., schema = NULL) {
  UseMethod("as_nanoarrow_array_extension")
}

#' @export
as_nanoarrow_array_extension.default <- function(extension_spec, x, ..., schema = NULL) {
  stop("Not implemented")
}

# Mutable registry to look up extension specifications
extension_registry <- new.env(parent = emptyenv())
