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
#' @param extension_name An Arrow extension type name (e.g., nanoarrow.r.vctrs)
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
nanoarrow_extension_spec <- function(data = list(), subclass = character()) {
  structure(
    data,
    class = union(subclass, "nanoarrow_extension_spec")
  )
}

#' @rdname nanoarrow_extension_spec
#' @export
register_nanoarrow_extension <- function(extension_name, extension_spec) {
  extension_registry[[extension_name]] <- extension_spec
  invisible(extension_name)
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
#' @param warn_unregistered Use `FALSE` to infer/convert based on the storage
#'   type without a warning.
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
infer_nanoarrow_ptype_extension <- function(extension_spec, x, ...,
                                            warn_unregistered = TRUE) {
  UseMethod("infer_nanoarrow_ptype_extension")
}

#' @rdname infer_nanoarrow_ptype_extension
#' @export
convert_array_extension <- function(extension_spec, array, to, ...,
                                    warn_unregistered = TRUE) {
  UseMethod("convert_array_extension")
}

#' @rdname infer_nanoarrow_ptype_extension
#' @export
as_nanoarrow_array_extension <- function(extension_spec, x, ..., schema = NULL) {
  UseMethod("as_nanoarrow_array_extension")
}

#' @export
infer_nanoarrow_ptype_extension.default <- function(extension_spec, x, ...,
                                                    warn_unregistered = TRUE) {
  if (warn_unregistered) {
    warn_unregistered_extension_type(x)
  }

  x$metadata[["ARROW:extension:name"]] <- NULL
  infer_nanoarrow_ptype(x)
}

#' @export
convert_array_extension.default <- function(extension_spec, array, to,
                                            ...,
                                            warn_unregistered = TRUE) {
  storage <- .Call(nanoarrow_c_infer_schema_array, array)

  if (warn_unregistered) {
    warn_unregistered_extension_type(storage)
  }

  storage$metadata[["ARROW:extension:name"]] <- NULL

  array <- array_shallow_copy(array, validate = FALSE)
  nanoarrow_array_set_schema(array, storage)
  convert_array(array, to, ...)
}

#' @export
as_nanoarrow_array_extension.default <- function(extension_spec, x, ...,
                                                 schema = NULL) {
  stop(
    sprintf(
      "as_nanoarrow_array_extension() not implemented for extension %s",
      nanoarrow_schema_formatted(schema)
    )
  )
}

#' Create Arrow extension arrays
#'
#' @param storage_array A [nanoarrow_array][as_nanoarrow_array].
#' @inheritParams na_type
#'
#' @return A [nanoarrow_array][as_nanoarrow_array] with attached extension
#'   schema.
#' @export
#'
#' @examples
#' nanoarrow_extension_array(1:10, "some_ext", '{"key": "value"}')
#'
nanoarrow_extension_array <- function(storage_array, extension_name,
                                      extension_metadata = NULL) {
  storage_array <- as_nanoarrow_array(storage_array)

  schema <- .Call(nanoarrow_c_infer_schema_array, storage_array)
  schema$metadata[["ARROW:extension:name"]] <- extension_name
  schema$metadata[["ARROW:extension:metadata"]] <- extension_metadata

  shallow_copy <- array_shallow_copy(storage_array)
  nanoarrow_array_set_schema(shallow_copy, schema)
  shallow_copy
}

warn_unregistered_extension_type <- function(x) {
  # Allow an opt-out of this warning for consumers that don't have
  # control over their source and want to drop unknown extensions
  if (!getOption("nanoarrow.warn_unregistered_extension", TRUE)) {
    return()
  }

  # Warn that we're about to ignore an extension type
  message <- sprintf(
    paste0(
      "Converting unknown extension %s as storage type\n",
      "Disable warning with ",
      "options(nanoarrow.warn_unregistered_extension = FALSE)"
    ),
    nanoarrow_schema_formatted(x)
  )

  # Add the field name if we know it
  if (!is.null(x$name) && !identical(x$name, "")) {
    message <- paste0(x$name, ": ", message)
  }

  warning(message)
}

# Mutable registry to look up extension specifications
extension_registry <- new.env(parent = emptyenv())
