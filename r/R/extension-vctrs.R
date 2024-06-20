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

#' Vctrs extension type
#'
#' The Arrow format provides a rich type system that can handle most R
#' vector types; however, many R vector types do not roundtrip perfectly
#' through Arrow memory. The vctrs extension type uses [vctrs::vec_data()],
#' [vctrs::vec_restore()], and [vctrs::vec_ptype()] in calls to
#' [as_nanoarrow_array()] and [convert_array()] to ensure roundtrip fidelity.
#'
#' @param ptype A vctrs prototype as returned by [vctrs::vec_ptype()].
#'   The prototype can be of arbitrary size, but a zero-size vector
#'   is sufficient here.
#' @inheritParams na_type
#'
#' @return A [nanoarrow_schema][as_nanoarrow_schema].
#' @export
#'
#' @examplesIf requireNamespace("jsonlite", quietly = TRUE)
#' vctr <- as.POSIXlt("2000-01-02 03:45", tz = "UTC")
#' array <- as_nanoarrow_array(vctr, schema = na_vctrs(vctr))
#' infer_nanoarrow_ptype(array)
#' convert_array(array)
#'
na_vctrs <- function(ptype, storage_type = NULL) {
  ptype <- vctrs::vec_ptype(ptype)

  if (is.null(storage_type)) {
    storage_type <- infer_nanoarrow_schema(vctrs::vec_data(ptype))
  }

  na_extension(storage_type, "nanoarrow.r.vctrs", serialize_ptype(ptype))
}

register_vctrs_extension <- function() {
  register_nanoarrow_extension(
    "nanoarrow.r.vctrs",
    nanoarrow_extension_spec(subclass = "nanoarrow_extension_spec_vctrs")
  )
}

#' @export
infer_nanoarrow_ptype_extension.nanoarrow_extension_spec_vctrs <- function(extension_spec, x, ...) {
  parsed <- .Call(nanoarrow_c_schema_parse, x)
  unserialize_ptype(parsed$extension_metadata)
}

#' @export
convert_array_extension.nanoarrow_extension_spec_vctrs <- function(extension_spec,
                                                                   array, to,
                                                                   ...) {
  # Restore the vector data to the ptype that is serialized in the type metadata
  to_r_data <- infer_nanoarrow_ptype(array)
  to_data <- vctrs::vec_data(to_r_data)
  data <- convert_array_extension(NULL, array, to_data, warn_unregistered = FALSE)
  vctr <- vctrs::vec_restore(data, to_r_data)

  # Cast to `to` if a different ptype was requested
  if (!is.null(to)) {
    vctrs::vec_cast(vctr, to)
  } else {
    vctr
  }
}

#' @export
as_nanoarrow_array_extension.nanoarrow_extension_spec_vctrs <- function(
    extension_spec, x, ...,
    schema = NULL) {
  storage_schema <- schema
  storage_schema$metadata[["ARROW:extension:name"]] <- NULL
  storage_schema$metadata[["ARROW:extension:metadata"]] <- NULL

  storage_array <- as_nanoarrow_array(
    vctrs::vec_data(x),
    schema = storage_schema
  )

  nanoarrow_extension_array(
    storage_array,
    "nanoarrow.r.vctrs",
    schema$metadata[["ARROW:extension:metadata"]]
  )
}

# The logic for serializing and deserializing prototypes is a subset of
# the implementation in jsonlite. Unlike jsonlite, we don't need to handle
# arbitrary attributes because vector prototypes typically do not contain
# complex information like expression/language objects and environments.
serialize_ptype <- function(x) {
  type <- typeof(x)
  type_serialized <- sprintf('"type":"%s"', type)

  attrs <- attributes(x)
  attributes(x) <- NULL
  if (!is.null(attrs)) {
    attr_names_serialized  <- paste0('"', gsub('"', '\\"', names(attrs)), '"')
    attr_values_serialized <- lapply(unname(attrs), serialize_ptype)
    attrs_serialized <- sprintf(
      '"attributes":{%s}',
      paste0(attr_names_serialized, ":", attr_values_serialized, collapse = ",")
    )
  } else {
    attrs_serialized <- NULL
  }

  if (identical(type, "NULL")) {
    values_serialized <- NULL
  } else if (identical(type, "raw")) {
    values_serialized <- sprintf('"value":"%s"', jsonlite::base64_enc(x))
  } else if(length(x) == 0) {
    values_serialized <- '"value":[]'
  } else {
    values <- switch(
      type,
      character = paste0('"', gsub('"', '\\"', x), '"'),
      complex = paste0('"', format(x, digits = 16, justify = "none", na.encode = FALSE), '"'),
      logical = tolower(as.character(x)),
      integer = ,
      double = format(x, digits = 16, justify = "none", na.encode = FALSE),
      list = lapply(x, serialize_ptype),
      stop(sprintf("storage '%s' is not supported by serialize_ptype", type))
    )

    values[is.na(x)] <- "null"
    values_serialized <- sprintf(
      '"value":[%s]',
      paste(values, collapse = ",")
    )
  }

  sprintf(
    "{%s}",
    paste(
      c(type_serialized, attrs_serialized, values_serialized),
      collapse = ","
    )
  )
}

unserialize_ptype <- function(x) {
  if (is.raw(x)) {
    x <- rawToChar(x)
  }

  unserialize_ptype_impl(jsonlite::fromJSON(x, simplifyVector = FALSE))
}

unserialize_ptype_impl <- function(x) {
  if (identical(x$type, "NULL")) {
    return(NULL)
  } else if(identical(x$type, "raw")) {
    return(jsonlite::base64_dec(x$value))
  }

  sanitizer <- switch(
    x$type,
    raw = as.raw,
    complex = as.complex,
    character = as.character,
    logical = as.logical,
    integer = as.integer,
    double = as.double,
    list = function(x) list(unserialize_ptype_impl(x)),
    stop(sprintf("storage '%s' is not supported by unserialize_ptype", x$type))
  )

  na <- vector(x$type)[1]
  x$value[vapply(x$value, is.null, logical(1))] <- na
  x$value[vapply(x$value, identical, logical(1), "NA")] <- na
  out <- vapply(x$value, sanitizer, na)

  if (!is.null(x$attributes)) {
    attributes(out) <- lapply(x$attributes, unserialize_ptype_impl)
  }

  out
}
