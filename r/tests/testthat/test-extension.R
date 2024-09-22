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

test_that("extension types can be registered and unregistered", {
  spec <- nanoarrow_extension_spec()
  register_nanoarrow_extension("some_ext", spec)
  expect_identical(resolve_nanoarrow_extension("some_ext"), spec)
  unregister_nanoarrow_extension("some_ext")
  expect_identical(resolve_nanoarrow_extension("some_ext"), NULL)
})

test_that("infer_nanoarrow_ptype() dispatches on registered extension spec", {
  register_nanoarrow_extension(
    "some_ext",
    nanoarrow_extension_spec(subclass = "some_spec_class0")
  )
  on.exit(unregister_nanoarrow_extension("some_ext"))

  infer_nanoarrow_ptype_extension.some_spec_class0 <- function(spec, x, ...) {
    infer_nanoarrow_ptype_extension(NULL, x, ..., warn_unregistered = FALSE)
  }

  s3_register(
    "nanoarrow::infer_nanoarrow_ptype_extension",
    "some_spec_class0",
    infer_nanoarrow_ptype_extension.some_spec_class0
  )

  expect_identical(
    infer_nanoarrow_ptype(
      na_extension(na_struct(list(some_name = na_int32())), "some_ext")
    ),
    data.frame(some_name = integer())
  )
})

test_that("convert_array() dispatches on registered extension spec", {
  register_nanoarrow_extension(
    "some_ext",
    nanoarrow_extension_spec(subclass = "some_spec_class1")
  )
  on.exit(unregister_nanoarrow_extension("some_ext"))

  # Use unique spec class names to avoid interdependency between tests
  convert_array_extension.some_spec_class1 <- function(spec, array, to, ...) {
    convert_array_extension(NULL, array, to, ..., warn_unregistered = FALSE)
  }

  infer_nanoarrow_ptype_extension.some_spec_class1 <- function(spec, x, ...) {
    infer_nanoarrow_ptype_extension(NULL, x, ..., warn_unregistered = FALSE)
  }

  s3_register(
    "nanoarrow::convert_array_extension",
    "some_spec_class1",
    convert_array_extension.some_spec_class1
  )

  s3_register(
    "nanoarrow::infer_nanoarrow_ptype_extension",
    "some_spec_class1",
    infer_nanoarrow_ptype_extension.some_spec_class1
  )

  expect_identical(
    convert_array(
      nanoarrow_extension_array(data.frame(some_name = 1:5), "some_ext")
    ),
    data.frame(some_name = 1:5)
  )
})

test_that("as_nanoarrow_array() dispatches on registered extension spec", {
  register_nanoarrow_extension(
    "some_ext",
    nanoarrow_extension_spec(subclass = "some_spec_class2")
  )
  on.exit(unregister_nanoarrow_extension("some_ext"))

  expect_error(
    as_nanoarrow_array(
      data.frame(some_name = 1:5),
      schema = na_extension(
        na_struct(list(some_name = na_int32())),
        "some_ext"
      )
    ),
    "not implemented for extension"
  )

  as_nanoarrow_array_extension.some_spec_class2 <- function(spec, x, ..., schema = NULL) {
    nanoarrow_extension_array(x, "some_ext")
  }

  s3_register(
    "nanoarrow::as_nanoarrow_array_extension",
    "some_spec_class2",
    as_nanoarrow_array_extension.some_spec_class2
  )

  ext_array <- as_nanoarrow_array(
    data.frame(some_name = 1:5),
    schema = na_extension(
      na_struct(list(some_name = na_int32())),
      "some_ext"
    )
  )

  expect_identical(
    infer_nanoarrow_schema(ext_array)$metadata[["ARROW:extension:name"]],
    "some_ext"
  )
})

test_that("inferring the type of an unregistered extension warns", {
  unknown_extension <- na_extension(na_int32(), "definitely not registered")
  expect_warning(
    infer_nanoarrow_ptype(unknown_extension),
    "Converting unknown extension"
  )

  # Check that warning contains a field name if present
  struct_with_unknown_ext <- na_struct(list(some_col = unknown_extension))
  expect_warning(
    infer_nanoarrow_ptype(struct_with_unknown_ext),
    "some_col: Converting unknown extension"
  )

  previous_opts <- options(nanoarrow.warn_unregistered_extension = FALSE)
  on.exit(options(previous_opts))
  expect_warning(
    infer_nanoarrow_ptype(unknown_extension),
    NA
  )
})

test_that("extensions can infer a schema of a nanoarrow_vctr() subclass", {
  register_nanoarrow_extension(
    "some_ext",
    nanoarrow_extension_spec(subclass = "vctr_spec_class")
  )
  on.exit(unregister_nanoarrow_extension("some_ext"))

  infer_nanoarrow_ptype_extension.vctr_spec_class <- function(spec, x, ...) {
    nanoarrow_vctr(subclass = "some_vctr_subclass")
  }

  s3_register(
    "nanoarrow::infer_nanoarrow_ptype_extension",
    "vctr_spec_class",
    infer_nanoarrow_ptype_extension.vctr_spec_class
  )

  expect_identical(
    infer_nanoarrow_ptype(na_extension(na_string(), "some_ext")),
    nanoarrow_vctr(subclass = "some_vctr_subclass")
  )

  ext_array <- nanoarrow_extension_array(c("one", "two", "three"), "some_ext")
  vctr <- convert_array(ext_array)
  expect_s3_class(vctr, "some_vctr_subclass")

  # Ensure that registering a default conversion that returns a nanoarrow_vctr
  # does not result in infinite recursion when printing or formatting it.
  # An extension that does this should provide these methods for the subclass
  # they return.
  expect_length(format(vctr), length(vctr))
  expect_output(
    expect_identical(print(vctr), vctr),
    "some_vctr_subclass"
  )
})
