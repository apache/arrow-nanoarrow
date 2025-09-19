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

test_that("nanoarrow_array format, print, and str methods work", {
  array <- as_nanoarrow_array(1:10)
  expect_identical(format(array), "<nanoarrow_array int32[10]>")
  expect_output(expect_identical(str(array), array), "nanoarrow_array")
  expect_output(expect_identical(print(array), array), "nanoarrow_array")
})

test_that("released nanoarrow_array format, print, and str methods work", {
  array <- nanoarrow_allocate_array()
  expect_identical(format(array), "<nanoarrow_array[invalid pointer]>")
  expect_output(expect_identical(str(array), array), "nanoarrow_array")
  expect_output(expect_identical(print(array), array), "nanoarrow_array")
})

test_that("schemaless nanoarrow_array format, print, and str methods work", {
  array <- as_nanoarrow_array(1:10)
  nanoarrow_array_set_schema(array, NULL)
  expect_identical(format(array), "<nanoarrow_array <unknown schema>[10]>")
  expect_output(expect_identical(str(array), array), "nanoarrow_array")
  expect_output(expect_identical(print(array), array), "nanoarrow_array")
})

test_that("string/binary view nanoarrow_array buffers print correctly", {
  view_array_all_inlined <- as_nanoarrow_array(letters, schema = na_string_view())
  expect_snapshot(print(view_array_all_inlined))

  view_array_not_all_inlined <- as_nanoarrow_array(
    "this string is longer than 12 bytes",
    schema = na_string_view()
  )
  expect_snapshot(print(view_array_not_all_inlined))
})

test_that("as_nanoarrow_array() / convert_array() default method works", {
  array <- as_nanoarrow_array(1:10)
  expect_identical(convert_array(array), 1:10)

  array <- as_nanoarrow_array(as.double(1:10), schema = na_double())
  expect_identical(convert_array(array), as.double(1:10))
})

test_that("infer_nanoarrow_schema() works for nanoarrow_array", {
  array <- as_nanoarrow_array(1:10)
  schema <- infer_nanoarrow_schema(array)
  expect_true(nanoarrow_schema_identical(schema, na_int32()))

  nanoarrow_array_set_schema(array, NULL)
  expect_error(infer_nanoarrow_schema(array), "has no associated schema")
})

test_that("nanoarrow_array_set_schema() errors for invalid schema/array", {
  array <- as_nanoarrow_array(integer())
  schema <- na_string()
  expect_error(
    nanoarrow_array_set_schema(array, schema),
    "Expected array with 3 buffer\\(s\\) but found 2 buffer\\(s\\)"
  )
})

test_that("as.vector() and as.data.frame() work for array", {
  array <- as_nanoarrow_array(1:10)
  expect_identical(as.vector(array), 1:10)

  struct_array <- as_nanoarrow_array(data.frame(a = 1:10))
  expect_identical(as.data.frame(struct_array), data.frame(a = 1:10))
  expect_error(
    as.data.frame(array),
    "Can't convert array with type int32 to data.frame"
  )
})

test_that("as_tibble() works for array()", {
  skip_if_not_installed("tibble")

  struct_array <- as_nanoarrow_array(data.frame(a = 1:10))
  expect_identical(tibble::as_tibble(struct_array), tibble::tibble(a = 1:10))
})

test_that("schemaless array list interface works for non-nested types", {
  array <- as_nanoarrow_array(1:10)
  nanoarrow_array_set_schema(array, NULL)

  expect_identical(length(array), 6L)
  expect_identical(
    names(array),
    c("length",  "null_count", "offset", "buffers", "children",   "dictionary")
  )
  expect_identical(array$length, 10L)
  expect_identical(array$null_count, 0L)
  expect_identical(array$offset, 0L)
  expect_length(array$buffers, 2L)
  expect_s3_class(array$buffers[[1]], "nanoarrow_buffer")
  expect_s3_class(array$buffers[[2]], "nanoarrow_buffer")
  expect_null(array$children)
  expect_null(array$dictionary)
})

test_that("schemaless array list interface works for nested types", {
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two", stringsAsFactors = FALSE))
  nanoarrow_array_set_schema(array, NULL)

  expect_length(array$children, 2L)
  expect_length(array$children[[1]]$buffers, 2L)
  expect_length(array$children[[2]]$buffers, 3L)
  expect_s3_class(array$children[[1]], "nanoarrow_array")
  expect_s3_class(array$children[[2]], "nanoarrow_array")

  info_recursive <- nanoarrow_array_proxy(array, recursive = TRUE)
  expect_type(info_recursive$children[[1]], "list")
  expect_length(info_recursive$children[[1]]$buffers, 2L)
})

test_that("schemaless array list interface works for dictionary types", {
  array <- as_nanoarrow_array(factor(letters[1:5]))
  nanoarrow_array_set_schema(array, NULL)

  expect_length(array$buffers, 2L)
  expect_length(array$dictionary$buffers, 3L)
  expect_s3_class(array$dictionary, "nanoarrow_array")

  info_recursive <- nanoarrow_array_proxy_safe(array, recursive = TRUE)
  expect_type(info_recursive$dictionary, "list")
  expect_length(info_recursive$dictionary$buffers, 3L)
})

test_that("array list interface classes data buffers for relevant types", {
  types <- list(
    int8 = na_int8(),
    uint8 = na_uint8(),
    int16 = na_int16(),
    uint16 = na_uint16(),
    int32 = na_int32(),
    uint32 = na_uint32(),
    int64 = na_int64(),
    uint64 = na_uint64(),
    half_float = na_half_float(),
    float = na_float(),
    double = na_double(),
    decimal128 = na_decimal128(2, 3),
    decimal256 = na_decimal256(2, 3)
  )

  arrays <- lapply(types, function(x) nanoarrow_array_init(x))

  for (nm in names(arrays)) {
    expect_identical(arrays[[!!nm]]$buffers[[1]]$type, "validity")
    expect_identical(arrays[[!!nm]]$buffers[[1]]$data_type, "bool")
    expect_identical(arrays[[!!nm]]$buffers[[2]]$type, "data")
    expect_identical(arrays[[!!nm]]$buffers[[2]]$data_type, nm)
  }
})

test_that("array list interface classes offset buffers for relevant types", {
  arr_string <- nanoarrow_array_init(na_string())
  expect_identical(arr_string$buffers[[2]]$type, "data_offset")
  expect_identical(arr_string$buffers[[2]]$data_type, "int32")
  expect_identical(arr_string$buffers[[3]]$type, "data")
  expect_identical(arr_string$buffers[[3]]$data_type, "string")

  arr_large_string <- nanoarrow_array_init(na_large_string())
  expect_identical(arr_large_string$buffers[[2]]$type, "data_offset")
  expect_identical(arr_large_string$buffers[[2]]$data_type, "int64")

  arr_binary <- nanoarrow_array_init(na_binary())
  expect_identical(arr_binary$buffers[[2]]$type, "data_offset")
  expect_identical(arr_binary$buffers[[2]]$data_type, "int32")

  arr_large_binary <- nanoarrow_array_init(na_large_binary())
  expect_identical(arr_large_binary$buffers[[2]]$type, "data_offset")
  expect_identical(arr_large_binary$buffers[[2]]$data_type, "int64")
})

test_that("array list interface works for nested types", {
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two", stringsAsFactors = FALSE))

  expect_named(array$children, c("a", "b"))
  expect_s3_class(array$children[[1]], "nanoarrow_array")
  expect_s3_class(infer_nanoarrow_schema(array$children[[1]]), "nanoarrow_schema")

  info_recursive <- nanoarrow_array_proxy_safe(array, recursive = TRUE)
  expect_type(info_recursive$children, "list")
  expect_identical(
    info_recursive$children$a$buffers[[2]]$type,
    "data"
  )
  expect_identical(
    info_recursive$children$b$buffers[[2]]$type,
    "data_offset"
  )
})

test_that("array list interface works for dictionary types", {
  array <- as_nanoarrow_array(factor(letters[1:5]))

  expect_identical(array$buffers[[2]]$type, "data")
  expect_identical(array$dictionary$buffers[[2]]$type, "data_offset")

  info_recursive <- nanoarrow_array_proxy_safe(array, recursive = TRUE)
  expect_type(info_recursive$dictionary, "list")
  expect_identical(info_recursive$dictionary$buffers[[2]]$type, "data_offset")
})

test_that("array modify errors for invalid components", {
  array <- as_nanoarrow_array(1:5)

  expect_error(
    nanoarrow_array_modify(array, list(1, 2, 3)),
    "`new_values`"
  )

  expect_error(
    nanoarrow_array_modify(array, list(not_an_item = NULL)),
    "Can't modify array"
  )
})

test_that("array modify does not copy if length(new_values) == 0", {
  array <- as_nanoarrow_array(1:5)
  expect_identical(
    nanoarrow_pointer_addr_chr(nanoarrow_array_modify(array, list())),
    nanoarrow_pointer_addr_chr(array)
  )
})

test_that("array modify can modify length", {
  array <- as_nanoarrow_array(1:5)

  array2 <- nanoarrow_array_modify(array, list(length = 4))
  expect_identical(convert_array(array2), 1:4)
  expect_identical(array$length, 5L)

  expect_error(
    nanoarrow_array_modify(array, list(length = NULL)),
    "array\\$length must be double"
  )

  expect_error(
    nanoarrow_array_modify(array, list(length = NA_real_)),
    "array\\$length must be finite"
  )

  expect_error(
    nanoarrow_array_modify(array, list(length = -1)),
    "array\\$length must be finite and greater than zero"
  )
})

test_that("array modify can modify null_count", {
  array <- as_nanoarrow_array(c(1L, NA, 2L, NA, 3L))

  array2 <- nanoarrow_array_modify(array, list(null_count = -1))
  expect_identical(array2$null_count, -1L)
  expect_identical(array$null_count, 2L)

  expect_error(
    nanoarrow_array_modify(array, list(null_count = NULL)),
    "array\\$null_count must be double"
  )

  expect_error(
    nanoarrow_array_modify(array, list(null_count = NA_real_)),
    "array\\$null_count must be finite"
  )

  expect_error(
    nanoarrow_array_modify(array, list(null_count = -2)),
    "array\\$null_count must be finite and greater than -1"
  )
})

test_that("array modify can modify offset", {
  array <- as_nanoarrow_array(1:5)

  array2 <- nanoarrow_array_modify(array, list(length = 4, offset = 1))
  expect_identical(convert_array(array2), 2:5)
  expect_identical(array$length, 5L)

  expect_error(
    nanoarrow_array_modify(array, list(offset = NULL)),
    "array\\$offset must be double"
  )

  expect_error(
    nanoarrow_array_modify(array, list(offset = NA_real_)),
    "array\\$offset must be finite"
  )

  expect_error(
    nanoarrow_array_modify(array, list(offset = -1)),
    "array\\$offset must be finite and greater than zero"
  )
})

test_that("array modify can modify buffers", {
  array <- as_nanoarrow_array(1:5)

  # Replace with brand new buffer
  array2 <- nanoarrow_array_modify(array, list(buffers = list(NULL, 6:10)))
  expect_identical(convert_array(array2), 6:10)
  expect_identical(convert_array(array), 1:5)

  # Re-use buffers from another array
  array_with_nulls <- as_nanoarrow_array(c(1L, NA, 2L, NA, 3L))
  array2 <- nanoarrow_array_modify(
    array,
    list(
      null_count = -1,
      buffers = list(
        array_with_nulls$buffers[[1]],
        array$buffers[[2]]
      )
    )
  )

  expect_identical(convert_array(array2), c(1L, NA, 3L, NA, 5L))
  expect_identical(convert_array(array), 1:5)
  expect_identical(convert_array(array_with_nulls), c(1L, NA, 2L, NA, 3L))

  # Should work even after the source arrays go out of scope
  array <- NULL
  array_with_nulls <- NULL
  gc()
  expect_identical(convert_array(array2), c(1L, NA, 3L, NA, 5L))

  array <- as_nanoarrow_array(1:5)
  expect_error(
    nanoarrow_array_modify(array, list(buffers = rep(list(NULL), 4))),
    "Changing the number of buffers in array_modify is not supported"
  )

  # Check that specifying too few buffers will result in a validation error
  expect_error(
    nanoarrow_array_modify(array, list(buffers = list()), validate = TRUE),
    "Changing the number of buffers in array_modify is not supported"
  )
})

test_that("array modify can modify variadic buffers", {
  # Create a string view array with >1 variadic buffers. The default
  # internal threshold for splitting up internal buffers is ~30kb.
  boring_strings <- vapply(
    1:10000,
    function(i) paste0("boring string", i),
    character(1)
  )

  array <- as_nanoarrow_array(boring_strings, schema = na_string_view())
  expect_identical(length(array$buffers), 9L)
  variadic_sizes <- convert_buffer(array$buffers[[9]])
  expect_identical(variadic_sizes, c(32757, 32759, 32759, 32759, 32759, 5101))
  expect_identical(convert_array(array), boring_strings)

  # Save the original array
  original_array <- array

  # Modify one of the array buffers
  cool_string_bytes <- charToRaw("cooool string1")
  first_buffer <- as.raw(array$buffers[[3]])
  first_buffer[1:length(cool_string_bytes)] <- cool_string_bytes
  array$buffers[[3]] <- first_buffer

  cool_strings <- boring_strings
  cool_strings[1] <- "cooool string1"
  expect_identical(convert_array(array), cool_strings)

  # Check that the original was unmodified
  expect_identical(convert_array(original_array), boring_strings)
})

test_that("array modify checks buffer sizes", {
  array <- as_nanoarrow_array(1:5)
  expect_error(
    nanoarrow_array_modify(array, list(length = 6)),
    ">= 24 bytes but found buffer with 20 bytes"
  )
})

test_that("array modify can modify children", {
  array_with_children <- as_nanoarrow_array(data.frame(x = 1L))

  # Children -> no children
  array2 <- nanoarrow_array_modify(array_with_children, list(children = NULL))
  expect_identical(
    convert_array(array2),
    new_data_frame(setNames(list(), character()), nrow = 1L)
  )

  # No children -> no children
  array_without_children <- array2
  array2 <- nanoarrow_array_modify(array_with_children, list(children = NULL))
  expect_identical(
    convert_array(array2),
    new_data_frame(setNames(list(), character()), nrow = 1L)
  )

  # No children -> children
  array2 <- nanoarrow_array_modify(
    array_without_children,
    list(children = list(y = 2L))
  )
  expect_identical(convert_array(array2), data.frame(y = 2L))

  # Replace same number of children
  array2 <- nanoarrow_array_modify(
    array_with_children,
    list(children = list(y = 2L))
  )
  expect_identical(convert_array(array2), data.frame(y = 2L))
})

test_that("array modify can modify dictionary", {
  array_without_dictionary <- as_nanoarrow_array(0L)
  array_with_dictionary <- as_nanoarrow_array(factor("a"))

  # No dictionary -> no dictionary
  array2 <- nanoarrow_array_modify(
    array_without_dictionary,
    list(dictionary = NULL)
  )
  expect_identical(convert_array(array2), 0L)

  # No dictionary -> dictionary
  array2 <- nanoarrow_array_modify(
    array_without_dictionary,
    list(dictionary = "a")
  )
  expect_identical(convert_array(array2$dictionary), "a")

  # Dictionary -> no dictionary
  array2 <- nanoarrow_array_modify(
    array_with_dictionary,
    list(dictionary = NULL)
  )
  expect_identical(convert_array(array2), 0L)

  # Dictionary -> new dictionary
  array2 <- nanoarrow_array_modify(
    array_with_dictionary,
    list(dictionary = "b")
  )
  expect_identical(convert_array(array2$dictionary), "b")
})

test_that("array modify can modify array with no schema attached", {
  array <- as_nanoarrow_array(1L)
  nanoarrow_array_set_schema(array, NULL)

  array2 <- nanoarrow_array_modify(array, list(dictionary = c("a", "b")))
  expect_true(!is.null(array2$dictionary))

  array2 <- nanoarrow_array_modify(array, list(children = list("x")))
  expect_length(array2$children, 1)
})

test_that("array modify can skip validation", {
  array <- as_nanoarrow_array(1L)

  expect_error(
    nanoarrow_array_modify(array, list(children = list("x")), validate = TRUE),
    "Expected schema with 0 children"
  )

  array2 <- nanoarrow_array_modify(
    array,
    list(children = list("x")),
    validate = FALSE
  )

  expect_length(array2$children, 1)
})

test_that("[[<- works for array", {
  array <- as_nanoarrow_array(1L)
  array[["length"]] <- 0
  expect_identical(array$length, 0L)

  array <- as_nanoarrow_array(1L)
  array[[1]] <- 0
  expect_identical(array$length, 0L)

  expect_error(
    array[["not_an_item"]] <- "something",
    "Can't modify array"
  )

  expect_error(
    array[[NA_character_]] <- "something",
    "must be character"
  )

  expect_error(
    array[[character()]] <- "something",
    "must be character"
  )

  expect_error(
    array[[NA_integer_]] <- "something",
    "must be character"
  )

  expect_error(
    array[[integer()]] <- "something",
    "must be character"
  )

  expect_error(
    array[[12]] <- "something",
    "must be character"
  )
})

test_that("$<- works for array", {
  array <- as_nanoarrow_array(1L)
  array$length <- 0
  expect_identical(array$length, 0L)

  expect_error(
    array$not_an_item <- "something",
    "Can't modify array"
  )
})

test_that("<- assignment works for array$children", {
  array <- as_nanoarrow_array(
    data.frame(col1 = 1L, col2 = "a", stringsAsFactors = FALSE)
  )

  array$children$col1 <- 100
  expect_identical(
    convert_array(array),
    data.frame(col1 = 100, col2 = "a", stringsAsFactors = FALSE)
  )

  names(array$children)[1] <- "col1_new"
  expect_identical(
    convert_array(array),
    data.frame(col1_new = 100, col2 = "a", stringsAsFactors = FALSE)
  )
})

test_that("<- assignment works for array$buffers", {
  array <- as_nanoarrow_array(c(1:7, NA))
  array$null_count <- -1
  array$buffers[[1]] <- packBits(c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE))

  expect_identical(
    convert_array(array),
    c(1:4, rep(NA, 4))
  )
})

test_that("nanoarrow_array_init() creates an array", {
  array <- nanoarrow_array_init(na_int32())
  expect_identical(convert_array(array), integer())

  # Check error from init
  bad_schema <- nanoarrow_schema_modify(
    na_int32(),
    list(children = list(na_int32())),
    validate = FALSE
  )

  expect_error(
    nanoarrow_array_init(bad_schema),
    "Expected schema with 0 children"
  )
})
