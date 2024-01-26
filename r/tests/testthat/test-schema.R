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

test_that("nanoarrow_schema format, print, and str methods work", {
  schema <- na_int32()
  expect_identical(format(schema), "<nanoarrow_schema int32>")
  expect_output(expect_identical(str(schema), schema), "nanoarrow_schema")
  expect_output(expect_identical(print(schema), schema), "nanoarrow_schema")
})

test_that("nanoarrow_schema format, print, and str methods work for invalid pointers", {
  schema <- nanoarrow_allocate_schema()
  expect_identical(format(schema), "<nanoarrow_schema [invalid: schema is released]>")
  expect_output(expect_identical(str(schema), schema), "nanoarrow_schema")
  expect_output(expect_identical(print(schema), schema), "nanoarrow_schema")
})

test_that("as_nanoarrow_schema() works for nanoarrow_schema", {
  schema <- na_int32()
  expect_identical(as_nanoarrow_schema(schema), schema)
})

test_that("infer_nanoarrow_schema() errors for unsupported types", {
  expect_error(
    infer_nanoarrow_schema(environment()),
    "Can't infer Arrow type"
  )
})

test_that("infer_nanoarrow_schema() methods work for built-in types", {
  expect_identical(infer_nanoarrow_schema(raw())$format, "C")
  expect_identical(infer_nanoarrow_schema(logical())$format, "b")
  expect_identical(infer_nanoarrow_schema(integer())$format, "i")
  expect_identical(infer_nanoarrow_schema(double())$format, "g")
  expect_identical(infer_nanoarrow_schema(character())$format, "u")
  expect_identical(infer_nanoarrow_schema(Sys.Date())$format, "tdD")

  expect_identical(infer_nanoarrow_schema(factor())$format, "i")
  expect_identical(infer_nanoarrow_schema(factor())$dictionary$format, "u")

  time <- as.POSIXct("2000-01-01", tz = "UTC")
  expect_identical(infer_nanoarrow_schema(time)$format, "tsu:UTC")

  # Some systems (mostly Docker images) don't return a value for Sys.timezone()
  # so set one explicitly to test the Sys.timezone() fallback.
  withr::with_timezone("America/Halifax", {
    time <- as.POSIXct("2000-01-01", tz = "")
    expect_identical(
      infer_nanoarrow_schema(time)$format,
      paste0("tsu:", Sys.timezone())
    )
  })

  difftime <- as.difftime(double(), unit = "secs")
  expect_identical(infer_nanoarrow_schema(difftime)$format, "tDu")

  df_schema <- infer_nanoarrow_schema(data.frame(x = 1L))
  expect_identical(df_schema$format, "+s")
  expect_identical(df_schema$children$x$format, "i")
})

test_that("infer_nanoarrow_schema() methods work for blob type", {
  skip_if_not_installed("blob")

  expect_identical(infer_nanoarrow_schema(blob::blob())$format, "z")
})

test_that("infer_nanoarrow_schema() methods work for hms type", {
  skip_if_not_installed("hms")

  expect_identical(infer_nanoarrow_schema(hms::hms())$format, "ttm")
})

test_that("infer_nanoarrow_schema() methods work for vctrs types", {
  skip_if_not_installed("vctrs")

  expect_identical(infer_nanoarrow_schema(vctrs::unspecified())$format, "n")

  list_schema <- infer_nanoarrow_schema(vctrs::list_of(.ptype = integer()))
  expect_identical(list_schema$format, "+l")
  expect_identical(list_schema$children[[1]]$format, "i")
})

test_that("infer_nanoarrow_schema() method works for integer64()", {
  skip_if_not_installed("bit64")
  expect_identical(infer_nanoarrow_schema(bit64::integer64())$format, "l")
})

test_that("infer_nanoarrow_schema() method works for AsIs", {
  expect_identical(
    infer_nanoarrow_schema(I(integer()))$format,
    infer_nanoarrow_schema(integer())$format
  )
})

test_that("infer_nanoarrow_schema() returns list of null for empty or all null list", {
  expect_identical(infer_nanoarrow_schema(list())$format, "+l")
  expect_identical(infer_nanoarrow_schema(list())$children[[1]]$format, "n")
  expect_identical(infer_nanoarrow_schema(list(NULL))$format, "+l")
  expect_identical(infer_nanoarrow_schema(list())$children[[1]]$format, "n")
})

test_that("infer_nanoarrow_schema() returns binary for list of raw", {
  expect_identical(infer_nanoarrow_schema(list(raw()))$format, "z")
  expect_identical(infer_nanoarrow_schema(list(raw(), NULL))$format, "z")
})

test_that("nanoarrow_schema_parse() works", {
  simple_info <- nanoarrow_schema_parse(na_int32())
  expect_identical(simple_info$type, "int32")
  expect_identical(simple_info$storage_type, "int32")

  fixed_size_info <- nanoarrow_schema_parse(na_fixed_size_binary(1234))
  expect_identical(fixed_size_info$fixed_size, 1234L)

  decimal_info <- nanoarrow_schema_parse(na_decimal128(4, 5))
  expect_identical(decimal_info$decimal_bitwidth, 128L)
  expect_identical(decimal_info$decimal_precision, 4L)
  expect_identical(decimal_info$decimal_scale, 5L)

  time_unit_info <- nanoarrow_schema_parse(na_time32("s"))
  expect_identical(time_unit_info$time_unit, "s")

  timezone_info <- nanoarrow_schema_parse(na_timestamp("s", "America/Halifax"))
  expect_identical(timezone_info$timezone, "America/Halifax")

  recursive_info <- nanoarrow_schema_parse(
    na_struct(list(x = na_int32())),
    recursive = FALSE
  )
  expect_null(recursive_info$children)

  recursive_info <- nanoarrow_schema_parse(
    na_struct(list(x = na_int32())),
    recursive = TRUE
  )
  expect_length(recursive_info$children, 1L)
  expect_identical(
    recursive_info$children$x,
    nanoarrow_schema_parse(na_int32())
  )
})

test_that("nanoarrow_schema_parse() works for extension types", {
  ext_info <- nanoarrow_schema_parse(na_extension(na_int32(), "ext_name", "ext_meta"))
  expect_identical(ext_info$type, "int32")
  expect_identical(ext_info$storage_type, "int32")
  expect_identical(ext_info$extension_name, "ext_name")
  expect_identical(ext_info$extension_metadata, charToRaw("ext_meta"))
})

test_that("schema list interface works for non-nested types", {
  schema <- na_int32()
  expect_identical(length(schema), 6L)
  expect_identical(
    names(schema),
    c("format", "name", "metadata", "flags", "children", "dictionary")
  )
  expect_identical(schema$format, "i")
  expect_identical(schema$name, "")
  expect_identical(schema$metadata, list())
  expect_identical(schema$flags, 2L)
  expect_identical(schema$children, list())
  expect_identical(schema$dictionary, NULL)
})

test_that("schema list interface works for nested types", {
  schema <- na_struct(list(a = na_int32(), b = na_string()))

  expect_identical(schema$format, "+s")
  expect_named(schema$children, c("a", "b"))
  expect_identical(schema$children$a, schema$children[[1]])
  expect_identical(schema$children$a$format, "i")
  expect_identical(schema$children$b$format, "u")
  expect_s3_class(schema$children$a, "nanoarrow_schema")
  expect_s3_class(schema$children$b, "nanoarrow_schema")

  info_recursive <- nanoarrow_schema_proxy(schema, recursive = TRUE)
  expect_type(info_recursive$children$a, "list")
  expect_identical(info_recursive$children$a$format, "i")
})

test_that("schema list interface works for dictionary types", {
  schema <- na_dictionary(na_string(), na_int8())

  expect_identical(schema$format, "c")
  expect_identical(schema$dictionary$format, "u")
  expect_s3_class(schema$dictionary, "nanoarrow_schema")

  info_recursive <- nanoarrow_schema_proxy(schema, recursive = TRUE)
  expect_type(info_recursive$dictionary, "list")
  expect_identical(info_recursive$dictionary$format, "u")
})

test_that("schema list interface works with metadata", {
  schema <- na_extension(na_int32(), "ext_name", "ext_meta")
  expect_identical(
    schema$metadata[["ARROW:extension:name"]],
    "ext_name"
  )
  expect_identical(
    schema$metadata[["ARROW:extension:metadata"]],
    "ext_meta"
  )
})

test_that("schema modify errors for invalid components", {
  schema <- na_int32()

  expect_error(
    nanoarrow_schema_modify(schema, list(1, 2, 3)),
    "`new_values`"
  )

  expect_error(
    nanoarrow_schema_modify(schema, list(not_an_item = NULL)),
    "Can't modify schema"
  )
})

test_that("schema modify does not copy if length(new_values) == 0", {
  schema <- na_int32()
  expect_identical(
    nanoarrow_pointer_addr_chr(nanoarrow_schema_modify(schema, list())),
    nanoarrow_pointer_addr_chr(schema)
  )
})

test_that("schema modify can modify format", {
  schema <- na_int32()

  schema2 <- nanoarrow_schema_modify(schema, list(format = "I"))
  expect_identical(schema2$format, "I")
  expect_identical(schema2$name, schema$name)
  expect_identical(schema2$flags, schema$flags)

  expect_error(
    nanoarrow_schema_modify(schema, list(format = NULL)),
    "schema\\$format must be character"
  )

  expect_error(
    nanoarrow_schema_modify(schema, list(format = character())),
    "schema\\$format must be character"
  )
})

test_that("schema modify can modify name", {
  schema <- na_int32()

  schema2 <- nanoarrow_schema_modify(schema, list(name = "new_name"))
  expect_identical(schema2$name, "new_name")
  expect_identical(schema2$format, schema$format)
  expect_identical(schema2$flags, schema$flags)

  schema2 <- nanoarrow_schema_modify(schema, list(name = NULL))
  expect_null(schema2$name)
  expect_identical(schema2$format, schema$format)
  expect_identical(schema2$flags, schema$flags)

  expect_error(
    nanoarrow_schema_modify(schema, list(name = character())),
    "schema\\$name must be NULL or character"
  )
})

test_that("schema modify can modify flags", {
  schema <- na_int32()
  expect_identical(schema$flags, 2L)

  schema2 <- nanoarrow_schema_modify(schema, list(flags = 0))
  expect_identical(schema2$flags, 0L)
  expect_identical(schema2$format, schema$format)
  expect_identical(schema2$name, schema$name)

  expect_error(
    nanoarrow_schema_modify(schema, list(flags = integer())),
    "schema\\$flags must be integer"
  )
})

test_that("schema modify can modify metadata", {
  schema <- na_int32()

  schema2 <- nanoarrow_schema_modify(schema, list(metadata = list()))
  expect_identical(schema2$metadata, list())
  expect_identical(schema2$format, schema$format)

  schema3 <- nanoarrow_schema_modify(schema, list(metadata = NULL))
  expect_identical(schema3$metadata, list())
  expect_identical(schema3$format, schema$format)

  schema4 <- nanoarrow_schema_modify(schema, list(metadata = list(key = "value")))
  expect_identical(schema4$metadata, list(key = "value"))
  expect_identical(schema4$format, schema$format)

  schema5 <- nanoarrow_schema_modify(
    schema,
    list(metadata = list(new_key = charToRaw("new value")))
  )
  expect_identical(schema5$metadata, list(new_key = "new value"))
  expect_identical(schema5$format, schema$format)

  expect_error(
    nanoarrow_schema_modify(schema, list(metadata = list(1))),
    "schema\\$metadata must be named"
  )

  expect_error(
    nanoarrow_schema_modify(schema, list(metadata = setNames(list(1, 2), c("", "")))),
    "must be named"
  )

  expect_error(
    nanoarrow_schema_modify(schema, list(metadata = setNames(list(1), NA_character_))),
    "must be named"
  )

  expect_error(
    nanoarrow_schema_modify(schema, list(metadata = list(name = NULL))),
    "must be character\\(1\\) or raw"
  )

  expect_error(
    nanoarrow_schema_modify(schema, list(metadata = list(name = character()))),
    "must be character\\(1\\) or raw"
  )

  expect_error(
    nanoarrow_schema_modify(schema, list(metadata = list(name = NA_character_))),
    "must not be NA_character_"
  )
})

test_that("schema modify can modify children", {
  schema_without_children <- na_struct()
  child_to_be <- schema_without_children
  child_to_be$name <- "should not appear"

  # NULL children to NULL children
  schema2 <- nanoarrow_schema_modify(
    schema_without_children,
    list(children = NULL)
  )
  expect_identical(schema2$children, list())
  expect_identical(schema2$format, schema_without_children$format)

  # NULL children to zero-size list() children
  schema2 <- nanoarrow_schema_modify(
    schema_without_children,
    list(children = list())
  )
  expect_identical(schema2$children, list())
  expect_identical(schema2$format, schema_without_children$format)

  # with unnamed child list
  schema2 <- nanoarrow_schema_modify(
    schema_without_children,
    list(children = list(child_to_be))
  )
  expect_length(schema2$children, 1)
  expect_named(schema2$children, "")
  expect_identical(schema2$format, schema_without_children$format)
  expect_identical(schema2$children[[1]]$format, child_to_be$format)

  # with another type of unnamed child list
  schema2 <- nanoarrow_schema_modify(
    schema_without_children,
    list(children = setNames(list(child_to_be), ""))
  )
  expect_length(schema2$children, 1)
  expect_named(schema2$children, "")
  expect_identical(schema2$format, schema_without_children$format)
  expect_identical(schema2$children[[1]]$format, child_to_be$format)

  # with oddly unnamed child list
  schema2 <- nanoarrow_schema_modify(
    schema_without_children,
    list(children = setNames(list(child_to_be), NA_character_))
  )
  expect_length(schema2$children, 1)
  expect_named(schema2$children, "")
  expect_identical(schema2$format, schema_without_children$format)
  expect_identical(schema2$children[[1]]$format, child_to_be$format)

  # with a normal named child list
  schema2 <- nanoarrow_schema_modify(
    schema_without_children,
    list(children = list("a new name" = child_to_be))
  )
  expect_length(schema2$children, 1)
  expect_named(schema2$children, "a new name")
  expect_identical(schema2$format, schema_without_children$format)
  expect_identical(schema2$children[[1]]$format, child_to_be$format)


  schema_with_children <- na_struct(list(existing_name = na_string()))

  # some children to NULL children
  schema2 <- nanoarrow_schema_modify(
    schema_with_children,
    list(children = NULL)
  )
  expect_identical(schema2$children, list())
  expect_identical(schema2$format, schema_with_children$format)

  # replace identical number of children
  schema2 <- nanoarrow_schema_modify(
    schema_with_children,
    list(children = list("a new name" = child_to_be))
  )
  expect_length(schema2$children, 1)
  expect_named(schema2$children, "a new name")
  expect_identical(schema2$format, schema_with_children$format)
  expect_identical(schema2$children[[1]]$format, child_to_be$format)

  # replace with more children
  another_child_to_be <- na_bool()
  schema2 <- nanoarrow_schema_modify(
    schema_with_children,
    list(
      children = list(
        "a new name" = child_to_be,
        "another new name" = another_child_to_be
      )
    )
  )
  expect_length(schema2$children, 2)
  expect_named(schema2$children, c("a new name", "another new name"))
  expect_identical(schema2$format, schema_with_children$format)
  expect_identical(schema2$children[[1]]$format, child_to_be$format)
  expect_identical(schema2$children[[2]]$format, another_child_to_be$format)
})

test_that("schema modify can modify dictionary", {
  schema_without_dictionary <- na_int32()

  # NULL -> NULL
  schema2 <- nanoarrow_schema_modify(
    schema_without_dictionary,
    list(dictionary = NULL)
  )

  expect_null(schema2$dictionary)
  expect_identical(schema2$flags, schema_without_dictionary$flags)
  expect_identical(schema2$format, schema_without_dictionary$format)
  expect_identical(schema2$name, schema_without_dictionary$name)

  # NULL -> non-null
  schema2 <- nanoarrow_schema_modify(
    schema_without_dictionary,
    list(dictionary = na_int32())
  )

  expect_identical(schema2$dictionary$format, "i")
  expect_identical(schema2$flags, schema_without_dictionary$flags)
  expect_identical(schema2$format, schema_without_dictionary$format)
  expect_identical(schema2$name, schema_without_dictionary$name)

  # non-null -> NULL
  schema_with_dictionary <- schema2
  schema2 <- nanoarrow_schema_modify(
    schema_with_dictionary,
    list(dictionary = NULL)
  )

  expect_null(schema2$dictionary)
  expect_identical(schema2$flags, schema_with_dictionary$flags)
  expect_identical(schema2$format, schema_with_dictionary$format)
  expect_identical(schema2$name, schema_with_dictionary$name)

  # non-null -> non-null
  schema2 <- nanoarrow_schema_modify(
    schema_with_dictionary,
    list(dictionary = na_string())
  )

  expect_identical(schema2$dictionary$format, "u")
  expect_identical(schema2$flags, schema_with_dictionary$flags)
  expect_identical(schema2$format, schema_with_dictionary$format)
  expect_identical(schema2$name, schema_with_dictionary$name)
})

test_that("schema modify respects the validate flag", {
  schema <- na_int32()

  schema2 <- nanoarrow_schema_modify(
    schema,
    list(format = "totally invalid"),
    validate = FALSE
  )

  expect_identical(schema2$format, "totally invalid")

  expect_error(
    nanoarrow_schema_modify(
      schema,
      list(format = "totally invalid"),
      validate = TRUE
    ),
    "Error parsing schema->format"
  )
})

test_that("[[<- works for schema", {
  schema <- na_int32()
  schema[["name"]] <- "a new name"
  expect_identical(schema$name, "a new name")

  schema <- na_int32()
  schema[[2]] <- "yet a new name"
  expect_identical(schema$name, "yet a new name")

  expect_error(
    schema[["not_an_item"]] <- "something",
    "Can't modify schema"
  )

  expect_error(
    schema[[NA_character_]] <- "something",
    "must be character"
  )

  expect_error(
    schema[[character()]] <- "something",
    "must be character"
  )

  expect_error(
    schema[[NA_integer_]] <- "something",
    "must be character"
  )

  expect_error(
    schema[[integer()]] <- "something",
    "must be character"
  )

  expect_error(
    schema[[12]] <- "something",
    "must be character"
  )
})

test_that("$<- works for schema", {
  schema <- na_int32()
  schema$name <- "a new name"
  expect_identical(schema$name, "a new name")

  expect_error(
    schema$not_an_item <- "something",
    "Can't modify schema"
  )
})

test_that("<- assignment works for schema$children", {
  schema <- na_struct(list(col1 = na_int32(), col2 = na_string()))

  schema$children$col1 <- na_bool()
  expect_named(schema$children, c("col1", "col2"))
  expect_identical(schema$children$col1$format, "b")
  expect_identical(schema$children$col1$name, "col1")

  names(schema$children)[1] <- "col1_new"
  expect_named(schema$children, c("col1_new", "col2"))
  expect_identical(schema$children$col1_new$format, "b")
  expect_identical(schema$children$col1_new$name, "col1_new")
})

test_that("<- assignment works for schema$metadata", {
  schema <- na_int32()

  schema$metadata$key <- "value"
  expect_identical(schema$metadata$key, "value")

  names(schema$metadata)[1] <- "new_key"
  expect_identical(schema$metadata$new_key, "value")

  schema$metadata$new_key <- "new value"
  expect_identical(schema$metadata$new_key, "new value")
})
