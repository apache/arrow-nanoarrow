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

test_that("infer_type() for nanoarrow_array works", {
  skip_if_not_installed("arrow")

  array <- as_nanoarrow_array(1:5)
  expect_true(
    arrow::infer_type(array)$Equals(arrow::int32())
  )
})

test_that("infer_nanoarrow_schema() works for arrow objects", {
  skip_if_not_installed("arrow")

  int_schema <- infer_nanoarrow_schema(arrow::Array$create(1:10))
  expect_true(arrow::as_data_type(int_schema)$Equals(arrow::int32()))

  int_schema <- infer_nanoarrow_schema(arrow::Scalar$create(1L))
  expect_true(arrow::as_data_type(int_schema)$Equals(arrow::int32()))

  int_schema <- infer_nanoarrow_schema(arrow::ChunkedArray$create(1:10))
  expect_true(arrow::as_data_type(int_schema)$Equals(arrow::int32()))

  int_schema <- infer_nanoarrow_schema(arrow::Expression$scalar(1L))
  expect_true(arrow::as_data_type(int_schema)$Equals(arrow::int32()))

  tbl_schema_expected <- arrow::schema(x = arrow::int32())
  tbl_schema <- infer_nanoarrow_schema(arrow::record_batch(x = 1L))
  expect_true(arrow::as_schema(tbl_schema)$Equals(tbl_schema_expected))

  tbl_schema <- infer_nanoarrow_schema(arrow::arrow_table(x = 1L))
  expect_true(arrow::as_schema(tbl_schema)$Equals(tbl_schema_expected))

  tbl_schema <- infer_nanoarrow_schema(
    arrow::RecordBatchReader$create(arrow::record_batch(x = 1L))
  )
  expect_true(arrow::as_schema(tbl_schema)$Equals(tbl_schema_expected))

  skip_if_not(arrow::arrow_info()$capabilities["dataset"])

  tbl_schema <- infer_nanoarrow_schema(
    arrow::InMemoryDataset$create(arrow::record_batch(x = 1L))
  )
  expect_true(arrow::as_schema(tbl_schema)$Equals(tbl_schema_expected))

  tbl_schema <- infer_nanoarrow_schema(
    arrow::Scanner$create(
      arrow::InMemoryDataset$create(arrow::record_batch(x = 1L))
    )
  )
  expect_true(arrow::as_schema(tbl_schema)$Equals(tbl_schema_expected))
})

test_that("nanoarrow_array to Array works", {
  skip_if_not_installed("arrow")

  int <- arrow::as_arrow_array(as_nanoarrow_array(1:5))
  expect_true(int$Equals(arrow::Array$create(1:5)))

  dbl <- arrow::as_arrow_array(as_nanoarrow_array(1:5, schema = arrow::float64()))
  expect_true(dbl$Equals(arrow::Array$create(1:5, type = arrow::float64())))

  dbl_casted <- arrow::as_arrow_array(as_nanoarrow_array(1:5), type = arrow::float64())
  expect_true(dbl_casted$Equals(arrow::Array$create(1:5, type = arrow::float64())))

  chr <- arrow::as_arrow_array(as_nanoarrow_array(c("one", "two")))
  expect_true(chr$Equals(arrow::Array$create(c("one", "two"))))
})

test_that("nanoarrow_array to Array works for child arrays", {
  skip_if_not_installed("arrow")

  df <- data.frame(a = 1, b = "two")
  batch <- as_nanoarrow_array(df)

  # This type of export is special because batch$children[[2]] has an SEXP
  # dependency on the original array. When we export it, we reverse that
  # dependency such that the exported array and the batch->children[1] array
  # are shells that call nanoarrow_release_sexp on a common object (i.e., sort of like
  # a shared pointer).
  array_from_column <- arrow::as_arrow_array(batch$children[[2]])

  # The exported array should be valid
  expect_null(array_from_column$Validate())

  # All the nanoarrow pointers should still be valid
  expect_true(nanoarrow_pointer_is_valid(batch))
  expect_true(nanoarrow_pointer_is_valid(batch$children[[1]]))
  expect_true(nanoarrow_pointer_is_valid(batch$children[[2]]))

  # Let the exported arrow::Array go out of scope and maximize the
  # chance that the exported data release callback is called
  array_from_column <- NULL
  gc()
  Sys.sleep(0.1)

  # All the nanoarrow pointers should *still* be valid even after that
  # release callback is called
  expect_true(nanoarrow_pointer_is_valid(batch))
  expect_true(nanoarrow_pointer_is_valid(batch$children[[1]]))
  expect_true(nanoarrow_pointer_is_valid(batch$children[[2]]))

  # Export one column again but this time let the `batch` go out of scope
  array_from_column <- arrow::as_arrow_array(batch$children[[1]])
  batch <- NULL
  gc()
  Sys.sleep(0.1)

  # The exported array should still be valid
  expect_null(array_from_column$Validate())
})

test_that("Array to nanoarrow_array works", {
  skip_if_not_installed("arrow")

  int <- arrow::Array$create(1:5)
  int_array <- as_nanoarrow_array(int)
  expect_s3_class(int_array, "nanoarrow_array")
  int_schema <- infer_nanoarrow_schema(int_array)
  expect_s3_class(int_schema, "nanoarrow_schema")

  expect_true(
    arrow::as_arrow_array(int_array)$Equals(
      arrow::Array$create(1:5)
    )
  )

  dbl_array <- as_nanoarrow_array(int, schema = arrow::float64())
  expect_s3_class(dbl_array, "nanoarrow_array")
  dbl_schema <- infer_nanoarrow_schema(dbl_array)
  expect_s3_class(dbl_schema, "nanoarrow_schema")

  expect_true(
    arrow::as_arrow_array(dbl_array)$Equals(
      arrow::Array$create(1:5, type = arrow::float64())
    )
  )
})

test_that("nanoarrow_array to ChunkedArray works", {
  skip_if_not_installed("arrow")

  int <- arrow::as_chunked_array(as_nanoarrow_array(1:5))
  expect_true(int$Equals(arrow::ChunkedArray$create(1:5)))

  dbl_casted <- arrow::as_chunked_array(as_nanoarrow_array(1:5), type = arrow::float64())
  expect_true(dbl_casted$Equals(arrow::ChunkedArray$create(1:5, type = arrow::float64())))
})

test_that("ChunkedArray to nanoarrow_array works", {
  skip_if_not_installed("arrow")

  int <- arrow::ChunkedArray$create(1:5)
  int_array <- as_nanoarrow_array(int)
  expect_s3_class(int_array, "nanoarrow_array")
  int_schema <- infer_nanoarrow_schema(int_array)
  expect_s3_class(int_schema, "nanoarrow_schema")

  expect_true(
    arrow::as_chunked_array(int_array)$Equals(
      arrow::ChunkedArray$create(1:5)
    )
  )

  dbl_array <- as_nanoarrow_array(int, schema = arrow::float64())
  expect_s3_class(dbl_array, "nanoarrow_array")
  dbl_schema <- infer_nanoarrow_schema(dbl_array)
  expect_s3_class(dbl_schema, "nanoarrow_schema")

  expect_true(
    arrow::as_chunked_array(dbl_array)$Equals(
      arrow::ChunkedArray$create(1:5, type = arrow::float64())
    )
  )
})

test_that("ChunkedArray to nanoarrow_array_stream works", {
  skip_if_not_installed("arrow")

  int <- arrow::ChunkedArray$create(1:5)
  int_array_stream <- as_nanoarrow_array_stream(int)
  expect_s3_class(int_array_stream, "nanoarrow_array_stream")

  expect_true(
    arrow::as_chunked_array(int_array_stream)$Equals(
      arrow::ChunkedArray$create(1:5)
    )
  )

  dbl_array_stream <- as_nanoarrow_array_stream(int, schema = arrow::float64())
  expect_s3_class(dbl_array_stream, "nanoarrow_array_stream")

  expect_true(
    arrow::as_chunked_array(dbl_array_stream)$Equals(
      arrow::ChunkedArray$create(1:5, type = arrow::float64())
    )
  )
})

test_that("Array to nanoarrow_array_stream works", {
  skip_if_not_installed("arrow")

  int <- arrow::Array$create(1:5)
  int_array_stream <- as_nanoarrow_array_stream(int)
  expect_s3_class(int_array_stream, "nanoarrow_array_stream")

  expect_true(
    arrow::as_arrow_array(int_array_stream)$Equals(
      arrow::Array$create(1:5)
    )
  )

  dbl_array_stream <- as_nanoarrow_array_stream(int, schema = arrow::float64())
  expect_s3_class(dbl_array_stream, "nanoarrow_array_stream")

  expect_true(
    arrow::as_arrow_array(dbl_array_stream)$Equals(
      arrow::Array$create(1:5, type = arrow::float64())
    )
  )

  empty_array_stream <- basic_array_stream(list(), na_int32())
  expect_true(
    arrow::as_arrow_array(empty_array_stream)$Equals(
      arrow::concat_arrays(type = arrow::int32())
    )
  )
})

test_that("nanoarrow_array to RecordBatch works", {
  skip_if_not_installed("arrow")

  df <- data.frame(a = 1:5, b = letters[1:5])
  batch <- arrow::as_record_batch(as_nanoarrow_array(df))
  expect_true(
    batch$Equals(arrow::record_batch(a = 1:5, b = letters[1:5]))
  )

  batch_casted <- arrow::as_record_batch(
    as_nanoarrow_array(df),
    schema = arrow::schema(a = arrow::float64(), b = arrow::string())
  )
  expect_true(
    batch_casted$Equals(
      arrow::record_batch(a = as.double(1:5), b = letters[1:5])
    )
  )
})

test_that("RecordBatch to nanoarrow_array", {
  skip_if_not_installed("arrow")

  batch <- arrow::record_batch(a = 1:5, b = letters[1:5])
  struct_array <- as_nanoarrow_array(batch)
  expect_s3_class(struct_array, "nanoarrow_array")
  struct_schema <- infer_nanoarrow_schema(struct_array)
  expect_s3_class(struct_schema, "nanoarrow_schema")

  expect_true(
    arrow::as_record_batch(struct_array)$Equals(
      arrow::record_batch(a = 1:5, b = letters[1:5])
    )
  )

  struct_array_casted <- as_nanoarrow_array(
    batch,
    schema = arrow::schema(a = arrow::float64(), b = arrow::string())
  )
  expect_s3_class(struct_array_casted, "nanoarrow_array")
  struct_schema_casted <- infer_nanoarrow_schema(struct_array_casted)
  expect_s3_class(struct_schema_casted, "nanoarrow_schema")

  expect_true(
    arrow::as_record_batch(struct_array_casted)$Equals(
      arrow::record_batch(a = as.double(1:5), b = letters[1:5])
    )
  )
})

test_that("nanoarrow_array to Table works", {
  skip_if_not_installed("arrow")

  df <- data.frame(a = 1:5, b = letters[1:5])
  table <- arrow::as_arrow_table(as_nanoarrow_array(df))
  expect_true(
    table$Equals(arrow::arrow_table(a = 1:5, b = letters[1:5]))
  )

  table_casted <- arrow::as_arrow_table(
    as_nanoarrow_array(df),
    schema = arrow::schema(a = arrow::float64(), b = arrow::string())
  )
  expect_true(
    table_casted$Equals(
      arrow::arrow_table(a = as.double(1:5), b = letters[1:5])
    )
  )
})

test_that("Table to nanoarrow_array", {
  skip_if_not_installed("arrow")

  table <- arrow::arrow_table(a = 1:5, b = letters[1:5])
  struct_array <- as_nanoarrow_array(table)
  expect_s3_class(struct_array, "nanoarrow_array")
  struct_schema <- infer_nanoarrow_schema(struct_array)
  expect_s3_class(struct_schema, "nanoarrow_schema")

  expect_true(
    arrow::as_arrow_table(struct_array)$Equals(
      arrow::arrow_table(a = 1:5, b = letters[1:5])
    )
  )

  struct_array_casted <- as_nanoarrow_array(
    table,
    schema = arrow::schema(a = arrow::float64(), b = arrow::string())
  )
  expect_s3_class(struct_array_casted, "nanoarrow_array")
  struct_schema_casted <- infer_nanoarrow_schema(struct_array_casted)
  expect_s3_class(struct_schema_casted, "nanoarrow_schema")

  expect_true(
    arrow::as_arrow_table(struct_array_casted)$Equals(
      arrow::arrow_table(a = as.double(1:5), b = letters[1:5])
    )
  )
})

test_that("Table to nanoarrow_array_stream works", {
  skip_if_not_installed("arrow")

  table <- arrow::arrow_table(a = 1:5, b = letters[1:5])
  stream <- as_nanoarrow_array_stream(table)
  expect_s3_class(stream, "nanoarrow_array_stream")

  expect_true(
    arrow::as_arrow_table(stream)$Equals(
      arrow::arrow_table(a = 1:5, b = letters[1:5])
    )
  )

  # Check cast in the stream -> table direction
  stream <- as_nanoarrow_array_stream(table)
  expect_true(
    arrow::as_arrow_table(
      stream,
      schema = arrow::schema(a = arrow::float64(), b = arrow::string())
    )$Equals(
      arrow::arrow_table(a = as.double(1:5), b = letters[1:5])
    )
  )

  # Check cast in the table -> stream direction
  stream_casted <- as_nanoarrow_array_stream(
    table,
    schema = arrow::schema(a = arrow::float64(), b = arrow::string())
  )
  expect_s3_class(stream_casted, "nanoarrow_array_stream")

  expect_true(
    arrow::as_arrow_table(stream_casted)$Equals(
      arrow::arrow_table(a = as.double(1:5), b = letters[1:5])
    )
  )
})

test_that("Dataset to nanoarrow_array_stream works", {
  skip_if_not_installed("arrow")
  skip_if_not(arrow::arrow_info()$capabilities["dataset"])

  dataset <- arrow::InMemoryDataset$create(arrow::arrow_table(a = 1:5, b = letters[1:5]))
  stream <- as_nanoarrow_array_stream(dataset)
  expect_s3_class(stream, "nanoarrow_array_stream")

  expect_true(
    arrow::as_arrow_table(stream)$Equals(
      arrow::arrow_table(a = 1:5, b = letters[1:5])
    )
  )
})

test_that("Scanner to nanoarrow_array_stream works", {
  skip_if_not_installed("arrow")
  skip_if_not(arrow::arrow_info()$capabilities["dataset"])

  dataset <- arrow::InMemoryDataset$create(arrow::arrow_table(a = 1:5, b = letters[1:5]))
  scanner <- arrow::Scanner$create(dataset)
  stream <- as_nanoarrow_array_stream(scanner)
  expect_s3_class(stream, "nanoarrow_array_stream")

  expect_true(
    arrow::as_arrow_table(stream)$Equals(
      arrow::arrow_table(a = 1:5, b = letters[1:5])
    )
  )
})

test_that("nanoarrow_schema to DataType works", {
  skip_if_not_installed("arrow")

  int_schema <- as_nanoarrow_schema(arrow::int32())
  arrow_type <- arrow::as_data_type(int_schema)
  expect_true(arrow_type$Equals(arrow::int32()))
})

test_that("DataType to nanoarrow_schema", {
  skip_if_not_installed("arrow")

  schema <- as_nanoarrow_schema(arrow::int32())
  expect_s3_class(schema, "nanoarrow_schema")
  expect_true(arrow::as_data_type(schema)$Equals(arrow::int32()))
})

test_that("Field to nanoarrow_schema", {
  skip_if_not_installed("arrow")

  schema <- as_nanoarrow_schema(arrow::field("name", arrow::int32()))
  expect_s3_class(schema, "nanoarrow_schema")
  expect_true(arrow::as_data_type(schema)$Equals(arrow::int32()))
})

test_that("nanoarrow_schema to Schema works", {
  skip_if_not_installed("arrow")

  struct_schema <- as_nanoarrow_schema(
    arrow::struct(a = arrow::int32(), b = arrow::string())
  )
  arrow_schema <- arrow::as_schema(struct_schema)
  expect_true(arrow_schema$Equals(arrow::schema(a = arrow::int32(), b = arrow::string())))
})

test_that("Schema to nanoarrow_schema", {
  skip_if_not_installed("arrow")

  schema <- as_nanoarrow_schema(arrow::schema(name = arrow::int32()))
  expect_s3_class(schema, "nanoarrow_schema")
  expect_true(arrow::as_schema(schema)$Equals(arrow::schema(name = arrow::int32())))
})

test_that("nanoarrow_array_stream to RecordBatchReader works", {
  skip_if_not_installed("arrow")

  reader <- arrow::as_record_batch_reader(
    arrow::record_batch(a = 1:5, b = letters[1:5])
  )
  array_stream <- as_nanoarrow_array_stream(reader)

  reader_roundtrip <- arrow::as_record_batch_reader(array_stream)
  expect_false(nanoarrow_pointer_is_valid(array_stream))
  expect_true(
    reader_roundtrip$read_next_batch()$Equals(
      arrow::record_batch(a = 1:5, b = letters[1:5])
    )
  )
  expect_null(reader_roundtrip$read_next_batch())
})

test_that("RecordBatchReader to nanoarrow_array_stream works", {
  skip_if_not_installed("arrow")

  reader <- arrow::as_record_batch_reader(
    arrow::record_batch(a = 1:5, b = letters[1:5])
  )
  array_stream <- as_nanoarrow_array_stream(reader)
  expect_s3_class(array_stream, "nanoarrow_array_stream")

  reader_roundtrip <- arrow::as_record_batch_reader(array_stream)
  expect_true(
    reader_roundtrip$read_next_batch()$Equals(
      arrow::record_batch(a = 1:5, b = letters[1:5])
    )
  )
  expect_null(reader_roundtrip$read_next_batch())
})
