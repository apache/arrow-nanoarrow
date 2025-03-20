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

test_that("nanoarrow_altrep_chr() returns NULL for unsupported types", {
  expect_null(nanoarrow_altrep_chr(as_nanoarrow_array(1:10)))
  expect_null(nanoarrow_altrep_chr(as_nanoarrow_array(1:10)))
})

test_that("nanoarrow_altrep_chr() works for string", {
  x <- as_nanoarrow_array(c(NA, letters), schema = na_string())
  x_altrep <- nanoarrow_altrep_chr(x)

  expect_output(.Internal(inspect(x_altrep)), "<nanoarrow::altrep_chr\\[27\\]>")

  # Check that some common operations that call the string elt method
  # don't materialize the vector
  expect_identical(x_altrep, c(NA, letters))
  expect_length(x_altrep, 27)
  expect_false(is_nanoarrow_altrep_materialized(x_altrep))

  # Setting an element will materialize, duplicate, then modify
  x_altrep2 <- x_altrep
  x_altrep2[1] <- "not a letter"
  expect_identical(x_altrep2, c("not a letter", letters))
  expect_true(is_nanoarrow_altrep_materialized(x_altrep))

  # Check the same operations on the materialized output
  expect_identical(x_altrep, c(NA, letters))
  expect_length(x_altrep, 27)

  # Materialization should get printed in inspect()
  expect_output(.Internal(inspect(x_altrep)), "<materialized nanoarrow::altrep_chr\\[27\\]>")

  # For good measure, force materialization again and check
  nanoarrow_altrep_force_materialize(x_altrep)
  expect_identical(x_altrep, c(NA, letters))
  expect_length(x_altrep, 27)
})

test_that("nanoarrow_altrep_chr() works for large string", {
  x <- as_nanoarrow_array(letters, schema = na_large_string())
  x_altrep <- nanoarrow_altrep_chr(x)
  expect_identical(x_altrep, letters)
})

test_that("is_nanoarrow_altrep() returns true for nanoarrow altrep objects", {
  expect_false(is_nanoarrow_altrep("not altrep"))
  expect_false(is_nanoarrow_altrep(1:10))
  expect_true(is_nanoarrow_altrep(nanoarrow_altrep_chr(as_nanoarrow_array("whee"))))
})

test_that("nanoarrow_altrep_chr_force_materialize() forces materialization", {
  x <- as_nanoarrow_array(letters, schema = na_string())
  x_altrep <- nanoarrow_altrep_chr(x)

  expect_identical(nanoarrow_altrep_force_materialize("not altrep"), 0L)
  expect_identical(nanoarrow_altrep_force_materialize(x_altrep), 1L)

  x <- as_nanoarrow_array(letters, schema = na_string())
  x_altrep_df <- data.frame(x = nanoarrow_altrep_chr(x), stringsAsFactors = FALSE)
  expect_identical(
    nanoarrow_altrep_force_materialize(x_altrep_df, recursive = FALSE),
    0L
  )
  expect_identical(
    nanoarrow_altrep_force_materialize(x_altrep_df, recursive = TRUE),
    1L
  )
  expect_identical(
    nanoarrow_altrep_force_materialize(x_altrep_df, recursive = TRUE),
    0L
  )
})

test_that("is_nanoarrow_altrep_materialized() checks for materialization", {
  expect_identical(is_nanoarrow_altrep_materialized("not altrep"), NA)
  expect_identical(is_nanoarrow_altrep_materialized(1:10), NA)

  x <- as_nanoarrow_array(letters, schema = na_string())
  x_altrep <- nanoarrow_altrep_chr(x)
  expect_false(is_nanoarrow_altrep_materialized(x_altrep))
  expect_identical(nanoarrow_altrep_force_materialize(x_altrep), 1L)
  expect_true(is_nanoarrow_altrep_materialized(x_altrep))
})
