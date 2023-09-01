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

test_that("can set option/env var to pretend the arrow package is not installed", {
  skip_if_not_installed("arrow")

  expect_true(arrow_installed())
  expect_silent(assert_arrow_installed("life"))

  withr::with_options(list(nanoarrow.without_arrow = TRUE), {
    expect_false(arrow_installed())

    expect_error(
      assert_arrow_installed("life"),
      "Package 'arrow' required for life"
    )
  })

  withr::with_envvar(list(R_NANOARROW_WITHOUT_ARROW = "true"), {
    expect_false(arrow_installed())
  })
})

test_that("preserve/release works when release happens on another thread", {
  some_non_null_sexp <- 1L

  preserved_empty()
  expect_identical(preserved_empty(), 0)
  preserve_and_release_on_other_thread(some_non_null_sexp)
  # We can't test the exact value of preserved_count() because what the
  # garbage collector releases and when is not stable.
  expect_true(preserved_count() > 0)
  expect_identical(preserved_empty(), 1)
  expect_identical(preserved_empty(), 0)
})

test_that("vector slicer works", {
  expect_identical(vec_slice2(letters, 1), "a")
  expect_identical(
    vec_slice2(data.frame(letters = letters, stringsAsFactors = FALSE), 1),
    data.frame(letters = "a", stringsAsFactors = FALSE)
  )
})

test_that("new_data_frame() works", {
  expect_identical(
    new_data_frame(list(x = 1, y = 2), nrow = 1),
    data.frame(x = 1, y = 2)
  )
})

test_that("vector fuzzers work", {
  ptype <- data.frame(
    a = logical(),
    b = integer(),
    c = double(),
    d = character(),
    stringsAsFactors = FALSE
  )
  df_gen <- vec_gen(ptype, n = 123)

  expect_identical(nrow(df_gen), 123L)
  expect_identical(df_gen[integer(), ], ptype)

  expect_error(vec_gen(environment()), "Don't know how to generate vector")
})

test_that("vector shuffler works", {
  df <- data.frame(letters = letters, stringsAsFactors = FALSE)
  df_shuffled <- vec_shuffle(df)
  expect_setequal(df_shuffled$letters, df$letters)

  letters_shuffled <- vec_shuffle(letters)
  expect_setequal(letters_shuffled, letters)
})
