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

test_that("new_data_frame() works", {
  expect_identical(
    new_data_frame(list(x = 1, y = 2), nrow = 1),
    data.frame(x = 1, y = 2)
  )
})

test_that("vector fuzzers work", {
  ptype <- data.frame(a = logical(), b = integer(), c = double(), d = character())
  df_gen <- vec_gen(ptype, n = 123)

  expect_identical(nrow(df_gen), 123L)
  expect_identical(df_gen[integer(), ], ptype)

  expect_error(vec_gen(environment()), "Don't know how to generate vector")
})

test_that("vector shuffler works", {
  df <- data.frame(letters = letters)
  df_shuffled <- vec_shuffle(df)
  expect_setequal(df_shuffled$letters, df$letters)

  letters_shuffled <- vec_shuffle(letters)
  expect_setequal(letters_shuffled, letters)
})
