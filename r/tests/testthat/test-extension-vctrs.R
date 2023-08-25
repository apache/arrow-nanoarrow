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

test_that("vctrs extension type can roundtrip built-in vector types", {
  skip_if_not_installed("tibble")

  # list()s aren't implemented here yet
  vectors <- list(
    lgl = c(FALSE, TRUE, NA),
    int = c(0L, 1L, NA_integer_),
    dbl = c(0, 1, NA_real_),
    chr = c("a", NA_character_),
    posixct = as.POSIXct("2000-01-01 12:23", tz = "UTC"),
    posixlt = as.POSIXlt("2000-01-01 12:23", tz = "UTC"),
    date = as.Date("2000-01-01"),
    difftime = as.difftime(123, units = "secs"),
    data_frame_simple = data.frame(x = 1:5),
    data_frame_nested = tibble::tibble(x = 1:5, y = data.frame(z = letters[1:5]))
  )

  for (nm in names(vectors)) {
    vctr <- vectors[[nm]]
    ptype <- vctrs::vec_ptype(vctr)
    schema <- na_vctrs(vctr)

    array <- as_nanoarrow_array(vctr, schema = schema)
    array_schema <- infer_nanoarrow_schema(array)

    # Roundtrip through convert_array()
    expect_true(nanoarrow_schema_identical(array_schema, schema))
    expect_identical(infer_nanoarrow_ptype(array), ptype)
    expect_identical(convert_array(array), vctr)

    # Roundtrip with an empty array stream
    stream <- basic_array_stream(list(), schema = schema)
    expect_identical(convert_array_stream(stream), ptype)

    # Roundtrip with multiple chunks
    stream <- basic_array_stream(list(array, array))
    expect_identical(convert_array_stream(stream), vctrs::vec_rep(vctr, 2))
  }
})
