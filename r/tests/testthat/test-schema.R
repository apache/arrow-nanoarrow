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

test_that("as_nanoarrow_schema() works for nanoarrow_schema", {
  schema <- infer_nanoarrow_schema(1:10)
  expect_identical(as_nanoarrow_schema(schema), schema)
})

test_that("infer_nanoarrow_schema() default method works", {
  schema <- infer_nanoarrow_schema(1:10)
  expect_true(arrow::as_data_type(schema)$Equals(arrow::int32()))
})
