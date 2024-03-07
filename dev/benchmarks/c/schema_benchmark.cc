// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <benchmark/benchmark.h>

#include <nanoarrow/nanoarrow.hpp>

/// \defgroup nanoarrow-benchmark-schema Schema-related benchmarks
///
/// Benchmarks for producing and consuming ArrowSchema.
///
/// @{

// Utility to initialize a wide struct schema
static ArrowErrorCode SchemaInitStruct(struct ArrowSchema* schema, int64_t n_columns) {
  ArrowSchemaInit(schema);
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, n_columns));
  for (int64_t i = 0; i < n_columns; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaSetType(schema->children[i], NANOARROW_TYPE_INT32));
  }
  return NANOARROW_OK;
}

/// \brief Benchmark ArrowSchema creation for very wide tables
///
/// Simulates part of the process of creating a very wide table with a
/// simple column type (integer).
static void BenchmarkSchemaInitWideStruct(benchmark::State& state);

static void BenchmarkSchemaInitWideStruct(benchmark::State& state) {
  struct ArrowSchema schema;

  int64_t n_columns = 10000;

  for (auto _ : state) {
    NANOARROW_THROW_NOT_OK(SchemaInitStruct(&schema, 10000));
    ArrowSchemaRelease(&schema);
  }

  state.SetItemsProcessed(n_columns * state.iterations());
}

BENCHMARK(BenchmarkSchemaInitWideStruct);

/// \brief Benchmark ArrowSchema parsing for very wide tables
///
/// Simulates part of the process of consuming a very wide table. Typically
/// the ArrowSchemaViewInit() is done by ArrowArrayViewInit() but uses a
/// similar pattern.
static void BenchmarkSchemaViewInitWideStruct(benchmark::State& state);

static ArrowErrorCode SchemaViewInitChildren(struct ArrowSchema* schema,
                                             struct ArrowError* error) {
  for (int64_t i = 0; i < schema->n_children; i++) {
    struct ArrowSchemaView schema_view;
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaViewInit(&schema_view, schema->children[i], error));
  }

  return NANOARROW_OK;
}

static void BenchmarkSchemaViewInitWideStruct(benchmark::State& state) {
  struct ArrowSchema schema;
  struct ArrowError error;

  int64_t n_columns = 10000;
  SchemaInitStruct(&schema, n_columns);

  for (auto _ : state) {
    NANOARROW_ASSERT_OK(SchemaViewInitChildren(&schema, &error));
  }
  state.SetItemsProcessed(n_columns * state.iterations());

  ArrowSchemaRelease(&schema);
}

BENCHMARK(BenchmarkSchemaViewInitWideStruct);

/// @}
