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

#include <stdio.h>

#include <benchmark/benchmark.h>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_ipc.hpp>

static ArrowErrorCode MakeFixtureInputStreamFile(const std::string& fixture_name,
                                                 ArrowIpcInputStream* out) {
  const char* fixture_dir = std::getenv("NANOARROW_BENCHMARK_FIXTURE_DIR");
  if (fixture_dir == NULL) {
    fixture_dir = "fixtures";
  }

  std::string fixture_path = std::string(fixture_dir) + std::string("/") + fixture_name;
  FILE* fixture_file = fopen(fixture_path.c_str(), "rb");

  NANOARROW_RETURN_NOT_OK(
      ArrowIpcInputStreamInitFile(out, fixture_file, /*close_on_release*/ true));
  return NANOARROW_OK;
}

static ArrowErrorCode MakeFixtureBuffer(const std::string& fixture_name,
                                        ArrowBuffer* out) {
  nanoarrow::ipc::UniqueInputStream input_stream;
  NANOARROW_RETURN_NOT_OK(MakeFixtureInputStreamFile(fixture_name, input_stream.get()));

  nanoarrow::UniqueBuffer buffer;
  int64_t size_read_out = 0;
  int64_t chunk_size = 1024;
  do {
    NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(buffer.get(), chunk_size));
    NANOARROW_RETURN_NOT_OK(input_stream->read(input_stream.get(),
                                               buffer->data + buffer->size_bytes,
                                               chunk_size, &size_read_out, nullptr));
    buffer->size_bytes += size_read_out;
  } while (size_read_out > 0);

  ArrowBufferMove(buffer.get(), out);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrayStreamReadAll(ArrowArrayStream* array_stream,
                                         int64_t* batch_count, int64_t* column_count) {
  nanoarrow::UniqueSchema schema;
  NANOARROW_RETURN_NOT_OK(array_stream->get_schema(array_stream, schema.get()));
  *column_count = schema->n_children;
  benchmark::DoNotOptimize(schema);

  nanoarrow::UniqueArrayView array_view;
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr));

  while (true) {
    nanoarrow::UniqueArray array;
    NANOARROW_RETURN_NOT_OK(array_stream->get_next(array_stream, array.get()));
    if (array->release == nullptr) {
      break;
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr));

    *batch_count = *batch_count + 1;
  }

  return NANOARROW_OK;
}

/// \defgroup nanoarrow-benchmark-ipc IPC Reader Benchmarks
///
/// Benchmarks for the ArrowArrayStream IPC reader.
///
/// @{

static void BaseBenchmarIpcFixtureBuffer(const std::string& fixture_name,
                                         benchmark::State& state) {
  int64_t batch_count = 0;
  int64_t column_count = 0;

  nanoarrow::UniqueBuffer buffer;
  NANOARROW_THROW_NOT_OK(MakeFixtureBuffer(fixture_name, buffer.get()));

  for (auto _ : state) {
    // Don't copy the buffer within the benchmarking loop
    nanoarrow::UniqueBuffer buffer_copy;
    NANOARROW_THROW_NOT_OK(ArrowBufferSetAllocator(
        buffer_copy.get(),
        ArrowBufferDeallocator([](ArrowBufferAllocator*, uint8_t*, int64_t) -> void {},
                               nullptr)));
    buffer_copy->data = buffer->data;
    buffer_copy->size_bytes = buffer->size_bytes;

    nanoarrow::ipc::UniqueInputStream input_stream;
    NANOARROW_THROW_NOT_OK(
        ArrowIpcInputStreamInitBuffer(input_stream.get(), buffer_copy.get()));

    nanoarrow::UniqueArrayStream array_stream;
    NANOARROW_THROW_NOT_OK(
        ArrowIpcArrayStreamReaderInit(array_stream.get(), input_stream.get(), nullptr));

    NANOARROW_THROW_NOT_OK(
        ArrayStreamReadAll(array_stream.get(), &batch_count, &column_count));

    benchmark::DoNotOptimize(batch_count);
  }

  state.SetBytesProcessed(state.iterations() * buffer->size_bytes);
}

/// \brief Use the ArrowArrayStream IPC reader to read a ~10 MB stream with 10
/// float64 columns.
static void BenchmarkIpcReadFloat64FromBuffer(benchmark::State& state) {
  BaseBenchmarIpcFixtureBuffer("float64_basic.arrows", state);
}

/// \brief Use the ArrowArrayStream IPC reader to read a ~10 MB stream with 1
/// float64 column.
static void BenchmarkIpcReadFloat64LongFromBuffer(benchmark::State& state) {
  BaseBenchmarIpcFixtureBuffer("float64_long.arrows", state);
}

/// \brief Use the ArrowArrayStream IPC reader to read a ~10 MB stream with 1280
/// float64 columns.
static void BenchmarkIpcReadFloat64WideFromBuffer(benchmark::State& state) {
  BaseBenchmarIpcFixtureBuffer("float64_wide.arrows", state);
}

BENCHMARK(BenchmarkIpcReadFloat64FromBuffer);
BENCHMARK(BenchmarkIpcReadFloat64LongFromBuffer);
BENCHMARK(BenchmarkIpcReadFloat64WideFromBuffer);

/// @}
