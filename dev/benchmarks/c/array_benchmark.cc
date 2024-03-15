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

/// \defgroup nanoarrow-benchmark-array-view ArrowArrayView-related benchmarks
///
/// Benchmarks for consuming ArrowArrays using the ArrowArrayViewXXX() functions.
///
/// @{

template <typename Buffer1T, typename Buffer2T = int8_t>
ArrowErrorCode InitArrayViewFromBuffers(ArrowType type, ArrowArray* array,
                                        ArrowArrayView* array_view,
                                        std::vector<int8_t> validity,
                                        std::vector<Buffer1T> buffer1,
                                        std::vector<Buffer2T> buffer2 = {}) {
  // Initialize arrays
  nanoarrow::UniqueSchema schema;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema.get(), type));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromSchema(array, schema.get(), nullptr));

  // Initialize buffers
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(ArrowArrayBuffer(array, 1), buffer1.data(),
                                            buffer1.size() * sizeof(Buffer1T)));
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(ArrowArrayBuffer(array, 2), buffer2.data(),
                                            buffer2.size() * sizeof(Buffer2T)));

  // Pack the validity bitmap
  if (validity.size() > 0) {
    ArrowBitmap* validity_bitmap = ArrowArrayValidityBitmap(array);
    NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(validity_bitmap, validity.size()));
    ArrowBitmapAppendInt8Unsafe(validity_bitmap, validity.data(), validity.size());
  }

  // Set the length
  switch (type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (buffer1.size() > 0) {
        array->length = buffer1.size() - 1;
      } else {
        array->length = 0;
      }
      break;

    default:
      array->length = buffer1.size();
      break;
  }

  // Set the null count
  if (validity.size() > 0) {
    array->null_count = array->length - ArrowBitCountSet(ArrowArrayBuffer(array, 0)->data,
                                                         0, array->length);
  } else {
    array->null_count = 0;
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuildingDefault(array, nullptr));
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(array_view, schema.get(), nullptr));
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArray(array_view, array, nullptr));
  return NANOARROW_OK;
}

template <typename CType, ArrowType type>
static void BaseArrayViewGet(benchmark::State& state) {
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;

  int64_t n_values = 1000000;

  std::vector<CType> values(n_values);
  for (int64_t i = 0; i < n_values; i++) {
    values[i] = i % std::numeric_limits<CType>::max();
  }

  NANOARROW_THROW_NOT_OK(
      InitArrayViewFromBuffers(type, array.get(), array_view.get(), {}, values));

  std::vector<CType> values_out(n_values);
  for (auto _ : state) {
    for (int64_t i = 0; i < n_values; i++) {
      values_out[i] = ArrowArrayViewGetIntUnsafe(array_view.get(), i);
    }
    benchmark::DoNotOptimize(values_out);
  }

  state.SetItemsProcessed(n_values * state.iterations());
}

/// \brief Use ArrowArrayViewGet() to consume an int8 array
static void BenchmarkArrayViewGetInt8(benchmark::State& state) {
  BaseArrayViewGet<int8_t, NANOARROW_TYPE_INT8>(state);
}

/// \brief Use ArrowArrayViewGet() to consume an int16 array
static void BenchmarkArrayViewGetInt16(benchmark::State& state) {
  BaseArrayViewGet<int16_t, NANOARROW_TYPE_INT16>(state);
}

/// \brief Use ArrowArrayViewGet() to consume an int32 array
static void BenchmarkArrayViewGetInt32(benchmark::State& state) {
  BaseArrayViewGet<int32_t, NANOARROW_TYPE_INT32>(state);
}

/// \brief Use ArrowArrayViewGet() to consume an int64 array
static void BenchmarkArrayViewGetInt64(benchmark::State& state) {
  BaseArrayViewGet<int64_t, NANOARROW_TYPE_INT64>(state);
}

/// \brief Use ArrowArrayViewIsNull() to check for nulls while consuming an int32 array
/// that does not contain a validity buffer.
static void BenchmarkArrayViewIsNullNonNullable(benchmark::State& state) {
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;

  int64_t n_values = 1000000;

  // Create values
  std::vector<int32_t> values(n_values);
  for (int64_t i = 0; i < n_values; i++) {
    values[i] = i % 1000;
  }

  NANOARROW_THROW_NOT_OK(InitArrayViewFromBuffers(NANOARROW_TYPE_INT32, array.get(),
                                                  array_view.get(), {}, values));

  // Read the array
  std::vector<int32_t> values_out(n_values);
  for (auto _ : state) {
    for (int64_t i = 0; i < n_values; i++) {
      if (ArrowArrayViewIsNull(array_view.get(), i)) {
        values_out[i] = 0;
      } else {
        values_out[i] = ArrowArrayViewGetIntUnsafe(array_view.get(), i);
      }
    }
    benchmark::DoNotOptimize(values_out);
  }

  state.SetItemsProcessed(n_values * state.iterations());
}

/// \brief Use ArrowArrayViewIsNull() to check for nulls while consuming an int32 array
/// that contains nulls.
static void BenchmarkArrayViewIsNull(benchmark::State& state) {
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;

  int64_t n_values = 1000000;

  // Create values
  std::vector<int32_t> values(n_values);
  for (int64_t i = 0; i < n_values; i++) {
    values[i] = i % 1000;
  }

  // Create validity buffer
  double prop_null = 0.2;
  int64_t num_nulls = n_values * prop_null;
  int64_t null_spacing = n_values / num_nulls;

  std::vector<int8_t> validity(n_values);
  for (int64_t i = 0; i < n_values; i++) {
    validity[i] = i % null_spacing != 0;
  }

  NANOARROW_THROW_NOT_OK(InitArrayViewFromBuffers(NANOARROW_TYPE_INT32, array.get(),
                                                  array_view.get(), validity, values));

  // Read the array
  std::vector<int32_t> values_out(n_values);
  for (auto _ : state) {
    for (int64_t i = 0; i < n_values; i++) {
      if (ArrowArrayViewIsNull(array_view.get(), i)) {
        values_out[i] = 0;
      } else {
        values_out[i] = ArrowArrayViewGetIntUnsafe(array_view.get(), i);
      }
    }
    benchmark::DoNotOptimize(values_out);
  }

  state.SetItemsProcessed(n_values * state.iterations());
}

static void BenchmarkArrayViewGetString(benchmark::State& state) {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;

  // Create a large array with intentionally tiny strings (to maximize overhead)
  int64_t n_values = 1000000;
  int64_t value_size = 7;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

  std::vector<int32_t> offsets(n_values + 1);
  for (int64_t i = 0; i < n_values; i++) {
    offsets[i + 1] = i * value_size;
  }

  int64_t n_alphabets = n_values / alphabet.size() + 1;
  std::vector<char> data(alphabet.size() * n_alphabets);
  for (int64_t data_pos = 0; data_pos < data.size(); data_pos += alphabet.size()) {
    memcpy(data.data() + data_pos, alphabet.data(), alphabet.size());
  }

  // Read the array as non-copying views. Possibly less realistic than
  // what somebody might actually do, but also is a more direct benchmark
  // of the overhead associated with calling it.
  std::vector<ArrowStringView> values_out(n_values);
  for (auto _ : state) {
    for (int64_t i = 0; i < n_values; i++) {
      values_out[i] = ArrowArrayViewGetStringUnsafe(array_view.get(), i);
    }
    benchmark::DoNotOptimize(values_out);
  }
  state.SetItemsProcessed(n_values * state.iterations());
}

BENCHMARK(BenchmarkArrayViewGetInt8);
BENCHMARK(BenchmarkArrayViewGetInt16);
BENCHMARK(BenchmarkArrayViewGetInt32);
BENCHMARK(BenchmarkArrayViewGetInt64);
BENCHMARK(BenchmarkArrayViewGetString);
BENCHMARK(BenchmarkArrayViewIsNullNonNullable);
BENCHMARK(BenchmarkArrayViewIsNull);
