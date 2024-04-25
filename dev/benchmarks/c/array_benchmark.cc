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

// The length of most arrays used in these benchmarks. Just big enough so
// that the benchmark takes a non-trivial amount of time to run.
static const int64_t kNumItemsPrettyBig = 1000000;

// Used to generate string/binary arrays
static const std::string kAlphabet =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// \defgroup nanoarrow-benchmark-array-view ArrowArrayView-related benchmarks
///
/// Benchmarks for consuming ArrowArrays using the `ArrowArrayViewXXX()` functions.
///
/// @{

// Helper to initialize an ArrowArrayView from buffers. The ArrowArray isn't used
// by the benchmark but is needed to hold the memory.
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
static void BaseArrayViewGetInt(benchmark::State& state) {
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;

  int64_t n_values = kNumItemsPrettyBig;

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
  BaseArrayViewGetInt<int8_t, NANOARROW_TYPE_INT8>(state);
}

/// \brief Use ArrowArrayViewGet() to consume an int16 array
static void BenchmarkArrayViewGetInt16(benchmark::State& state) {
  BaseArrayViewGetInt<int16_t, NANOARROW_TYPE_INT16>(state);
}

/// \brief Use ArrowArrayViewGet() to consume an int32 array
static void BenchmarkArrayViewGetInt32(benchmark::State& state) {
  BaseArrayViewGetInt<int32_t, NANOARROW_TYPE_INT32>(state);
}

/// \brief Use ArrowArrayViewGet() to consume an int64 array
static void BenchmarkArrayViewGetInt64(benchmark::State& state) {
  BaseArrayViewGetInt<int64_t, NANOARROW_TYPE_INT64>(state);
}

/// \brief Use ArrowArrayViewIsNull() to check for nulls while consuming an int32 array
/// that does not contain a validity buffer.
static void BenchmarkArrayViewIsNullNonNullable(benchmark::State& state) {
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;

  int64_t n_values = kNumItemsPrettyBig;

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
/// that contains 20% nulls.
static void BenchmarkArrayViewIsNull(benchmark::State& state) {
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;

  int64_t n_values = kNumItemsPrettyBig;

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

/// \brief Use ArrowArrayViewGetStringUnsafe() to consume a string array
static void BenchmarkArrayViewGetString(benchmark::State& state) {
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;

  // Create an array of relatively small strings
  int64_t n_values = kNumItemsPrettyBig;
  int64_t value_size = 7;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

  std::vector<int32_t> offsets(n_values + 1);
  for (int64_t i = 0; i < n_values; i++) {
    offsets[i + 1] = i * value_size;
  }

  int64_t n_alphabets = n_values / alphabet.size() + 1;
  std::vector<char> data(alphabet.size() * n_alphabets);
  for (size_t data_pos = 0; data_pos < data.size(); data_pos += alphabet.size()) {
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

/// @}

/// \defgroup nanoarrow-benchmark-array ArrowArray-related benchmarks
///
/// Benchmarks for producing ArrowArrays using the `ArrowArrayXXX()` functions.
///
/// @{

template <typename CType, ArrowType type>
static ArrowErrorCode CreateAndAppendToArrayInt(ArrowArray* array,
                                                const std::vector<CType>& values) {
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, type));
  NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array));

  for (size_t i = 0; i < values.size(); i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, values[i]));
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuildingDefault(array, nullptr));
  return NANOARROW_OK;
}

template <ArrowType type>
static ArrowErrorCode CreateAndAppendToArrayString(
    ArrowArray* array, const std::vector<std::string>& values) {
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, type));
  NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array));

  for (const std::string& s : values) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayAppendString(array, {s.data(), static_cast<int64_t>(s.size())}));
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuildingDefault(array, nullptr));
  return NANOARROW_OK;
}

/// \brief Use ArrowArrayAppendString() to build a string array
static void BenchmarkArrayAppendString(benchmark::State& state) {
  nanoarrow::UniqueArray array;

  int64_t n_values = kNumItemsPrettyBig;
  int64_t value_size = 7;

  std::vector<std::string> values(n_values);
  size_t alphabet_pos = 0;
  for (std::string& value : values) {
    if ((alphabet_pos + value_size) >= kAlphabet.size()) {
      alphabet_pos = 0;
    }

    value.assign(kAlphabet.data() + alphabet_pos, value_size);
    alphabet_pos += value_size;
  }

  for (auto _ : state) {
    array.reset();
    NANOARROW_THROW_NOT_OK(
        CreateAndAppendToArrayString<NANOARROW_TYPE_STRING>(array.get(), values));
    benchmark::DoNotOptimize(array);
  }

  state.SetItemsProcessed(n_values * state.iterations());
}

template <typename CType, ArrowType type>
static void BaseBenchmarkArrayAppendInt(benchmark::State& state) {
  nanoarrow::UniqueArray array;

  int64_t n_values = kNumItemsPrettyBig;

  std::vector<CType> values(n_values);
  for (int64_t i = 0; i < n_values; i++) {
    values[i] = i % std::numeric_limits<CType>::max();
  }

  for (auto _ : state) {
    array.reset();
    int code = CreateAndAppendToArrayInt<CType, type>(array.get(), values);
    NANOARROW_THROW_NOT_OK(code);
    benchmark::DoNotOptimize(array);
  }

  state.SetItemsProcessed(n_values * state.iterations());
}

/// \brief Use ArrowArrayAppendInt() to build an int8 array
static void BenchmarkArrayAppendInt8(benchmark::State& state) {
  BaseBenchmarkArrayAppendInt<int8_t, NANOARROW_TYPE_INT8>(state);
}

/// \brief Use ArrowArrayAppendInt() to build an int16 array
static void BenchmarkArrayAppendInt16(benchmark::State& state) {
  BaseBenchmarkArrayAppendInt<int16_t, NANOARROW_TYPE_INT16>(state);
}

/// \brief Use ArrowArrayAppendInt() to build an int32 array
static void BenchmarkArrayAppendInt32(benchmark::State& state) {
  BaseBenchmarkArrayAppendInt<int32_t, NANOARROW_TYPE_INT32>(state);
}

/// \brief Use ArrowArrayAppendInt() to build an int64 array
static void BenchmarkArrayAppendInt64(benchmark::State& state) {
  BaseBenchmarkArrayAppendInt<int64_t, NANOARROW_TYPE_INT64>(state);
}

template <typename CType, ArrowType type>
static ArrowErrorCode CreateAndAppendIntWithNulls(ArrowArray* array,
                                                  const std::vector<int8_t>& validity) {
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, type));
  NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array));
  CType non_null_value = std::numeric_limits<CType>::max() / 2;

  for (size_t i = 0; i < validity.size(); i++) {
    if (validity[i]) {
      NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, non_null_value));
    } else {
      NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 1));
    }
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuildingDefault(array, nullptr));
  return NANOARROW_OK;
}

/// \brief Use ArrowArrayAppendNulls() to build an int32 array that contains 80%
/// null values
static void BenchmarkArrayAppendNulls(benchmark::State& state) {
  nanoarrow::UniqueArray array;

  int64_t n_values = kNumItemsPrettyBig;
  double prop_null = 0.8;
  int64_t num_nulls = n_values * prop_null;
  int64_t null_spacing = n_values / num_nulls;

  std::vector<int8_t> validity(n_values);
  for (int64_t i = 0; i < n_values; i++) {
    validity[i] = i % null_spacing != 0;
  }

  for (auto _ : state) {
    array.reset();
    int code =
        CreateAndAppendIntWithNulls<int32_t, NANOARROW_TYPE_INT32>(array.get(), validity);
    NANOARROW_THROW_NOT_OK(code);
    benchmark::DoNotOptimize(array);
  }

  state.SetItemsProcessed(n_values * state.iterations());
}

/// @}

BENCHMARK(BenchmarkArrayViewGetInt8);
BENCHMARK(BenchmarkArrayViewGetInt16);
BENCHMARK(BenchmarkArrayViewGetInt32);
BENCHMARK(BenchmarkArrayViewGetInt64);
BENCHMARK(BenchmarkArrayViewGetString);
BENCHMARK(BenchmarkArrayViewIsNullNonNullable);
BENCHMARK(BenchmarkArrayViewIsNull);

BENCHMARK(BenchmarkArrayAppendString);
BENCHMARK(BenchmarkArrayAppendInt8);
BENCHMARK(BenchmarkArrayAppendInt16);
BENCHMARK(BenchmarkArrayAppendInt32);
BENCHMARK(BenchmarkArrayAppendInt64);
BENCHMARK(BenchmarkArrayAppendNulls);
