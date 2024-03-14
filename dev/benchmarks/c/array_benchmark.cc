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

// Utility for building primitive arrays
template <typename CType, ArrowType type>
ArrowErrorCode InitSchemaAndArrayPrimitive(ArrowSchema* schema, ArrowArray* array,
                                           std::vector<CType> values,
                                           std::vector<int8_t> validity = {}) {
  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, type));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromSchema(array, schema, nullptr));

  // Set the data buffer
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(ArrowArrayBuffer(array, 1), values.data(),
                                            values.size() * sizeof(CType)));

  // Pack the validity bitmap
  if (validity.size() > 0) {
    ArrowBitmap* validity_bitmap = ArrowArrayValidityBitmap(array);
    NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(validity_bitmap, validity.size()));
    ArrowBitmapAppendInt8Unsafe(validity_bitmap, validity.data(), validity.size());
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuildingDefault(array, nullptr));
  return NANOARROW_OK;
}

template <typename CType, ArrowType type>
static void BaseArrayViewGetIntUnsafe(benchmark::State& state, double prop_null = 0.0) {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;

  int64_t n_values = 1000000;

  std::vector<CType> values(n_values);
  for (int64_t i = 0; i < n_values; i++) {
    values[i] = i % std::numeric_limits<CType>::max();
  }

  std::vector<int8_t> validity;

  if (prop_null > 0) {
    int64_t num_nulls = n_values * prop_null;
    int64_t null_spacing = n_values / num_nulls;
    validity.resize(n_values);
    for (int64_t i = 0; i < n_values; i++) {
      validity[i] = i % null_spacing != 0;
    }
  }

  int code = InitSchemaAndArrayPrimitive<CType, type>(
      schema.get(), array.get(), std::move(values), std::move(validity));
  NANOARROW_THROW_NOT_OK(code);
  NANOARROW_THROW_NOT_OK(
      ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr));

  std::vector<CType> values_out(n_values);

  if (prop_null > 0) {
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
  } else {
    for (auto _ : state) {
      for (int64_t i = 0; i < n_values; i++) {
        values_out[i] = ArrowArrayViewGetIntUnsafe(array_view.get(), i);
      }
      benchmark::DoNotOptimize(values_out);
    }
  }

  state.SetItemsProcessed(n_values * state.iterations());
}

/// \brief Use ArrowArrayViewGetIntUnsafe() to consume an int8 array
static void BenchmarkArrayViewGetIntUnsafeInt8(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int8_t, NANOARROW_TYPE_INT8>(state);
}

/// \brief Use ArrowArrayViewGetIntUnsafe() to consume an int16 array
static void BenchmarkArrayViewGetIntUnsafeInt16(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int16_t, NANOARROW_TYPE_INT16>(state);
}

/// \brief Use ArrowArrayViewGetIntUnsafe() to consume an int32 array
static void BenchmarkArrayViewGetIntUnsafeInt32(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int32_t, NANOARROW_TYPE_INT32>(state);
}

/// \brief Use ArrowArrayViewGetIntUnsafe() to consume an int64 array
static void BenchmarkArrayViewGetIntUnsafeInt64(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int64_t, NANOARROW_TYPE_INT64>(state);
}

/// \brief Use ArrowArrayViewGetIntUnsafe() to consume an int64 array (checking for nulls)
static void BenchmarkArrayViewGetIntUnsafeInt64CheckNull(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int64_t, NANOARROW_TYPE_INT64>(state, 0.2);
}

BENCHMARK(BenchmarkArrayViewGetIntUnsafeInt8);
BENCHMARK(BenchmarkArrayViewGetIntUnsafeInt16);
BENCHMARK(BenchmarkArrayViewGetIntUnsafeInt32);
BENCHMARK(BenchmarkArrayViewGetIntUnsafeInt64);
BENCHMARK(BenchmarkArrayViewGetIntUnsafeInt64CheckNull);
