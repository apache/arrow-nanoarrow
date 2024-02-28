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

#include "nanoarrow.hpp"

// Utility for making
template <typename CType, ArrowType type>
ArrowErrorCode InitSchemaAndArrayPrimitive(ArrowSchema* schema, ArrowArray* array,
                                           std::vector<CType> values,
                                           std::vector<int8_t> validity = {}) {
  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, type));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromSchema(array, schema, nullptr));

  // Set the data buffer
  nanoarrow::BufferInitSequence(ArrowArrayBuffer(array, 1), std::move(values));

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

static void BM_ArrayViewGetIntUnsafeInt8(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int8_t, NANOARROW_TYPE_INT8>(state);
}

static void BM_ArrayViewGetIntUnsafeInt16(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int16_t, NANOARROW_TYPE_INT16>(state);
}

static void BM_ArrayViewGetIntUnsafeInt32(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int32_t, NANOARROW_TYPE_INT32>(state);
}

static void BM_ArrayViewGetIntUnsafeInt64(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int64_t, NANOARROW_TYPE_INT64>(state);
}

static void BM_ArrayViewGetIntUnsafeInt8CheckNull(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int8_t, NANOARROW_TYPE_INT8>(state, 0.2);
}

static void BM_ArrayViewGetIntUnsafeInt16CheckNull(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int16_t, NANOARROW_TYPE_INT16>(state, 0.2);
}

static void BM_ArrayViewGetIntUnsafeInt32CheckNull(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int32_t, NANOARROW_TYPE_INT32>(state, 0.2);
}

static void BM_ArrayViewGetIntUnsafeInt64CheckNull(benchmark::State& state) {
  BaseArrayViewGetIntUnsafe<int64_t, NANOARROW_TYPE_INT64>(state, 0.2);
}

BENCHMARK(BM_ArrayViewGetIntUnsafeInt8);
BENCHMARK(BM_ArrayViewGetIntUnsafeInt16);
BENCHMARK(BM_ArrayViewGetIntUnsafeInt32);
BENCHMARK(BM_ArrayViewGetIntUnsafeInt64);
BENCHMARK(BM_ArrayViewGetIntUnsafeInt8CheckNull);
BENCHMARK(BM_ArrayViewGetIntUnsafeInt16CheckNull);
BENCHMARK(BM_ArrayViewGetIntUnsafeInt32CheckNull);
BENCHMARK(BM_ArrayViewGetIntUnsafeInt64CheckNull);
