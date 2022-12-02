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

#include <cstring>
#include <string>

#include <arrow/memory_pool.h>
#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.h"

using namespace arrow;

TEST(BuildIdTest, BuildIdTest) {
  EXPECT_STREQ(ArrowNanoarrowBuildId(), NANOARROW_BUILD_ID);
}

TEST(BuildIdTest, VersionTest) {
  EXPECT_STREQ(ArrowNanoarrowVersion(), NANOARROW_VERSION);
  EXPECT_EQ(ArrowNanoarrowVersionInt(), NANOARROW_VERSION_INT);
}

TEST(ErrorTest, ErrorTestSet) {
  ArrowError error;
  EXPECT_EQ(ArrowErrorSet(&error, "there were %d foxes", 4), NANOARROW_OK);
  EXPECT_STREQ(ArrowErrorMessage(&error), "there were 4 foxes");
}

TEST(ErrorTest, ErrorTestSetOverrun) {
  ArrowError error;
  char big_error[2048];
  const char* a_few_chars = "abcdefg";
  for (int i = 0; i < 2047; i++) {
    big_error[i] = a_few_chars[i % strlen(a_few_chars)];
  }
  big_error[2047] = '\0';

  EXPECT_EQ(ArrowErrorSet(&error, "%s", big_error), ERANGE);
  EXPECT_EQ(std::string(ArrowErrorMessage(&error)), std::string(big_error, 1023));

  wchar_t bad_string[] = {0xFFFF, 0};
  EXPECT_EQ(ArrowErrorSet(&error, "%ls", bad_string), EINVAL);
}

static uint8_t* MemoryPoolReallocate(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                                     int64_t old_size, int64_t new_size) {
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(allocator->private_data);
  uint8_t* out = ptr;
  if (pool->Reallocate(old_size, new_size, &out).ok()) {
    return out;
  } else {
    return nullptr;
  }
}

static void MemoryPoolFree(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                           int64_t size) {
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(allocator->private_data);
  pool->Free(ptr, size);
}

static void MemoryPoolAllocatorInit(MemoryPool* pool,
                                    struct ArrowBufferAllocator* allocator) {
  allocator->reallocate = &MemoryPoolReallocate;
  allocator->free = &MemoryPoolFree;
  allocator->private_data = pool;
}

TEST(AllocatorTest, AllocatorTestDefault) {
  struct ArrowBufferAllocator allocator = ArrowBufferAllocatorDefault();

  uint8_t* buffer = allocator.reallocate(&allocator, nullptr, 0, 10);
  const char* test_str = "abcdefg";
  memcpy(buffer, test_str, strlen(test_str) + 1);

  buffer = allocator.reallocate(&allocator, buffer, 10, 100);
  EXPECT_STREQ(reinterpret_cast<const char*>(buffer), test_str);

  allocator.free(&allocator, buffer, 100);

  buffer =
      allocator.reallocate(&allocator, nullptr, 0, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);

  buffer =
      allocator.reallocate(&allocator, buffer, 0, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);
}

// In a non-trivial test this struct could hold a reference to an object
// that keeps the buffer from being garbage collected (e.g., an SEXP in R)
struct CustomFreeData {
  void* pointer_proxy;
};

static void CustomFree(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                       int64_t size) {
  auto data = reinterpret_cast<struct CustomFreeData*>(allocator->private_data);
  ArrowFree(data->pointer_proxy);
  data->pointer_proxy = nullptr;
}

TEST(AllocatorTest, AllocatorTestDeallocator) {
  struct CustomFreeData data;
  data.pointer_proxy = reinterpret_cast<uint8_t*>(ArrowMalloc(12));

  struct ArrowBufferAllocator deallocator = ArrowBufferDeallocator(&CustomFree, &data);

  EXPECT_EQ(deallocator.reallocate(&deallocator, nullptr, 0, 12), nullptr);
  EXPECT_EQ(deallocator.reallocate(&deallocator, nullptr, 0, 12), nullptr);
  deallocator.free(&deallocator, nullptr, 12);
  EXPECT_EQ(data.pointer_proxy, nullptr);
}

TEST(AllocatorTest, AllocatorTestMemoryPool) {
  struct ArrowBufferAllocator arrow_allocator;
  MemoryPoolAllocatorInit(system_memory_pool(), &arrow_allocator);

  int64_t allocated0 = system_memory_pool()->bytes_allocated();

  uint8_t* buffer = arrow_allocator.reallocate(&arrow_allocator, nullptr, 0, 10);
  EXPECT_EQ(system_memory_pool()->bytes_allocated() - allocated0, 10);
  memset(buffer, 0, 10);

  const char* test_str = "abcdefg";
  memcpy(buffer, test_str, strlen(test_str) + 1);

  buffer = arrow_allocator.reallocate(&arrow_allocator, buffer, 10, 100);
  EXPECT_EQ(system_memory_pool()->bytes_allocated() - allocated0, 100);
  EXPECT_STREQ(reinterpret_cast<const char*>(buffer), test_str);

  arrow_allocator.free(&arrow_allocator, buffer, 100);
  EXPECT_EQ(system_memory_pool()->bytes_allocated(), allocated0);

  buffer = arrow_allocator.reallocate(&arrow_allocator, nullptr, 0,
                                      std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);

  buffer = arrow_allocator.reallocate(&arrow_allocator, buffer, 0,
                                      std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);
}
