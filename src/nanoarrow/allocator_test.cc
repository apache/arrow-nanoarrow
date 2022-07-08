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

static uint8_t* MemoryPoolAllocate(struct ArrowBufferAllocator* allocator, int64_t size) {
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(allocator->private_data);
  uint8_t* out;
  if (pool->Allocate(size, &out).ok()) {
    return out;
  } else {
    return nullptr;
  }
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
  allocator->allocate = &MemoryPoolAllocate;
  allocator->reallocate = &MemoryPoolReallocate;
  allocator->free = &MemoryPoolFree;
  allocator->private_data = pool;
}

TEST(AllocatorTest, AllocatorTestDefault) {
  struct ArrowBufferAllocator* allocator = ArrowBufferAllocatorDefault();

  uint8_t* buffer = allocator->allocate(allocator, 10);
  const char* test_str = "abcdefg";
  memcpy(buffer, test_str, strlen(test_str) + 1);

  buffer = allocator->reallocate(allocator, buffer, 10, 100);
  EXPECT_STREQ(reinterpret_cast<const char*>(buffer), test_str);

  allocator->free(allocator, buffer, 100);

  buffer = allocator->allocate(allocator, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);

  buffer =
      allocator->reallocate(allocator, buffer, 0, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);
}

TEST(AllocatorTest, AllocatorTestMemoryPool) {
  struct ArrowBufferAllocator arrow_allocator;
  MemoryPoolAllocatorInit(system_memory_pool(), &arrow_allocator);

  int64_t allocated0 = system_memory_pool()->bytes_allocated();

  uint8_t* buffer = arrow_allocator.allocate(&arrow_allocator, 10);
  EXPECT_EQ(system_memory_pool()->bytes_allocated() - allocated0, 10);
  memset(buffer, 0, 10);

  const char* test_str = "abcdefg";
  memcpy(buffer, test_str, strlen(test_str) + 1);

  buffer = arrow_allocator.reallocate(&arrow_allocator, buffer, 10, 100);
  EXPECT_EQ(system_memory_pool()->bytes_allocated() - allocated0, 100);
  EXPECT_STREQ(reinterpret_cast<const char*>(buffer), test_str);

  arrow_allocator.free(&arrow_allocator, buffer, 100);
  EXPECT_EQ(system_memory_pool()->bytes_allocated(), allocated0);

  buffer =
      arrow_allocator.allocate(&arrow_allocator, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);

  buffer = arrow_allocator.reallocate(&arrow_allocator, buffer, 0,
                                      std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);
}
