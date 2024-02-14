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

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nanoarrow.h"

#define NANOARROW_DEBUG_PRESERVE

// Without this infrastructure, it's possible to check that all objects
// are released by running devtools::test(); gc() in a fresh session and
// making sure that nanoarrow:::preserved_count() is zero afterward.
// When this isn't the case the process of debugging unreleased SEXPs
// is almost impossible without the bookkeeping below.
#if defined(NANOARROW_DEBUG_PRESERVE)
#include <unordered_map>
#endif

extern "C" void intptr_as_string(intptr_t ptr_int, char* buf) {
  std::string ptr_str = std::to_string(ptr_int);
  memcpy(buf, ptr_str.data(), ptr_str.size());
}

#if defined(NANOARROW_DEBUG_PRESERVE)
static std::string get_r_traceback(void) {
  SEXP fun = PROTECT(Rf_install("current_stack_trace_chr"));
  SEXP call = PROTECT(Rf_lang1(fun));
  SEXP nanoarrow_str = PROTECT(Rf_mkString("nanoarrow"));
  SEXP nanoarrow_ns = PROTECT(R_FindNamespace(nanoarrow_str));
  SEXP result = PROTECT(Rf_eval(call, nanoarrow_ns));
  const char* traceback_chr = Rf_translateCharUTF8(STRING_ELT(result, 0));
  std::string traceback_str(traceback_chr);
  UNPROTECT(5);
  return traceback_str;
}
#endif

class PreservedSEXPRegistry {
 public:
  PreservedSEXPRegistry()
      : preserved_count_(0), main_thread_id_(std::this_thread::get_id()) {}

  int64_t size() { return preserved_count_; }

  bool is_main_thread() { return std::this_thread::get_id() == main_thread_id_; }

  void preserve(SEXP obj) {
    if (obj == R_NilValue) {
      return;
    }

#if defined(NANOARROW_DEBUG_PRESERVE)
    Rprintf("PreservedSEXPRegistry::preserve(%p)\n", obj);
#endif

    R_PreserveObject(obj);
    preserved_count_++;

#if defined(NANOARROW_DEBUG_PRESERVE)
    if (tracebacks_.find(obj) != tracebacks_.end()) {
      tracebacks_[obj].first++;
    } else {
      tracebacks_[obj] = {1, get_r_traceback()};
    }

#endif
  }

  bool release(SEXP obj) {
    if (obj == R_NilValue) {
      return true;
    }

#if defined(NANOARROW_DEBUG_PRESERVE)
    Rprintf("PreservedSEXPRegistry::release(%p)\n", obj);
#endif

    // If there is an attempt to delete this object from another thread,
    // R_ReleaseObject() will almost certainly crash R or corrupt memory
    // leading to confusing errors. Instead, save a reference to the object
    // and provide an opportunity to delete it later.
    if (std::this_thread::get_id() != main_thread_id_) {
      std::lock_guard<std::mutex> lock(trash_can_lock_);
      trash_can_.push_back(obj);
      return false;
    } else {
      R_ReleaseObject(obj);
      preserved_count_--;

#if defined(NANOARROW_DEBUG_PRESERVE)
      if (tracebacks_.find(obj) != tracebacks_.end()) {
        tracebacks_[obj].first--;

        // Check for a situation where we've released more than we've preserved
        if (tracebacks_[obj].first < 0) {
          Rprintf("----%p---- (%ld reference(s) remaining)\nFirst preserved at\n%s\n\n",
                  obj, tracebacks_[obj].first, tracebacks_[obj].second.c_str());
          Rprintf("----%p----");
        }
      }
#endif
      return true;
    }
  }

  int64_t empty_trash() {
    std::lock_guard<std::mutex> lock(trash_can_lock_);
    int64_t trash_size = trash_can_.size();
    for (SEXP obj : trash_can_) {
      R_ReleaseObject(obj);
      preserved_count_--;

#if defined(NANOARROW_DEBUG_PRESERVE)
      if (tracebacks_.find(obj) != tracebacks_.end()) {
        tracebacks_[obj].first--;
        if (tracebacks_[obj].first == 0) {
          tracebacks_.erase(obj);
        }
      }
#endif
    }
    trash_can_.clear();

#if defined(NANOARROW_DEBUG_PRESERVE)
    if (preserved_count_ > 0) {
      Rprintf("%ld unreleased SEXP(s) after emptying the trash:\n",
              (long)preserved_count_);
      for (const auto& item : tracebacks_) {
        Rprintf("----%p---- (%ld reference(s) remaining)\nFirst preserved at\n%s\n\n",
                item.first, item.second.first, item.second.second.c_str());
      }
    }
#endif

    return trash_size;
  }

  static PreservedSEXPRegistry& GetInstance() {
    static PreservedSEXPRegistry singleton;
    return singleton;
  }

 private:
  int64_t preserved_count_;
  std::thread::id main_thread_id_;
  std::vector<SEXP> trash_can_;
  std::mutex trash_can_lock_;

#if defined(NANOARROW_DEBUG_PRESERVE)
  std::unordered_map<SEXP, std::pair<int64_t, std::string>> tracebacks_;
#endif
};

extern "C" void nanoarrow_preserve_init(void) { PreservedSEXPRegistry::GetInstance(); }

extern "C" void nanoarrow_preserve_sexp(SEXP obj) {
  PreservedSEXPRegistry::GetInstance().preserve(obj);
}

extern "C" void nanoarrow_release_sexp(SEXP obj) {
  try {
    PreservedSEXPRegistry::GetInstance().release(obj);
  } catch (std::exception& e) {
    // Just for safety...we really don't want to crash here
  }
}

extern "C" int64_t nanoarrow_preserved_count(void) {
  return PreservedSEXPRegistry::GetInstance().size();
}

extern "C" int64_t nanoarrow_preserved_empty(void) {
  try {
    return PreservedSEXPRegistry::GetInstance().empty_trash();
  } catch (std::exception& e) {
    return 0;
  }
}

static void ReleaseSharedArray(ArrowArray* array) {
  auto shared_array = reinterpret_cast<std::shared_ptr<ArrowArray>*>(array->private_data);
  delete shared_array;
  array->release = nullptr;
}

extern "C" void nanoarrow_array_shallow_copy(struct ArrowArray* src,
                                             struct ArrowArray* dst) {
  auto shared_array =
      std::shared_ptr<ArrowArray>(new ArrowArray(), [](ArrowArray* value) {
        if (value && value->release) {
          value->release(value);
        }
      });

  ArrowArrayMove(src, shared_array.get());

  std::memcpy(shared_array.get(), src, sizeof(ArrowArray));
  src->private_data = new std::shared_ptr<ArrowArray>(shared_array);
  src->release = &ReleaseSharedArray;

  if (dst != nullptr) {
    std::memcpy(shared_array.get(), dst, sizeof(ArrowArray));
    dst->private_data = new std::shared_ptr<ArrowArray>(shared_array);
    dst->release = &ReleaseSharedArray;
  }
}

static void DeallocateBorrowedBuffer(ArrowBufferAllocator* allocator, uint8_t* ptr,
                                     int64_t size) {
  auto shared_array =
      reinterpret_cast<std::shared_ptr<ArrowArray>*>(allocator->private_data);
  delete shared_array;
}

static ArrowErrorCode BorrowBuffer(const std::shared_ptr<ArrowArray>& shared_array,
                                   const void* addr, ArrowBuffer* buffer) {
  NANOARROW_RETURN_NOT_OK(ArrowBufferSetAllocator(
      buffer, ArrowBufferDeallocator(&DeallocateBorrowedBuffer,
                                     new std::shared_ptr<ArrowArray>(shared_array))));
  buffer->data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(addr));
  buffer->size_bytes = 0;
  buffer->capacity_bytes = 0;
  return NANOARROW_OK;
}

extern "C" ArrowErrorCode nanoarrow_array_editable_copy(struct ArrowArray* src,
                                                        struct ArrowArray* dst) {
  nanoarrow_array_shallow_copy(src, nullptr);
  auto shared_array = std::shared_ptr<ArrowArray>(
      *reinterpret_cast<std::shared_ptr<ArrowArray>*>(src->private_data));

  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(dst, NANOARROW_TYPE_UNINITIALIZED));

  for (int64_t i = 0; i < src->n_buffers; i++) {
    NANOARROW_RETURN_NOT_OK(
        BorrowBuffer(shared_array, src->buffers[i], ArrowArrayBuffer(dst, i)));
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(dst, src->n_children));
  for (int64_t i = 0; i < src->n_children; i++) {
    nanoarrow_array_shallow_copy(src->children[i], dst->children[i]);
  }

  if (src->dictionary) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateDictionary(dst));
    nanoarrow_array_shallow_copy(src->dictionary, dst->dictionary);
  }

  dst->length = src->length;
  dst->offset = src->offset;
  dst->n_buffers = src->n_buffers;

  NANOARROW_RETURN_NOT_OK(
      ArrowArrayFinishBuilding(dst, NANOARROW_VALIDATION_LEVEL_NONE, nullptr));
  return NANOARROW_OK;
}

extern "C" int nanoarrow_is_main_thread(void) {
  return PreservedSEXPRegistry::GetInstance().is_main_thread();
}

extern "C" void nanoarrow_preserve_and_release_on_other_thread(SEXP obj) {
  nanoarrow_preserve_sexp(obj);
  std::thread worker([obj] { nanoarrow_release_sexp(obj); });
  worker.join();
}
