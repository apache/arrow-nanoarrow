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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nanoarrow.hpp"
#include "nanoarrow/r.h"

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
        if (tracebacks_[obj].first == 0) {
          tracebacks_.erase(obj);
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

extern "C" int nanoarrow_is_main_thread(void) {
  return PreservedSEXPRegistry::GetInstance().is_main_thread();
}

extern "C" void nanoarrow_preserve_and_release_on_other_thread(SEXP obj) {
  nanoarrow_preserve_sexp(obj);
  std::thread worker([obj] { nanoarrow_release_sexp(obj); });
  worker.join();
}

// Collector utility for iterating over and collecting batches
// Keeping this all in a single object reduces the amount of C++ deletion
// we need to keep track of.
struct ArrayVector {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray batch;
  std::vector<nanoarrow::UniqueArray> vec;
};

// Use an external pointer to handle deleting the ArrayVector in
// the event of a longjmp
static void release_array_vector_xptr(SEXP array_vector_xptr) {
  auto ptr = reinterpret_cast<ArrayVector*>(R_ExternalPtrAddr(array_vector_xptr));
  if (ptr != NULL) {
    delete ptr;
  }
}

// Collects the entire array stream and collects the total number of rows and
// total number of batches so that the R code on the end of this can decide
// how best to proceed.
extern "C" SEXP nanoarrow_c_collect_array_stream(SEXP array_stream_xptr, SEXP n_sexp) {
  struct ArrowArrayStream* array_stream =
      nanoarrow_array_stream_from_xptr(array_stream_xptr);

  double n_real = REAL(n_sexp)[0];
  int n;
  if (R_FINITE(n_real)) {
    n = (int)n_real;
  } else {
    n = INT_MAX;
  }

  auto array_vector = new ArrayVector();
  SEXP array_vector_xptr =
      PROTECT(R_MakeExternalPtr(array_vector, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(array_vector_xptr, &release_array_vector_xptr);

  struct ArrowError error;
  ArrowErrorInit(&error);
  int code = ArrowArrayStreamGetSchema(array_stream, array_vector->schema.get(), &error);
  if (code != NANOARROW_OK) {
    Rf_error("ArrowArrayStreamGetSchema() failed (%d): %s", code, error.message);
  }

  int64_t n_actual = 0;
  int64_t size = 0;
  while (n > 0) {
    code = ArrowArrayStreamGetNext(array_stream, array_vector->batch.get(), &error);
    if (code != NANOARROW_OK) {
      Rf_error("ArrowArrayStreamGetNext() failed (%d): %s", code, error.message);
    }

    if (array_vector->batch->release == nullptr) {
      break;
    }

    size += array_vector->batch->length;
    ++n_actual;
    --n;
    array_vector->vec.push_back(std::move(array_vector->batch));
    array_vector->batch.reset();

    R_CheckUserInterrupt();
  }

  SEXP array_stream_out_xptr = PROTECT(nanoarrow_array_stream_owning_xptr());
  struct ArrowArrayStream* array_stream_out =
      nanoarrow_output_array_stream_from_xptr(array_stream_out_xptr);

  nanoarrow::VectorArrayStream(array_vector->schema.get(), std::move(array_vector->vec))
      .ToArrayStream(array_stream_out);

  SEXP size_sexp = PROTECT(Rf_ScalarReal(size));
  SEXP n_actual_sexp = PROTECT(Rf_ScalarReal(n_actual));
  const char* names[] = {"stream", "size", "n", ""};
  SEXP out = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(out, 0, array_stream_out_xptr);
  SET_VECTOR_ELT(out, 1, size_sexp);
  SET_VECTOR_ELT(out, 2, n_actual_sexp);

  UNPROTECT(5);
  return out;
}
