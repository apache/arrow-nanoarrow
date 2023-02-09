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

#include <cstring>
#include <string>
#include <thread>
#include <mutex>
#include <vector>

extern "C" void intptr_as_string(intptr_t ptr_int, char* buf) {
  std::string ptr_str = std::to_string(ptr_int);
  memcpy(buf, ptr_str.data(), ptr_str.size());
}

class PreservedSEXPRegistry {
 public:
  PreservedSEXPRegistry()
      : preserved_count_(0), main_thread_id_(std::this_thread::get_id()) {}

  int64_t size() {
    return preserved_count_;
  }

  void preserve(SEXP obj) {
    if (obj == R_NilValue) {
      return;
    }

    R_PreserveObject(obj);
    preserved_count_++;
  }

  bool release(SEXP obj) {
    if (obj == R_NilValue) {
      return true;
    }

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
      return true;
    }
  }

  int64_t empty_trash() {
    std::lock_guard<std::mutex> lock(trash_can_lock_);
    if (trash_can_.empty()) {
      return 0;
    }

    int64_t trash_size = trash_can_.size();
    for (const auto& obj : trash_can_) {
      R_ReleaseObject(obj);
      preserved_count_--;
    }
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
};

extern "C" void nanoarrow_preserve_init(void) {
  PreservedSEXPRegistry::GetInstance();
}

extern "C" void nanoarrow_preserve_sexp(SEXP obj) {
  PreservedSEXPRegistry::GetInstance().preserve(obj);
}

extern "C" void nanoarrow_release_sexp(SEXP obj) {
  try {
    PreservedSEXPRegistry::GetInstance().release(obj);
  } catch(std::exception& e) {
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

extern "C" void nanoarrow_preserve_and_release_on_other_thread(SEXP obj) {
  nanoarrow_preserve_sexp(obj);

  std::thread worker([obj] {
    nanoarrow_release_sexp(obj);
  });

  worker.join();
}