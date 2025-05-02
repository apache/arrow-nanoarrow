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

#ifndef NANOARROW_HPP_VIEW_HPP_INCLUDED
#define NANOARROW_HPP_VIEW_HPP_INCLUDED

#include <stdint.h>
#include <type_traits>

#include "nanoarrow/hpp/unique.hpp"
#include "nanoarrow/nanoarrow.h"

NANOARROW_CXX_NAMESPACE_BEGIN

namespace internal {
struct Nothing {};

template <typename T>
class Maybe {
 public:
  Maybe() : is_something_(false) {}
  Maybe(Nothing) : Maybe() {}

  Maybe(T something)  // NOLINT(google-explicit-constructor)
      : is_something_(true), something_(something) {}

  explicit constexpr operator bool() const { return is_something_; }

  const T& operator*() const { return something_; }

  friend inline bool operator==(Maybe l, Maybe r) {
    if (l.is_something_) {
      return r.is_something_ && l.something_ == r.something_;
    } else if (r.is_something_) {
      return l.is_something_ && l.something_ == r.something_;
    } else {
      return l.is_something_ == r.is_something_;
    }
  }
  friend inline bool operator!=(Maybe l, Maybe r) { return !(l == r); }

  T value_or(T val) const { return is_something_ ? something_ : val; }

 private:
  // When support for gcc 4.8 is dropped, we should also assert
  // is_trivially_copyable<T>::value
  static_assert(std::is_trivially_destructible<T>::value, "");

  bool is_something_{};
  T something_{};
};

template <typename Get>
struct RandomAccessRange {
  Get get;
  int64_t offset;
  int64_t size;

  using value_type = decltype(std::declval<Get>()(0));

  struct const_iterator {
    int64_t i;
    const RandomAccessRange* range;
    bool operator==(const_iterator other) const { return i == other.i; }
    bool operator!=(const_iterator other) const { return i != other.i; }
    const_iterator& operator++() { return ++i, *this; }
    value_type operator*() const { return range->get(i); }
  };

  const_iterator begin() const { return {offset, this}; }
  const_iterator end() const { return {offset + size, this}; }
};

template <typename Next>
struct InputRange {
  Next next;
  using ValueOrFalsy = decltype(std::declval<Next>()());

  static_assert(std::is_constructible<bool, ValueOrFalsy>::value, "");
  static_assert(std::is_default_constructible<ValueOrFalsy>::value, "");
  using value_type = decltype(*std::declval<ValueOrFalsy>());

  struct iterator {
    InputRange* range;
    ValueOrFalsy stashed;

    bool operator==(iterator other) const {
      return static_cast<bool>(stashed) == static_cast<bool>(other.stashed);
    }
    bool operator!=(iterator other) const { return !(*this == other); }

    iterator& operator++() {
      stashed = range->next();
      return *this;
    }
    value_type operator*() const { return *stashed; }
  };

  iterator begin() { return {this, next()}; }
  iterator end() { return {this, ValueOrFalsy()}; }
};
}  // namespace internal

/// \defgroup nanoarrow_hpp-range_for Range-for helpers
///
/// The Arrow C Data interface and the Arrow C Stream interface represent
/// data which can be iterated through using C++'s range-for statement.
///
/// @{

/// \brief An object convertible to any empty optional
constexpr internal::Nothing NA{};

/// \brief A range-for compatible wrapper for ArrowArray of fixed size type
///
/// Provides a sequence of optional<T> copied from each non-null
/// slot of the wrapped array (null slots result in empty optionals).
template <typename T>
class ViewArrayAs {
 private:
  struct Get {
    const uint8_t* validity;
    const void* values;

    internal::Maybe<T> operator()(int64_t i) const {
      if (validity == nullptr || ArrowBitGet(validity, i)) {
        if (std::is_same<T, bool>::value) {
          return ArrowBitGet(static_cast<const uint8_t*>(values), i);
        } else {
          return static_cast<const T*>(values)[i];
        }
      }
      return NA;
    }
  };

  internal::RandomAccessRange<Get> range_;

 public:
  ViewArrayAs(const ArrowArrayView* array_view)
      : range_{
            Get{
                array_view->buffer_views[0].data.as_uint8,
                array_view->buffer_views[1].data.data,
            },
            array_view->offset,
            array_view->length,
        } {}

  ViewArrayAs(const ArrowArray* array)
      : range_{
            Get{
                static_cast<const uint8_t*>(array->buffers[0]),
                array->buffers[1],
            },
            array->offset,
            array->length,
        } {}

  using value_type = typename internal::RandomAccessRange<Get>::value_type;
  using const_iterator = typename internal::RandomAccessRange<Get>::const_iterator;
  const_iterator begin() const { return range_.begin(); }
  const_iterator end() const { return range_.end(); }
  value_type operator[](int64_t i) const { return range_.get(i); }
};

/// \brief A range-for compatible wrapper for ArrowArray of binary or utf8
///
/// Provides a sequence of optional<ArrowStringView> referencing each non-null
/// slot of the wrapped array (null slots result in empty optionals). Large
/// binary and utf8 arrays can be wrapped by specifying 64 instead of 32 for
/// the template argument.
template <int OffsetSize>
class ViewArrayAsBytes {
 private:
  static_assert(OffsetSize == 32 || OffsetSize == 64, "");
  using OffsetType = typename std::conditional<OffsetSize == 32, int32_t, int64_t>::type;

  struct Get {
    const uint8_t* validity;
    const void* offsets;
    const char* data;

    internal::Maybe<ArrowStringView> operator()(int64_t i) const {
      auto* offsets = static_cast<const OffsetType*>(this->offsets);
      if (validity == nullptr || ArrowBitGet(validity, i)) {
        return ArrowStringView{data + offsets[i], offsets[i + 1] - offsets[i]};
      }
      return NA;
    }
  };

  internal::RandomAccessRange<Get> range_;

 public:
  ViewArrayAsBytes(const ArrowArrayView* array_view)
      : range_{
            Get{
                array_view->buffer_views[0].data.as_uint8,
                array_view->buffer_views[1].data.data,
                array_view->buffer_views[2].data.as_char,
            },
            array_view->offset,
            array_view->length,
        } {}

  ViewArrayAsBytes(const ArrowArray* array)
      : range_{
            Get{
                static_cast<const uint8_t*>(array->buffers[0]),
                array->buffers[1],
                static_cast<const char*>(array->buffers[2]),
            },
            array->offset,
            array->length,
        } {}

  using value_type = typename internal::RandomAccessRange<Get>::value_type;
  using const_iterator = typename internal::RandomAccessRange<Get>::const_iterator;
  const_iterator begin() const { return range_.begin(); }
  const_iterator end() const { return range_.end(); }
  value_type operator[](int64_t i) const { return range_.get(i); }
};

class ViewBinaryViewArrayAsBytes {
 private:
  struct Get {
    const uint8_t* validity;
    const union ArrowBinaryView* inline_data;
    const void** variadic_buffers;

    internal::Maybe<ArrowStringView> operator()(int64_t i) const {
      if (validity == nullptr || ArrowBitGet(validity, i)) {
        const union ArrowBinaryView* bv = &inline_data[i];
        if (bv->inlined.size <= NANOARROW_BINARY_VIEW_INLINE_SIZE) {
          return ArrowStringView{reinterpret_cast<const char*>(bv->inlined.data),
                                 bv->inlined.size};
        }

        return ArrowStringView{
            reinterpret_cast<const char*>(variadic_buffers[bv->ref.buffer_index]) +
                bv->ref.offset,
            bv->ref.size};
      }
      return NA;
    }
  };

  internal::RandomAccessRange<Get> range_;

 public:
  ViewBinaryViewArrayAsBytes(const ArrowArrayView* array_view)
      : range_{
            Get{
                array_view->buffer_views[0].data.as_uint8,
                array_view->buffer_views[1].data.as_binary_view,
                array_view->variadic_buffers,
            },
            array_view->offset,
            array_view->length,
        } {}

  ViewBinaryViewArrayAsBytes(const ArrowArray* array)
      : range_{
            Get{static_cast<const uint8_t*>(array->buffers[0]),
                static_cast<const union ArrowBinaryView*>(array->buffers[1]),
                array->buffers + NANOARROW_BINARY_VIEW_FIXED_BUFFERS},
            array->offset,
            array->length,
        } {}

  using value_type = typename internal::RandomAccessRange<Get>::value_type;
  using const_iterator = typename internal::RandomAccessRange<Get>::const_iterator;
  const_iterator begin() const { return range_.begin(); }
  const_iterator end() const { return range_.end(); }
  value_type operator[](int64_t i) const { return range_.get(i); }
};

/// \brief A range-for compatible wrapper for ArrowArray of fixed size binary
///
/// Provides a sequence of optional<ArrowStringView> referencing each non-null
/// slot of the wrapped array (null slots result in empty optionals).
class ViewArrayAsFixedSizeBytes {
 private:
  struct Get {
    const uint8_t* validity;
    const char* data;
    int fixed_size;

    internal::Maybe<ArrowStringView> operator()(int64_t i) const {
      if (validity == nullptr || ArrowBitGet(validity, i)) {
        return ArrowStringView{data + i * fixed_size, fixed_size};
      }
      return NA;
    }
  };

  internal::RandomAccessRange<Get> range_;

 public:
  ViewArrayAsFixedSizeBytes(const ArrowArrayView* array_view, int fixed_size)
      : range_{
            Get{
                array_view->buffer_views[0].data.as_uint8,
                array_view->buffer_views[1].data.as_char,
                fixed_size,
            },
            array_view->offset,
            array_view->length,
        } {}

  ViewArrayAsFixedSizeBytes(const ArrowArray* array, int fixed_size)
      : range_{
            Get{
                static_cast<const uint8_t*>(array->buffers[0]),
                static_cast<const char*>(array->buffers[1]),
                fixed_size,
            },
            array->offset,
            array->length,
        } {}

  using value_type = typename internal::RandomAccessRange<Get>::value_type;
  using const_iterator = typename internal::RandomAccessRange<Get>::const_iterator;
  const_iterator begin() const { return range_.begin(); }
  const_iterator end() const { return range_.end(); }
  value_type operator[](int64_t i) const { return range_.get(i); }
};

/// \brief A range-for compatible wrapper for ArrowArrayStream
///
/// Provides a sequence of ArrowArray& referencing the most recent array drawn
/// from the wrapped stream. (Each array may be moved from if necessary.)
/// When streams terminate due to an error, the error code and message are
/// available for inspection through the code() and error() member functions
/// respectively. Failure to inspect the error code will result in
/// an assertion failure. The number of arrays drawn from the stream is also
/// available through the count() member function.
class ViewArrayStream {
 public:
  ViewArrayStream(ArrowArrayStream* stream, ArrowErrorCode* code, ArrowError* error)
      : code_{code}, error_{error} {
    // Using a slightly more verbose constructor to silence a warning that occurs
    // on some versions of MSVC.
    range_.next.self = this;
    range_.next.stream = stream;
  }

  ViewArrayStream(ArrowArrayStream* stream, ArrowError* error)
      : ViewArrayStream{stream, &internal_code_, error} {}

  ViewArrayStream(ArrowArrayStream* stream)
      : ViewArrayStream{stream, &internal_code_, &internal_error_} {}

  // disable copy/move of this view, since its error references may point into itself
  ViewArrayStream(ViewArrayStream&&) = delete;
  ViewArrayStream& operator=(ViewArrayStream&&) = delete;
  ViewArrayStream(const ViewArrayStream&) = delete;
  ViewArrayStream& operator=(const ViewArrayStream&) = delete;

  // ensure the error code of this stream was accessed at least once
  ~ViewArrayStream() { NANOARROW_DCHECK(code_was_accessed_); }

 private:
  struct Next {
    ViewArrayStream* self;
    ArrowArrayStream* stream;
    UniqueArray array;

    ArrowArray* operator()() {
      array.reset();
      *self->code_ = ArrowArrayStreamGetNext(stream, array.get(), self->error_);

      if (array->release != nullptr) {
        NANOARROW_DCHECK(*self->code_ == NANOARROW_OK);
        ++self->count_;
        return array.get();
      }

      return nullptr;
    }
  };

  internal::InputRange<Next> range_;
  ArrowErrorCode* code_;
  ArrowError* error_;
  ArrowError internal_error_ = {};
  ArrowErrorCode internal_code_;
  bool code_was_accessed_ = false;
  int count_ = 0;

 public:
  using value_type = typename internal::InputRange<Next>::value_type;
  using iterator = typename internal::InputRange<Next>::iterator;
  iterator begin() { return range_.begin(); }
  iterator end() { return range_.end(); }

  /// The error code which caused this stream to terminate, if any.
  ArrowErrorCode code() {
    code_was_accessed_ = true;
    return *code_;
  }
  /// The error message which caused this stream to terminate, if any.
  ArrowError* error() { return error_; }

  /// The number of arrays streamed so far.
  int count() const { return count_; }
};

/// @}

NANOARROW_CXX_NAMESPACE_END

#endif
