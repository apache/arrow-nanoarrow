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

#include <tuple>

#include <nanoarrow/nanoarrow.hpp>

#if !defined(__cpp_variadic_using) || !defined(__cpp_deduction_guides) ||       \
    !defined(__cpp_fold_expressions) || !defined(__cpp_lib_integer_sequence) || \
    __cpp_range_based_for < 201603L  // end sentinels
#error "Zip cannot be supported without C++17 features"
#endif

namespace nanoarrow {

template <typename Ranges, typename Indices>
struct Zip;

template <typename... Ranges>
Zip(Ranges&&...) -> Zip<std::tuple<Ranges...>, std::index_sequence_for<Ranges...>>;

template <typename... Ranges, size_t... I>
struct Zip<std::tuple<Ranges...>, std::index_sequence<I...>> {
  explicit Zip(Ranges... ranges) : ranges_(std::forward<Ranges>(ranges)...) {}

  std::tuple<Ranges...> ranges_;

  using sentinel = std::tuple<decltype(std::get<I>(ranges_).end())...>;

  struct iterator : std::tuple<decltype(std::get<I>(ranges_).begin())...> {
    using iterator::tuple::tuple;

    auto operator*() {
      return std::tuple<decltype(*std::get<I>(*this))...>{*std::get<I>(*this)...};
    }

    iterator& operator++() {
      (++std::get<I>(*this), ...);
      return *this;
    }

    bool operator!=(const sentinel& s) const {
      bool any_iterator_at_end = (... || (std::get<I>(*this) == std::get<I>(s)));
      return !any_iterator_at_end;
    }
  };

  iterator begin() { return {std::get<I>(ranges_).begin()...}; }

  sentinel end() { return {std::get<I>(ranges_).end()...}; }
};

constexpr auto Enumerate = [] {
  struct {
    struct sentinel {};
    constexpr sentinel end() const { return {}; }

    struct iterator {
      int64_t i{0};

      constexpr int64_t operator*() { return i; }

      constexpr iterator& operator++() {
        ++i;
        return *this;
      }

      constexpr std::true_type operator!=(sentinel) const { return {}; }
      constexpr std::false_type operator==(sentinel) const { return {}; }
    };
    constexpr iterator begin() const { return {}; }
  } out;

  return out;
}();

}  // namespace nanoarrow
