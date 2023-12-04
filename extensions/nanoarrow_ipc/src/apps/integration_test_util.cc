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

#include <cerrno>
#include <deque>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "nanoarrow/nanoarrow_ipc.hpp"
#include "nanoarrow/nanoarrow_testing.hpp"

void print_help() {
  std::cerr << "nanoarrow version " << ArrowNanoarrowVersion() << "\n";
  std::cerr << "  Usage: integration_test_util convert\n";
  std::cerr << "           --from [json|ipc] [file or -]\n";
  std::cerr << "           [--to [json] [-] [--check [json|ipc] [file or -]]]\n";
}

class ArgumentParser {
 public:
  ArrowErrorCode parse(int argc, char* argv[]) {
    std::deque<std::string> args;
    for (int i = 0; i < argc; i++) {
      args.push_back(argv[i]);
    }

    // executable name is first
    if (!args.empty()) {
      args.pop_front();
    }

    while (!args.empty()) {
      std::string item = args.front();
      args.pop_front();

      if (item.substr(0, 2) == "--") {
        if (item != "--from" && item != "--to" && item != "--check") {
          std::cerr << "Unknown kwarg: '" << item << "'\n";
        }

        if (args.size() < 2) {
          std::cerr << "kwarg " << item << ": expected following [format] [file or -]\n";
          return EINVAL;
        }

        std::string format = args.front();
        args.pop_front();
        std::string ref = args.front();
        args.pop_front();

        kwargs_[item.substr(2)] = {format, ref};
      } else {
        std::cerr << "Unexpected arg: '" << item << "'\n";
        return EINVAL;
      }
    }

    if (!has_kwarg("from")) {
      std::cerr << "--from is a required argument\n";
      return EINVAL;
    }

    if (has_kwarg("to") && has_kwarg("check")) {
      std::cerr << "--to with --check is not supported";
    }

    return NANOARROW_OK;
  }

  bool has_kwarg(const std::string& key) { return kwargs_.find(key) != kwargs_.end(); }

  const std::pair<std::string, std::string>& kwarg(const std::string& key) {
    return kwargs_[key];
  }

 private:
  std::string action_;
  std::unordered_map<std::string, std::pair<std::string, std::string>> kwargs_;
};

ArrowErrorCode Open(const std::string& ref, ArrowIpcInputStream* out, ArrowError* error) {
  FILE* file_ptr;
  if (ref == "-") {
    file_ptr = freopen(NULL, "rb", stdin);
  } else {
    file_ptr = fopen(ref.c_str(), "rb");
  }

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowIpcInputStreamInitFile(out, file_ptr, true),
                                     error);
  return NANOARROW_OK;
}

ArrowErrorCode GetArrayStream(const std::string& format, ArrowIpcInputStream* input,
                              ArrowArrayStream* out, ArrowError* error) {
  if (format == "ipc") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowIpcArrayStreamReaderInit(out, input, nullptr),
                                       error);
    return NANOARROW_OK;
  } else if (format == "json") {
    // Read input
    std::stringstream ss;
    int64_t bytes_read = 0;
    uint8_t buf[1024];
    do {
      ss << std::string(reinterpret_cast<char*>(buf), bytes_read);
      NANOARROW_RETURN_NOT_OK(input->read(input, buf, sizeof(buf), &bytes_read, error));
    } while (bytes_read > 0);

    // Parse it
    nanoarrow::testing::TestingJSONReader json_reader;
    NANOARROW_RETURN_NOT_OK(json_reader.ReadDataFile(ss.str(), out, error));
    return NANOARROW_OK;
  } else {
    std::cerr << "Unknown or unsupported format --from " << format << "\n";
    return EINVAL;
  }
}

ArrowErrorCode WriteArrayStream(const std::string& format, ArrowArrayStream* stream,
                                ArrowError* error) {
  if (format == "json") {
    nanoarrow::testing::TestingJSONWriter writer;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(writer.WriteDataFile(std::cout, stream), error);
    return NANOARROW_OK;
  } else {
    std::cerr << "Unknown or unsupported format --to " << format << "\n";
    return EINVAL;
  }
}

ArrowErrorCode CheckArrayStream(const std::string& format, const std::string& ref,
                                ArrowArrayStream* actual, ArrowError* error) {
  nanoarrow::ipc::UniqueInputStream check;
  NANOARROW_RETURN_NOT_OK(Open(ref, check.get(), error));

  nanoarrow::UniqueArrayStream expected;
  NANOARROW_RETURN_NOT_OK(GetArrayStream(format, check.get(), expected.get(), error));

  nanoarrow::UniqueSchema actual_schema;
  nanoarrow::UniqueSchema expected_schema;

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(actual->get_schema(actual, actual_schema.get()),
                                     error);
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      expected->get_schema(expected.get(), expected_schema.get()), error);

  nanoarrow::testing::TestingJSONComparison comparison;
  NANOARROW_RETURN_NOT_OK(
      comparison.CompareSchema(expected_schema.get(), actual_schema.get(), error));
  if (comparison.num_differences() > 0) {
    std::cerr << comparison.num_differences()
              << " Difference(s) found between actual Schema and expected Schema:\n";
    comparison.WriteDifferences(std::cerr);
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(comparison.SetSchema(expected_schema.get(), error));

  int64_t n_batches = -1;
  nanoarrow::UniqueArray actual_array;
  nanoarrow::UniqueArray expected_array;
  do {
    n_batches++;
    actual_array.reset();
    expected_array.reset();
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(actual->get_next(actual, actual_array.get()),
                                       error);
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        expected->get_next(expected.get(), expected_array.get()), error);

    if (actual_array->release == nullptr && expected_array->release != nullptr) {
      std::cerr << "Actual stream finished; expected stream is not finished\n";
      return EINVAL;
    }

    if (actual_array->release != nullptr && expected_array->release == nullptr) {
      std::cerr << "Expected stream finished; actual stream is not finished\n";
      return EINVAL;
    }

    if (actual_array->release == nullptr) {
      break;
    }

    NANOARROW_RETURN_NOT_OK(
        comparison.CompareBatch(actual_array.get(), expected_array.get(), error));
    if (comparison.num_differences() > 0) {
      std::cerr << comparison.num_differences()
                << " Difference(s) found between actual Batch " << n_batches
                << " and expected Batch " << n_batches << ":\n";
      comparison.WriteDifferences(std::cerr);
      return EINVAL;
    }
  } while (true);

  return NANOARROW_OK;
}

int DoMain(int argc, char* argv[], ArrowError* error) {
  ArgumentParser args;
  NANOARROW_RETURN_NOT_OK(args.parse(argc, argv));

  nanoarrow::ipc::UniqueInputStream from;
  NANOARROW_RETURN_NOT_OK(Open(args.kwarg("from").second, from.get(), error));

  nanoarrow::UniqueArrayStream stream;
  NANOARROW_RETURN_NOT_OK(
      GetArrayStream(args.kwarg("from").first, from.get(), stream.get(), error));

  if (args.has_kwarg("to")) {
    if (args.kwarg("to").second != "-") {
      std::cerr << "--to output is only supported to stdout ('-')\n";
      return EINVAL;
    }

    NANOARROW_RETURN_NOT_OK(
        WriteArrayStream(args.kwarg("to").first, stream.get(), error));
  } else if (args.has_kwarg("check")) {
    NANOARROW_RETURN_NOT_OK(CheckArrayStream(
        args.kwarg("check").first, args.kwarg("check").second, stream.get(), error));
  } else {
    std::cerr << "One of --check or --to must be specified";
    return EINVAL;
  }

  return NANOARROW_OK;
}

int main(int argc, char* argv[]) {
  ArrowError error;
  error.message[0] = '\0';

  int result = DoMain(argc, argv, &error);
  if (result != NANOARROW_OK) {
    std::cerr << error.message << "\n";
    print_help();
    return result;
  }

  return 0;
}
