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
  std::cerr << "           [[--to [json] [-]] OR [--check [json|ipc] [file or -]]]\n";
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

  bool has_kwarg(const std::string& key) const {
    return kwargs_.find(key) != kwargs_.end();
  }

  const std::pair<std::string, std::string>& kwarg(const std::string& key) const {
    return kwargs_.find(key)->second;
  }

 private:
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
    print_help();
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
    print_help();
    return EINVAL;
  }
}

ArrowErrorCode CheckArrayStream(const std::string& format, const std::string& ref,
                                ArrowArrayStream* actual, ArrowError* error) {
  nanoarrow::ipc::UniqueInputStream check;
  NANOARROW_RETURN_NOT_OK(Open(ref, check.get(), error));

  nanoarrow::UniqueArrayStream expected;
  NANOARROW_RETURN_NOT_OK(GetArrayStream(format, check.get(), expected.get(), error));

  nanoarrow::testing::TestingJSONComparison comparison;
  NANOARROW_RETURN_NOT_OK(comparison.CompareArrayStream(actual, expected.get(), error));

  if (comparison.num_differences() > 0) {
    std::cerr << comparison.num_differences()
              << " Difference(s) found between --from and --check:\n";
    comparison.WriteDifferences(std::cerr);
    return EINVAL;
  }

  return NANOARROW_OK;
}

int DoMain(const ArgumentParser& args, ArrowError* error) {
  nanoarrow::ipc::UniqueInputStream from;
  NANOARROW_RETURN_NOT_OK(Open(args.kwarg("from").second, from.get(), error));

  nanoarrow::UniqueArrayStream stream;
  NANOARROW_RETURN_NOT_OK(
      GetArrayStream(args.kwarg("from").first, from.get(), stream.get(), error));

  if (args.has_kwarg("to")) {
    if (args.kwarg("to").second != "-") {
      std::cerr << "--to output is only supported to stdout ('-')\n";
      print_help();
      return EINVAL;
    }

    NANOARROW_RETURN_NOT_OK(
        WriteArrayStream(args.kwarg("to").first, stream.get(), error));
  } else if (args.has_kwarg("check")) {
    NANOARROW_RETURN_NOT_OK(CheckArrayStream(
        args.kwarg("check").first, args.kwarg("check").second, stream.get(), error));
  } else {
    std::cerr << "One of --check or --to must be specified";
    print_help();
    return EINVAL;
  }

  return NANOARROW_OK;
}

int main(int argc, char* argv[]) {
  ArrowError error;
  error.message[0] = '\0';

  ArgumentParser args;
  int result = args.parse(argc, argv);
  if (result != NANOARROW_OK) {
    print_help();
    return result;
  }

  result = DoMain(args, &error);
  if (result != NANOARROW_OK) {
    std::cerr << error.message << "\n";
    return result;
  }

  return 0;
}
