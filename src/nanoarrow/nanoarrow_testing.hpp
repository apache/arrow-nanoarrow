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

#ifndef NANOARROW_TESTING_HPP_INCLUDED
#define NANOARROW_TESTING_HPP_INCLUDED

#include <iostream>
#include <string>

#include "nanoarrow/nanoarrow.hpp"

/// \defgroup nanoarrow_testing Nanoarrow Testing Helpers
///
/// Utilities for testing nanoarrow structures and functions.

namespace nanoarrow {

namespace testing {

// Forward-declaration of internal types
namespace internal {
class DictionaryContext;
struct Differences;
}  // namespace internal

/// \defgroup nanoarrow_testing-json Integration test helpers
///
/// See testing format documentation for details of the JSON representation. This
/// representation is not canonical but can be used to implement integration tests with
/// other implementations.
///
/// @{

/// \brief Writer for the Arrow integration testing JSON format
class NANOARROW_DLL TestingJSONWriter {
 public:
  TestingJSONWriter();
  virtual ~TestingJSONWriter();

  /// \brief Set the floating point precision of the writer
  ///
  /// The floating point precision by default is -1, which uses the JSON serializer
  /// to encode the value in the output. When writing files specifically for
  /// integration tests, floating point values should be rounded to 3 decimal places to
  /// avoid serialization issues.
  void set_float_precision(int value) { float_precision_ = value; }

  /// \brief Set whether metadata should be included in the output of a schema or field
  ///
  /// Use false to skip writing schema/field metadata in the output.
  void set_include_metadata(bool value) { include_metadata_ = value; }

  void ResetDictionaries();

  /// \brief Write an ArrowArrayStream as a data file JSON object to out
  ///
  /// Creates output like `{"schema": {...}, "batches": [...], ...}`.
  ArrowErrorCode WriteDataFile(std::ostream& out, ArrowArrayStream* stream);

  /// \brief Write a schema to out
  ///
  /// Creates output like `{"fields": [...], "metadata": [...]}`.
  ArrowErrorCode WriteSchema(std::ostream& out, const ArrowSchema* schema);

  /// \brief Write a field to out
  ///
  /// Creates output like `{"name" : "col", "type": {...}, ...}`
  ArrowErrorCode WriteField(std::ostream& out, const ArrowSchema* field);

  /// \brief Write the type portion of a field
  ///
  /// Creates output like `{"name": "int", ...}`
  ArrowErrorCode WriteType(std::ostream& out, const ArrowSchema* field);

  /// \brief Write the metadata portion of a field
  ///
  /// Creates output like `[{"key": "...", "value": "..."}, ...]`.
  ArrowErrorCode WriteMetadata(std::ostream& out, const char* metadata);

  /// \brief Write a "batch" to out
  ///
  /// Creates output like `{"count": 123, "columns": [...]}`.
  ArrowErrorCode WriteBatch(std::ostream& out, const ArrowSchema* schema,
                            const ArrowArrayView* value);

  /// \brief Write a column to out
  ///
  /// Creates output like `{"name": "col", "count": 123, "VALIDITY": [...], ...}`.
  ArrowErrorCode WriteColumn(std::ostream& out, const ArrowSchema* field,
                             const ArrowArrayView* value);

  ArrowErrorCode WriteDictionaryBatches(std::ostream& out);

 private:
  int float_precision_;
  bool include_metadata_;
  internal::DictionaryContext* dictionaries_;

  bool ShouldWriteMetadata(const char* metadata) {
    return metadata != nullptr && include_metadata_;
  }

  ArrowErrorCode WriteDictionaryBatch(std::ostream& out, int32_t dictionary_id);

  ArrowErrorCode WriteFieldChildren(std::ostream& out, const ArrowSchema* field);

  ArrowErrorCode WriteFieldDictionary(std::ostream& out, int32_t dictionary_id,
                                      bool is_ordered,
                                      const ArrowSchemaView* indices_field);

  ArrowErrorCode WriteChildren(std::ostream& out, const ArrowSchema* field,
                               const ArrowArrayView* value);
};

/// \brief Reader for the Arrow integration testing JSON format
class NANOARROW_DLL TestingJSONReader {
 public:
  TestingJSONReader(ArrowBufferAllocator allocator);
  TestingJSONReader();
  virtual ~TestingJSONReader();

  static const int kNumBatchOnlySchema = -2;
  static const int kNumBatchReadAll = -1;

  /// \brief Read JSON representing a data file object
  ///
  /// Read a JSON object in the form `{"schema": {...}, "batches": [...], ...}`,
  /// propagating `out` on success.
  ArrowErrorCode ReadDataFile(const std::string& data_file_json, ArrowArrayStream* out,
                              int num_batch = kNumBatchReadAll,
                              ArrowError* error = nullptr);

  /// \brief Read JSON representing a Schema
  ///
  /// Reads a JSON object in the form `{"fields": [...], "metadata": [...]}`,
  /// propagating `out` on success.
  ArrowErrorCode ReadSchema(const std::string& schema_json, ArrowSchema* out,
                            ArrowError* error = nullptr);

  /// \brief Read JSON representing a Field
  ///
  /// Read a JSON object in the form `{"name" : "col", "type": {...}, ...}`,
  /// propagating `out` on success.
  ArrowErrorCode ReadField(const std::string& field_json, ArrowSchema* out,
                           ArrowError* error = nullptr);

  /// \brief Read JSON representing a RecordBatch
  ///
  /// Read a JSON object in the form `{"count": 123, "columns": [...]}`, propagating `out`
  /// on success.
  ArrowErrorCode ReadBatch(const std::string& batch_json, const ArrowSchema* schema,
                           ArrowArray* out, ArrowError* error = nullptr);

  /// \brief Read JSON representing a Column
  ///
  /// Read a JSON object in the form
  /// `{"name": "col", "count": 123, "VALIDITY": [...], ...}`, propagating
  /// `out` on success.
  ArrowErrorCode ReadColumn(const std::string& column_json, const ArrowSchema* schema,
                            ArrowArray* out, ArrowError* error = nullptr);

 private:
  ArrowBufferAllocator allocator_;
  internal::DictionaryContext* dictionaries_;

  void SetArrayAllocatorRecursive(ArrowArray* array);
};

/// \brief Integration testing comparison utility
///
/// Utility to compare ArrowSchema, ArrowArray, and ArrowArrayStream instances.
/// This should only be used in the context of integration testing as the
/// comparison logic is specific to the integration testing JSON files and
/// specification. Notably:
///
/// - Map types are considered equal regardless of the child names "entries",
///   "key", and "value".
/// - Float32 and Float64 values are compared according to their JSON serialization.
class NANOARROW_DLL TestingJSONComparison {
 public:
  TestingJSONComparison();
  virtual ~TestingJSONComparison();

  /// \brief Compare top-level RecordBatch flags (e.g., nullability)
  ///
  /// Some Arrow implementations export batches as nullable, and some export them as
  /// non-nullable. Use false to consider these two types of batches as equivalent.
  void set_compare_batch_flags(bool value) { compare_batch_flags_ = value; }

  /// \brief Compare metadata order
  ///
  /// Some Arrow implementations store metadata using structures (e.g., hash map) that
  /// reorder metadata items. Use false to consider metadata whose keys/values have
  /// been reordered as equivalent.
  void set_compare_metadata_order(bool value) { compare_metadata_order_ = value; }

  /// \brief Set float precision
  ///
  /// The Arrow Integration Testing JSON document states that values should be compared
  /// to 3 decimal places to avoid floating point serialization issues. Use -1 to specify
  /// that all decimal places should be used (the default).
  void set_compare_float_precision(int value) {
    writer_actual_.set_float_precision(value);
    writer_expected_.set_float_precision(value);
  }

  /// \brief Returns the number of differences found by the previous call
  int64_t num_differences() const;

  /// \brief Dump a human-readable summary of differences to out
  void WriteDifferences(std::ostream& out);

  /// \brief Clear any existing differences
  void ClearDifferences();

  /// \brief Compare a stream of record batches
  ///
  /// Compares actual against expected using the following strategy:
  ///
  /// - Compares schemas for equality, returning if differences were found
  /// - Compares pairs of record batches, returning if one stream finished
  ///   before another.
  ///
  /// Returns NANOARROW_OK if the comparison ran without error. Callers must
  /// query num_differences() to obtain the result of the comparison on success.
  ArrowErrorCode CompareArrayStream(ArrowArrayStream* actual, ArrowArrayStream* expected,
                                    ArrowError* error = nullptr);

  /// \brief Compare a top-level ArrowSchema struct
  ///
  /// Returns NANOARROW_OK if the comparison ran without error. Callers must
  /// query num_differences() to obtain the result of the comparison on success.
  ArrowErrorCode CompareSchema(const ArrowSchema* actual, const ArrowSchema* expected,
                               ArrowError* error = nullptr, const std::string& path = "");

  /// \brief Set the ArrowSchema to be used to for future calls to CompareBatch().
  ArrowErrorCode SetSchema(const ArrowSchema* schema, ArrowError* error = nullptr);

  /// \brief Compare a top-level ArrowArray struct
  ///
  /// Returns NANOARROW_OK if the comparison ran without error. Callers must
  /// query num_differences() to obtain the result of the comparison on success.
  ArrowErrorCode CompareBatch(const ArrowArray* actual, const ArrowArray* expected,
                              ArrowError* error = nullptr, const std::string& path = "");

 private:
  TestingJSONWriter writer_actual_;
  TestingJSONWriter writer_expected_;
  internal::Differences* differences_;
  struct ArrowSchema schema_;
  struct ArrowArrayView actual_;
  struct ArrowArrayView expected_;

  // Comparison options
  bool compare_batch_flags_;
  bool compare_metadata_order_;

  ArrowErrorCode CompareField(ArrowSchema* actual, ArrowSchema* expected,
                              ArrowError* error, const std::string& path = "");

  ArrowErrorCode CompareFieldBase(ArrowSchema* actual, ArrowSchema* expected,
                                  ArrowError* error, const std::string& path = "");

  ArrowErrorCode CompareMetadata(const char* actual, const char* expected,
                                 ArrowError* error, const std::string& path = "");

  ArrowErrorCode MetadataEqualKeyValue(const char* actual, const char* expected,
                                       bool* out, ArrowError* error);

  ArrowErrorCode CompareColumn(ArrowSchema* schema, ArrowArrayView* actual,
                               ArrowArrayView* expected, ArrowError* error,
                               const std::string& path = "");
};

/// @}

}  // namespace testing
}  // namespace nanoarrow

#endif
