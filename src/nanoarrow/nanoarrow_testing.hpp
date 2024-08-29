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

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

#include "nanoarrow/nanoarrow.hpp"

#ifndef NANOARROW_TESTING_HPP_INCLUDED
#define NANOARROW_TESTING_HPP_INCLUDED

/// \defgroup nanoarrow_testing Nanoarrow Testing Helpers
///
/// Utilities for testing nanoarrow structures and functions.

namespace nanoarrow {

namespace testing {

namespace internal {

// Internal representation of the various structures needed to import and/or export
// a dictionary array. We use a serialized version of the dictionary value because
// nanoarrow doesn't currently have the ability to copy or reference count an Array.
struct Dictionary {
  nanoarrow::UniqueSchema schema;
  int64_t column_length;
  std::string column_json;
};

class DictionaryContext {
 public:
  DictionaryContext() : next_id_(0) {}

  ArrowErrorCode RecordSchema(int32_t dictionary_id, const ArrowSchema* values_schema) {
    if (!HasDictionaryForId(dictionary_id)) {
      dictionaries_[dictionary_id] = internal::Dictionary();
      NANOARROW_RETURN_NOT_OK(
          ArrowSchemaDeepCopy(values_schema, dictionaries_[dictionary_id].schema.get()));
    }

    dictionary_ids_[values_schema] = dictionary_id;
    return NANOARROW_OK;
  }

  ArrowErrorCode RecordSchema(const ArrowSchema* values_schema, int32_t* dictionary_id) {
    while (HasDictionaryForId(next_id_)) {
      next_id_++;
    }

    NANOARROW_RETURN_NOT_OK(RecordSchema(next_id_, values_schema));
    *dictionary_id = next_id_++;
    return NANOARROW_OK;
  }

  void RecordArray(int32_t dictionary_id, int64_t length, std::string column_json) {
    dictionaries_[dictionary_id].column_length = length;
    dictionaries_[dictionary_id].column_json = std::move(column_json);
  }

  void RecordArray(const ArrowSchema* values_schema, int64_t length,
                   std::string column_json) {
    auto ids_it = dictionary_ids_.find(values_schema);
    RecordArray(ids_it->second, length, column_json);
  }

  bool empty() { return dictionaries_.empty(); }

  void clear() {
    dictionaries_.clear();
    dictionary_ids_.clear();
    next_id_ = 0;
  }

  bool HasDictionaryForSchema(const ArrowSchema* values_schema) const {
    return dictionary_ids_.find(values_schema) != dictionary_ids_.end();
  }

  bool HasDictionaryForId(int32_t dictionary_id) const {
    return dictionaries_.find(dictionary_id) != dictionaries_.end();
  }

  const Dictionary& Get(int32_t dictionary_id) const {
    auto dict_it = dictionaries_.find(dictionary_id);
    return dict_it->second;
  }

  const Dictionary& Get(const ArrowSchema* values_schema) const {
    auto ids_it = dictionary_ids_.find(values_schema);
    return Get(ids_it->second);
  }

  const std::vector<int32_t> GetAllIds() const {
    std::vector<int32_t> out;
    out.reserve(dictionaries_.size());
    for (const auto& value : dictionaries_) {
      out.push_back(value.first);
    }
    return out;
  }

 private:
  int32_t next_id_;
  std::unordered_map<int32_t, Dictionary> dictionaries_;
  std::unordered_map<const ArrowSchema*, int32_t> dictionary_ids_;
};

}  // namespace internal

/// \defgroup nanoarrow_testing-json Integration test helpers
///
/// See testing format documentation for details of the JSON representation. This
/// representation is not canonical but can be used to implement integration tests with
/// other implementations.
///
/// @{

/// \brief Writer for the Arrow integration testing JSON format
class TestingJSONWriter {
 public:
  TestingJSONWriter() : float_precision_(-1), include_metadata_(true) {}

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

  void ResetDictionaries() { dictionaries_.clear(); }

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
  internal::DictionaryContext dictionaries_;

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
class TestingJSONReader {
 public:
  TestingJSONReader(ArrowBufferAllocator allocator) : allocator_(allocator) {}
  TestingJSONReader() : TestingJSONReader(ArrowBufferAllocatorDefault()) {}

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
  internal::DictionaryContext dictionaries_;

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
class TestingJSONComparison {
 private:
  // Internal representation of a human-readable inequality
  struct Difference {
    std::string path;
    std::string actual;
    std::string expected;
  };

 public:
  TestingJSONComparison() : compare_batch_flags_(true), compare_metadata_order_(true) {
    // We do our own metadata comparison
    writer_actual_.set_include_metadata(false);
    writer_expected_.set_include_metadata(false);
  }

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
  int64_t num_differences() const { return differences_.size(); }

  /// \brief Dump a human-readable summary of differences to out
  void WriteDifferences(std::ostream& out) {
    for (const auto& difference : differences_) {
      out << "Path: " << difference.path << "\n";
      out << "- " << difference.actual << "\n";
      out << "+ " << difference.expected << "\n";
      out << "\n";
    }
  }

  /// \brief Clear any existing differences
  void ClearDifferences() { differences_.clear(); }

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
                                    ArrowError* error = nullptr) {
    // Read both schemas
    nanoarrow::UniqueSchema actual_schema;
    nanoarrow::UniqueSchema expected_schema;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayStreamGetSchema(actual, actual_schema.get(), error));
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayStreamGetSchema(expected, expected_schema.get(), error));

    // Compare them and return if they are not equal
    NANOARROW_RETURN_NOT_OK(
        CompareSchema(expected_schema.get(), actual_schema.get(), error, "Schema"));
    if (num_differences() > 0) {
      return NANOARROW_OK;
    }

    // Keep a record of the schema to compare batches
    NANOARROW_RETURN_NOT_OK(SetSchema(expected_schema.get(), error));

    int64_t n_batches = -1;
    nanoarrow::UniqueArray actual_array;
    nanoarrow::UniqueArray expected_array;
    do {
      n_batches++;
      std::string batch_label = std::string("Batch ") + std::to_string(n_batches);

      // Read a batch from each stream
      actual_array.reset();
      expected_array.reset();
      NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetNext(actual, actual_array.get(), error));
      NANOARROW_RETURN_NOT_OK(
          ArrowArrayStreamGetNext(expected, expected_array.get(), error));

      // Check the finished/unfinished status of both streams
      if (actual_array->release == nullptr && expected_array->release != nullptr) {
        differences_.push_back({batch_label, "finished stream", "unfinished stream"});
        return NANOARROW_OK;
      }

      if (actual_array->release != nullptr && expected_array->release == nullptr) {
        differences_.push_back({batch_label, "unfinished stream", "finished stream"});
        return NANOARROW_OK;
      }

      // If both streams are done, break
      if (actual_array->release == nullptr) {
        break;
      }

      // Compare this batch
      NANOARROW_RETURN_NOT_OK(
          CompareBatch(actual_array.get(), expected_array.get(), error, batch_label));
    } while (true);

    return NANOARROW_OK;
  }

  /// \brief Compare a top-level ArrowSchema struct
  ///
  /// Returns NANOARROW_OK if the comparison ran without error. Callers must
  /// query num_differences() to obtain the result of the comparison on success.
  ArrowErrorCode CompareSchema(const ArrowSchema* actual, const ArrowSchema* expected,
                               ArrowError* error = nullptr,
                               const std::string& path = "") {
    writer_actual_.ResetDictionaries();
    writer_expected_.ResetDictionaries();

    // Compare the top-level schema "manually" because (1) map type needs special-cased
    // comparison and (2) it's easier to read the output if differences are separated
    // by field.
    ArrowSchemaView actual_view;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaViewInit(&actual_view, actual, nullptr),
                                       error);

    ArrowSchemaView expected_view;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaViewInit(&expected_view, expected, nullptr), error);

    if (actual_view.type != NANOARROW_TYPE_STRUCT ||
        expected_view.type != NANOARROW_TYPE_STRUCT) {
      ArrowErrorSet(error, "Top-level schema must be struct");
      return EINVAL;
    }

    // (Purposefully ignore the name field at the top level)

    // Compare flags
    if (compare_batch_flags_ && actual->flags != expected->flags) {
      differences_.push_back({path,
                              std::string(".flags: ") + std::to_string(actual->flags),
                              std::string(".flags: ") + std::to_string(expected->flags)});
    }

    // Compare children
    if (actual->n_children != expected->n_children) {
      differences_.push_back(
          {path, std::string(".n_children: ") + std::to_string(actual->n_children),
           std::string(".n_children: ") + std::to_string(expected->n_children)});
    } else {
      for (int64_t i = 0; i < expected->n_children; i++) {
        NANOARROW_RETURN_NOT_OK(CompareField(
            actual->children[i], expected->children[i], error,
            path + std::string(".children[") + std::to_string(i) + std::string("]")));
      }
    }

    // Compare metadata
    NANOARROW_RETURN_NOT_OK(CompareMetadata(actual->metadata, expected->metadata, error,
                                            path + std::string(".metadata")));

    return NANOARROW_OK;
  }

  /// \brief Set the ArrowSchema to be used to for future calls to CompareBatch().
  ArrowErrorCode SetSchema(const ArrowSchema* schema, ArrowError* error = nullptr) {
    schema_.reset();
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaDeepCopy(schema, schema_.get()), error);
    actual_.reset();
    expected_.reset();

    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewInitFromSchema(actual_.get(), schema_.get(), error));
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewInitFromSchema(expected_.get(), schema_.get(), error));

    if (actual_->storage_type != NANOARROW_TYPE_STRUCT) {
      ArrowErrorSet(error, "Can't SetSchema() with non-struct");
      return EINVAL;
    }

    // "Write" the schema using both writers to ensure dictionary ids can be resolved
    // using the ArrowSchema* pointers from schema_
    std::stringstream ss;
    writer_actual_.ResetDictionaries();
    writer_expected_.ResetDictionaries();
    writer_actual_.WriteSchema(ss, schema_.get());
    writer_expected_.WriteSchema(ss, schema_.get());

    return NANOARROW_OK;
  }

  /// \brief Compare a top-level ArrowArray struct
  ///
  /// Returns NANOARROW_OK if the comparison ran without error. Callers must
  /// query num_differences() to obtain the result of the comparison on success.
  ArrowErrorCode CompareBatch(const ArrowArray* actual, const ArrowArray* expected,
                              ArrowError* error = nullptr, const std::string& path = "") {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArray(expected_.get(), expected, error));
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArray(actual_.get(), actual, error));

    if (actual->offset != expected->offset) {
      differences_.push_back({path, ".offset: " + std::to_string(actual->offset),
                              ".offset: " + std::to_string(expected->offset)});
    }

    if (actual->length != expected->length) {
      differences_.push_back({path, ".length: " + std::to_string(actual->length),
                              ".length: " + std::to_string(expected->length)});
    }

    // ArrowArrayViewSetArray() ensured that number of children of both match schema
    for (int64_t i = 0; i < expected_->n_children; i++) {
      NANOARROW_RETURN_NOT_OK(CompareColumn(
          schema_->children[i], actual_->children[i], expected_->children[i], error,
          path + std::string(".children[") + std::to_string(i) + "]"));
    }

    return NANOARROW_OK;
  }

 private:
  TestingJSONWriter writer_actual_;
  TestingJSONWriter writer_expected_;
  std::vector<Difference> differences_;
  nanoarrow::UniqueSchema schema_;
  nanoarrow::UniqueArrayView actual_;
  nanoarrow::UniqueArrayView expected_;

  // Comparison options
  bool compare_batch_flags_;
  bool compare_metadata_order_;

  ArrowErrorCode CompareField(ArrowSchema* actual, ArrowSchema* expected,
                              ArrowError* error, const std::string& path = "") {
    // Preprocess both fields such that map types have canonical names
    nanoarrow::UniqueSchema actual_copy;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaDeepCopy(actual, actual_copy.get()),
                                       error);
    nanoarrow::UniqueSchema expected_copy;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaDeepCopy(expected, expected_copy.get()),
                                       error);

    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ForceMapNamesCanonical(actual_copy.get()), error);
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ForceMapNamesCanonical(expected_copy.get()),
                                       error);
    return CompareFieldBase(actual_copy.get(), expected_copy.get(), error, path);
  }

  ArrowErrorCode CompareFieldBase(ArrowSchema* actual, ArrowSchema* expected,
                                  ArrowError* error, const std::string& path = "") {
    std::stringstream ss;

    NANOARROW_RETURN_NOT_OK_WITH_ERROR(writer_expected_.WriteField(ss, expected), error);
    std::string expected_json = ss.str();

    ss.str("");
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(writer_actual_.WriteField(ss, actual), error);
    std::string actual_json = ss.str();

    if (actual_json != expected_json) {
      differences_.push_back({path, actual_json, expected_json});
    }

    NANOARROW_RETURN_NOT_OK(CompareMetadata(actual->metadata, expected->metadata, error,
                                            path + std::string(".metadata")));
    return NANOARROW_OK;
  }

  ArrowErrorCode CompareMetadata(const char* actual, const char* expected,
                                 ArrowError* error, const std::string& path = "") {
    std::stringstream ss;

    NANOARROW_RETURN_NOT_OK_WITH_ERROR(writer_actual_.WriteMetadata(ss, actual), error);
    std::string actual_json = ss.str();

    ss.str("");
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(writer_expected_.WriteMetadata(ss, expected),
                                       error);
    std::string expected_json = ss.str();

    bool metadata_equal = actual_json == expected_json;

    // If there is a difference in the rendered JSON but we aren't being strict about
    // order, check again using the KeyValue comparison.
    if (!metadata_equal && !compare_metadata_order_) {
      NANOARROW_RETURN_NOT_OK(
          MetadataEqualKeyValue(actual, expected, &metadata_equal, error));
    }

    // If we still have an inequality, add a difference.
    if (!metadata_equal) {
      differences_.push_back({path, actual_json, expected_json});
    }

    return NANOARROW_OK;
  }

  ArrowErrorCode MetadataEqualKeyValue(const char* actual, const char* expected,
                                       bool* out, ArrowError* error) {
    std::unordered_map<std::string, std::string> actual_map, expected_map;
    NANOARROW_RETURN_NOT_OK(MetadataToMap(actual, &actual_map, error));
    NANOARROW_RETURN_NOT_OK(MetadataToMap(expected, &expected_map, error));

    if (actual_map.size() != expected_map.size()) {
      *out = false;
      return NANOARROW_OK;
    }

    for (const auto& item : expected_map) {
      const auto& actual_item = actual_map.find(item.first);
      if (actual_item == actual_map.end()) {
        *out = false;
        return NANOARROW_OK;
      }

      if (actual_item->second != item.second) {
        *out = false;
        return NANOARROW_OK;
      }
    }

    *out = true;
    return NANOARROW_OK;
  }

  ArrowErrorCode MetadataToMap(const char* metadata,
                               std::unordered_map<std::string, std::string>* out,
                               ArrowError* error) {
    ArrowMetadataReader reader;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowMetadataReaderInit(&reader, metadata), error);

    ArrowStringView key, value;
    size_t metadata_num_keys = 0;
    while (reader.remaining_keys > 0) {
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowMetadataReaderRead(&reader, &key, &value),
                                         error);
      out->insert({std::string(key.data, key.size_bytes),
                   std::string(value.data, value.size_bytes)});
      metadata_num_keys++;
    }

    if (metadata_num_keys != out->size()) {
      ArrowErrorSet(error,
                    "Comparison of metadata containing duplicate keys without "
                    "considering order is not implemented");
      return ENOTSUP;
    }

    return NANOARROW_OK;
  }

  ArrowErrorCode CompareColumn(ArrowSchema* schema, ArrowArrayView* actual,
                               ArrowArrayView* expected, ArrowError* error,
                               const std::string& path = "") {
    // Compare children and dictionaries first, then higher-level structures after.
    // This is a redundant because the higher-level serialized JSON will also report
    // a difference if deeply nested children have differences; however, it will not
    // contain dictionaries and this output is slightly better (more targeted differences
    // that are slightly easier to read appear first).
    for (int64_t i = 0; i < schema->n_children; i++) {
      NANOARROW_RETURN_NOT_OK(
          CompareColumn(schema->children[i], actual->children[i], expected->children[i],
                        error, path + ".children[" + std::to_string(i) + "]"));
    }

    if (schema->dictionary != nullptr) {
      NANOARROW_RETURN_NOT_OK(CompareColumn(schema->dictionary, actual->dictionary,
                                            expected->dictionary, error,
                                            path + ".dictionary"));
    }

    std::stringstream ss;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(writer_expected_.WriteColumn(ss, schema, expected),
                                       error);
    std::string expected_json = ss.str();

    ss.str("");
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(writer_actual_.WriteColumn(ss, schema, actual),
                                       error);
    std::string actual_json = ss.str();

    if (actual_json != expected_json) {
      differences_.push_back({path, actual_json, expected_json});
    }

    return NANOARROW_OK;
  }

  ArrowErrorCode ForceMapNamesCanonical(ArrowSchema* schema) {
    ArrowSchemaView view;
    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));

    if (view.type == NANOARROW_TYPE_MAP) {
      NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[0], "entries"));
      NANOARROW_RETURN_NOT_OK(
          ArrowSchemaSetName(schema->children[0]->children[0], "key"));
      NANOARROW_RETURN_NOT_OK(
          ArrowSchemaSetName(schema->children[0]->children[1], "value"));
    }

    for (int64_t i = 0; i < schema->n_children; i++) {
      NANOARROW_RETURN_NOT_OK(ForceMapNamesCanonical(schema->children[i]));
    }

    if (schema->dictionary != nullptr) {
      NANOARROW_RETURN_NOT_OK(ForceMapNamesCanonical(schema->dictionary));
    }

    return NANOARROW_OK;
  }
};

/// @}

}  // namespace testing
}  // namespace nanoarrow

#endif
