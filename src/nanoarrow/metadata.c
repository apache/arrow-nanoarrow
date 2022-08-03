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

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow.h"

ArrowErrorCode ArrowMetadataReaderInit(struct ArrowMetadataReader* reader,
                                       const char* metadata) {
  reader->metadata = metadata;

  if (reader->metadata == NULL) {
    reader->offset = 0;
    reader->remaining_keys = 0;
  } else {
    memcpy(&reader->remaining_keys, reader->metadata, sizeof(int32_t));
    reader->offset = sizeof(int32_t);
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowMetadataReaderRead(struct ArrowMetadataReader* reader,
                                       struct ArrowStringView* key_out,
                                       struct ArrowStringView* value_out) {
  if (reader->remaining_keys <= 0) {
    return EINVAL;
  }

  int64_t pos = 0;

  int32_t key_size;
  memcpy(&key_size, reader->metadata + reader->offset + pos, sizeof(int32_t));
  pos += sizeof(int32_t);

  key_out->data = reader->metadata + reader->offset + pos;
  key_out->n_bytes = key_size;
  pos += key_size;

  int32_t value_size;
  memcpy(&value_size, reader->metadata + reader->offset + pos, sizeof(int32_t));
  pos += sizeof(int32_t);

  value_out->data = reader->metadata + reader->offset + pos;
  value_out->n_bytes = value_size;
  pos += value_size;

  reader->offset += pos;
  reader->remaining_keys--;
  return NANOARROW_OK;
}

int64_t ArrowMetadataSizeOf(const char* metadata) {
  if (metadata == NULL) {
    return 0;
  }

  struct ArrowMetadataReader reader;
  struct ArrowStringView key;
  struct ArrowStringView value;
  ArrowMetadataReaderInit(&reader, metadata);

  int64_t size = sizeof(int32_t);
  while (ArrowMetadataReaderRead(&reader, &key, &value) == NANOARROW_OK) {
    size += sizeof(int32_t) + key.n_bytes + sizeof(int32_t) + value.n_bytes;
  }

  return size;
}

ArrowErrorCode ArrowMetadataGetValue(const char* metadata, const char* key,
                                     const char* default_value,
                                     struct ArrowStringView* value_out) {
  struct ArrowStringView target_key_view = {key, strlen(key)};
  value_out->data = default_value;
  if (default_value != NULL) {
    value_out->n_bytes = strlen(default_value);
  } else {
    value_out->n_bytes = 0;
  }

  struct ArrowMetadataReader reader;
  struct ArrowStringView key_view;
  struct ArrowStringView value;
  ArrowMetadataReaderInit(&reader, metadata);

  int64_t size = sizeof(int32_t);
  while (ArrowMetadataReaderRead(&reader, &key_view, &value) == NANOARROW_OK) {
    int key_equal = target_key_view.n_bytes == key_view.n_bytes &&
                    strncmp(target_key_view.data, key_view.data, key_view.n_bytes) == 0;
    if (key_equal) {
      value_out->data = value.data;
      value_out->n_bytes = value.n_bytes;
      break;
    }
  }

  return NANOARROW_OK;
}

char ArrowMetadataHasKey(const char* metadata, const char* key) {
  struct ArrowStringView value;
  ArrowMetadataGetValue(metadata, key, NULL, &value);
  return value.data != NULL;
}

ArrowErrorCode ArrowMetadataBuilderInit(struct ArrowBuffer* buffer,
                                        const char* metadata) {
  ArrowBufferInit(buffer);
  int result = ArrowBufferAppend(buffer, metadata, ArrowMetadataSizeOf(metadata));
  if (result != NANOARROW_OK) {
    return result;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowMetadataBuilderAppendView(struct ArrowBuffer* buffer,
                                              struct ArrowStringView* key,
                                              struct ArrowStringView* value) {
  if (value == NULL) {
    return NANOARROW_OK;
  }

  int result;

  if (buffer->capacity_bytes == 0) {
    int32_t zero = 0;
    result = ArrowBufferAppend(buffer, &zero, sizeof(int32_t));
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  if (buffer->capacity_bytes < sizeof(int32_t)) {
    return EINVAL;
  }

  int32_t n_keys;
  memcpy(&n_keys, buffer->data, sizeof(int32_t));

  int32_t key_size = key->n_bytes;
  int32_t value_size = value->n_bytes;
  result = ArrowBufferReserve(buffer,
                              sizeof(int32_t) + key_size + sizeof(int32_t) + value_size);
  if (result != NANOARROW_OK) {
    return result;
  }

  ArrowBufferAppendUnsafe(buffer, &key_size, sizeof(int32_t));
  ArrowBufferAppendUnsafe(buffer, key->data, key_size);
  ArrowBufferAppendUnsafe(buffer, &value_size, sizeof(int32_t));
  ArrowBufferAppendUnsafe(buffer, value->data, value_size);

  n_keys++;
  memcpy(buffer->data, &n_keys, sizeof(int32_t));

  return NANOARROW_OK;
}

ArrowErrorCode ArrowMetadataBuilderAppend(struct ArrowBuffer* buffer, const char* key,
                                          const char* value) {
  struct ArrowStringView key_view = {key, strlen(key)};

  if (value == NULL) {
    return ArrowMetadataBuilderAppendView(buffer, &key_view, NULL);
  } else {
    struct ArrowStringView value_view = {value, strlen(value)};
    return ArrowMetadataBuilderAppendView(buffer, &key_view, &value_view);
  }
}

ArrowErrorCode ArrowMetadataBuilderSetView(struct ArrowBuffer* buffer,
                                           struct ArrowStringView* key,
                                           struct ArrowStringView* value) {
  struct ArrowMetadataReader reader;
  int result = ArrowMetadataReaderInit(&reader, (const char*)buffer->data);
  if (result != NANOARROW_OK) {
    return result;
  }

  struct ArrowBuffer new_buffer;
  result = ArrowMetadataBuilderInit(&new_buffer, NULL);
  if (result != NANOARROW_OK) {
    return result;
  }

  struct ArrowStringView existing_key;
  struct ArrowStringView existing_value;

  while (reader.remaining_keys > 0) {
    result = ArrowMetadataReaderRead(&reader, &existing_key, &existing_value);
    if (result != NANOARROW_OK) {
      ArrowBufferReset(&new_buffer);
      return result;
    }

    if (key->n_bytes == existing_key.n_bytes &&
        strncmp((const char*)key->data, (const char*)existing_key.data,
                existing_key.n_bytes) == 0) {
      result = ArrowMetadataBuilderAppendView(&new_buffer, key, value);
      value = NULL;
    } else {
      result =
          ArrowMetadataBuilderAppendView(&new_buffer, &existing_key, &existing_value);
    }

    if (result != NANOARROW_OK) {
      ArrowBufferReset(&new_buffer);
      return result;
    }
  }

  if (value != NULL) {
    result = ArrowMetadataBuilderAppendView(&new_buffer, key, value);
    if (result != NANOARROW_OK) {
      ArrowBufferReset(&new_buffer);
      return result;
    }
  }

  ArrowBufferReset(buffer);
  ArrowBufferMove(&new_buffer, buffer);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowMetadataBuilderSet(struct ArrowBuffer* buffer, const char* key,
                                       const char* value) {
  struct ArrowStringView key_view = {key, strlen(key)};

  if (value == NULL) {
    return ArrowMetadataBuilderSetView(buffer, &key_view, NULL);
  } else {
    struct ArrowStringView value_view = {value, strlen(value)};
    return ArrowMetadataBuilderSetView(buffer, &key_view, &value_view);
  }
}
