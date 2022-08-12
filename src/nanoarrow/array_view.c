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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow.h"

void ArrowArrayViewInit(struct ArrowArrayView* array_view, enum ArrowType storage_type) {
  memset(array_view, 0, sizeof(struct ArrowArrayView));
  array_view->storage_type = storage_type;
  ArrowLayoutInit(&array_view->layout, storage_type);
}

ArrowErrorCode ArrowArrayViewAllocateChildren(struct ArrowArrayView* array_view,
                                              int64_t n_children) {
  if (array_view->children != NULL) {
    return EINVAL;
  }

  array_view->children =
      (struct ArrowArrayView**)ArrowMalloc(n_children * sizeof(struct ArrowArrayView*));
  if (array_view->children == NULL) {
    return ENOMEM;
  }

  for (int64_t i = 0; i < n_children; i++) {
    array_view->children[i] = NULL;
  }

  array_view->n_children = n_children;

  for (int64_t i = 0; i < n_children; i++) {
    array_view->children[i] =
        (struct ArrowArrayView*)ArrowMalloc(sizeof(struct ArrowArrayView));
    if (array_view->children[i] == NULL) {
      return ENOMEM;
    }
    ArrowArrayViewInit(array_view->children[i], NANOARROW_TYPE_UNINITIALIZED);
  }

  return NANOARROW_OK;
}

void ArrowArrayViewReset(struct ArrowArrayView* array_view) {
  if (array_view->children != NULL) {
    for (int64_t i = 0; i < array_view->n_children; i++) {
      if (array_view->children[i] != NULL) {
        ArrowArrayViewReset(array_view->children[i]);
        ArrowFree(array_view->children[i]);
      }
    }

    ArrowFree(array_view->children);
  }

  ArrowArrayViewInit(array_view, NANOARROW_TYPE_UNINITIALIZED);
}
