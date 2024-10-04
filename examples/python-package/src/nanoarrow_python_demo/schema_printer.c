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

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "nanoarrow/nanoarrow.h"

static int PrintSchema(const struct ArrowSchema* schema) {
  for (int64_t i = 0; i < schema->n_children; i++) {
    const struct ArrowSchema* child = schema->children[i];
    struct ArrowSchemaView schema_view;
    struct ArrowError error;
    if (ArrowSchemaViewInit(&schema_view, child, &error) != NANOARROW_OK) {
      PyErr_SetString(PyExc_RuntimeError, error.message);
      return -1;
    }
    printf("Field: %s, Type: %s\n", child->name, ArrowTypeString(schema_view.type));
  }

  return 0;
}

static PyObject* PrintArrowSchema(PyObject* Py_UNUSED(self), PyObject* args) {
  PyObject* obj;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }

  PyObject* capsule = PyObject_CallMethod(obj, "__arrow_c_schema__", NULL);
  if (capsule == NULL) {
    PyErr_SetString(PyExc_TypeError,
                    "Could not call '__arrow_c_schema__' on provided object");
    return NULL;
  }

  const struct ArrowSchema* schema =
      (const struct ArrowSchema*)PyCapsule_GetPointer(capsule, "arrow_schema");
  if (schema == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Could not retrieve 'arrow_schema' pointer");
    Py_DECREF(capsule);
    return NULL;
  }

  int error = PrintSchema(schema);
  Py_DECREF(capsule);
  if (error) {
    return NULL;
  }

  return Py_None;
}

static PyMethodDef schema_printer_methods[] = {
    {.ml_name = "print_schema",
     .ml_meth = (PyCFunction)PrintArrowSchema,
     .ml_flags = METH_VARARGS,
     .ml_doc = PyDoc_STR("Prints an Arrow Schema")},
    {}  // sentinel
};

static PyModuleDef schema_printer_def = {.m_base = PyModuleDef_HEAD_INIT,
                                         .m_name = "schema_printer",
                                         .m_methods = schema_printer_methods};

PyMODINIT_FUNC PyInit_schema_printer(void) {
  return PyModuleDef_Init(&schema_printer_def);
}
