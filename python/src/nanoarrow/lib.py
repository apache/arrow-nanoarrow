# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from nanoarrow._lib import ArrayStream, CArray, CArrayView, CSchema


def array_view(obj):
    if isinstance(obj, CArrayView):
        return obj

    return CArrayView.from_cpu_array(carray(obj))


def cschema(obj):
    if isinstance(obj, CSchema):
        return obj

    if hasattr(obj, "__arrow_c_schema__"):
        return CSchema._import_from_c_capsule(obj.__arrow_c_schema__())

    # for pyarrow < 14.0
    if hasattr(obj, "_export_to_c"):
        out = CSchema.allocate()
        obj._export_to_c(out._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} "
            "to nanoarrow.lib.CSchema"
        )


def carray(obj):
    if isinstance(obj, CArray):
        return obj

    if hasattr(obj, "__arrow_c_array__"):
        # TODO support requested schema
        return CArray._import_from_c_capsule(*obj.__arrow_c_array__())

    # for pyarrow < 14.0
    if hasattr(obj, "_export_to_c"):
        out = CArray.allocate(CSchema.allocate())
        obj._export_to_c(out._addr(), out.schema._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} to nanoarrow.lib.CArray"
        )


def array_stream(obj):
    if isinstance(obj, ArrayStream):
        return obj

    if hasattr(obj, "__arrow_c_stream__"):
        # TODO support requested schema
        return ArrayStream._import_from_c_capsule(obj.__arrow_c_stream__())

    # for pyarrow < 14.0
    if hasattr(obj, "_export_to_c"):
        out = ArrayStream.allocate()
        obj._export_to_c(out._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} "
            "to nanoarrow.ArrowArrayStream"
        )
