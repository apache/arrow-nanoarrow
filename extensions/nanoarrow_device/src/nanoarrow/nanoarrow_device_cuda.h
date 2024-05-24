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

#ifndef NANOARROW_DEVICE_CUDA_H_INCLUDED
#define NANOARROW_DEVICE_CUDA_H_INCLUDED

#include "nanoarrow/nanoarrow_device.h"

#ifdef NANOARROW_NAMESPACE

#define ArrowDeviceCuda NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceCuda)

#endif

#ifdef __cplusplus
extern "C" {
#endif

/// \defgroup nanoarrow_device_cuda CUDA Device extension
///
/// A CUDA (i.e., `cuda_runtime_api.h`) implementation of the Arrow C Device
/// interface.
///
/// @{

/// \brief Get a CUDA device from type and ID
///
/// device_type must be one of ARROW_DEVICE_CUDA or ARROW_DEVICE_CUDA_HOST;
/// device_id must be between 0 and cudaGetDeviceCount - 1.
struct ArrowDevice* ArrowDeviceCuda(ArrowDeviceType device_type, int64_t device_id);

/// @}

#ifdef __cplusplus
}
#endif

#endif
