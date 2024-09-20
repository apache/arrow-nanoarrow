#!/usr/bin/env python

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

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup


# https://github.com/jbweston/miniver
def get_version(pkg_path):
    """
    Load version.py module without importing the whole package.

    Template code from miniver.
    """
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(pkg_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__


version = get_version("src/nanoarrow")


# Run bootstrap.py to run cmake generating a fresh bundle based on this
# checkout or copy from ../dist if the caller doesn't have cmake available.
# Note that bootstrap.py won't exist if building from sdist.
this_dir = os.path.dirname(__file__)
bootstrap_py = os.path.join(this_dir, "bootstrap.py")
if os.path.exists(bootstrap_py):
    subprocess.run([sys.executable, bootstrap_py])


# Set some extra flags for compiling with coverage support
extra_compile_args = []
extra_link_args = []
extra_define_macros = []
device_include_dirs = []
device_library_dirs = []
device_libraries = []
device_define_macros = []

if os.getenv("NANOARROW_PYTHON_COVERAGE") == "1":
    extra_compile_args.append("--coverage")
    extra_link_args.append("--coverage")
    extra_define_macros.append(("CYTHON_TRACE", 1))

if os.getenv("NANOARROW_DEBUG_EXTENSION") == "1":
    extra_compile_args.extend(["-g", "-O0"])

cuda_toolkit_root = os.getenv("NANOARROW_PYTHON_CUDA_HOME")
if cuda_toolkit_root:
    cuda_lib = "cuda.lib" if os.name == "nt" else "libcuda.so"
    include_dir = Path(cuda_toolkit_root) / "include"
    possible_libs = [
        Path(cuda_toolkit_root) / "lib" / cuda_lib,
        Path(cuda_toolkit_root) / "lib64" / cuda_lib,
        Path(cuda_toolkit_root) / "lib" / "x64" / cuda_lib,
        Path("/usr/lib/wsl/lib") / cuda_lib,
    ]

    if not include_dir.is_dir():
        raise ValueError(f"CUDA include directory does not exist: '{include_dir}'")

    device_include_dirs.append(str(include_dir))
    device_libraries.append("cuda")
    extra_define_macros.append(("NANOARROW_DEVICE_WITH_CUDA", 1))

    # Library might be already in a system library directory such that no -L flag
    # is needed
    lib_dirs = [d for d in possible_libs if d.exists()]
    if lib_dirs:
        device_library_dirs.append(str(lib_dirs[0].parent))


# This mechanism to build a static c library against which extensions
# can be linked is not well documented but is a better solution than
# simply including these files as sources to the extensions that need
# them. A more robust solution would be to use Meson or CMake to build
# the Python extensions since they can both build a shared nanoarrow
# and link it. This mechanism is the build_clib command available in
# setuptools (and previously from distutils).
common_libraries = [
    [
        "nanoarrow_python_shared",
        {
            "sources": [
                "vendor/nanoarrow.c",
                "vendor/nanoarrow_device.c",
                "vendor/nanoarrow_ipc.c",
                "vendor/flatcc.c",
            ],
            "include_dirs": ["vendor"] + device_include_dirs,
            "libraries": device_libraries,
            "library_dirs": device_library_dirs,
            "macros": extra_define_macros + device_define_macros,
        },
    ]
]


def nanoarrow_extension(name, *, link_device=False):
    return Extension(
        name=name,
        include_dirs=["vendor", "src/nanoarrow"],
        language="c",
        sources=["src/" + name.replace(".", "/") + ".pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=extra_define_macros + device_define_macros,
        libraries=["nanoarrow_python_shared"] + device_libraries if link_device else [],
    )


setup(
    ext_modules=[
        nanoarrow_extension("nanoarrow._types"),
        nanoarrow_extension("nanoarrow._utils"),
        nanoarrow_extension("nanoarrow._device", link_device=True),
        nanoarrow_extension("nanoarrow._array", link_device=True),
        nanoarrow_extension("nanoarrow._array_stream"),
        nanoarrow_extension("nanoarrow._buffer"),
        nanoarrow_extension("nanoarrow._ipc_lib"),
        nanoarrow_extension("nanoarrow._schema"),
    ],
    version=version,
    libraries=common_libraries,
)
