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
from pathlib import Path
import subprocess
import sys

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
extra_include_dirs = []
extra_compile_args = []
extra_link_args = []
extra_define_macros = []
library_dirs = []
libraries = []

if os.getenv("NANOARROW_PYTHON_COVERAGE") == "1":
    extra_compile_args.append("--coverage")
    extra_link_args.append("--coverage")
    extra_define_macros.append(("CYTHON_TRACE", 1))

if os.getenv("NANOARROW_DEBUG_EXTENSION") == "1":
    extra_compile_args.extend(["-g", "-O0"])

cuda_toolkit_root = os.getenv("NANOARROW_PYTHON_CUDA_HOME")
if cuda_toolkit_root:
    cuda_lib = "cuda.lib" if os.name == 'nt' else "libcuda.so"
    include_dir = Path(cuda_toolkit_root) / "include"
    possible_libs = [
        Path(cuda_toolkit_root) / "lib" / cuda_lib,
        Path(cuda_toolkit_root) / "lib64" / cuda_lib,
        Path(cuda_toolkit_root) / "lib" / "x64" / cuda_lib,
        Path("/usr/lib/wsl/lib") / cuda_lib,
    ]

    if not include_dir.is_dir():
        raise ValueError(f"CUDA include directory does not exist: '{include_dir}'")

    lib_dirs = [d for d in possible_libs if d.exists()]
    if not lib_dirs:
        lib_dirs_err = ", ".join(f"'{d}" for d in possible_libs)
        raise ValueError(f"Can't find CUDA library directory. Checked {lib_dirs_err}")

    extra_include_dirs.append(str(include_dir))
    library_dirs.append(str(lib_dirs[0].parent))
    libraries.append("cuda")
    extra_define_macros.append(("NANOARROW_DEVICE_WITH_CUDA", 1))


setup(
    ext_modules=[
        Extension(
            name="nanoarrow._types",
            include_dirs=["src/nanoarrow", "vendor"],
            language="c",
            sources=[
                "src/nanoarrow/_types.pyx",
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=extra_define_macros,
        ),
        Extension(
            name="nanoarrow._utils",
            include_dirs=extra_include_dirs + ["src/nanoarrow", "vendor"],
            language="c",
            sources=[
                "src/nanoarrow/_utils.pyx",
                "vendor/nanoarrow.c",
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=extra_define_macros,
        ),
        Extension(
            name="nanoarrow._lib",
            include_dirs=extra_include_dirs + ["src/nanoarrow", "vendor"],
            language="c",
            sources=[
                "src/nanoarrow/_lib.pyx",
                "vendor/nanoarrow.c",
                "vendor/nanoarrow_device.c",
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=extra_define_macros,
            library_dirs=library_dirs,
            libraries=libraries
        ),
        Extension(
            name="nanoarrow._ipc_lib",
            include_dirs=extra_include_dirs + ["src/nanoarrow", "vendor"],
            language="c",
            sources=[
                "src/nanoarrow/_ipc_lib.pyx",
                "vendor/nanoarrow.c",
                "vendor/nanoarrow_ipc.c",
                "vendor/flatcc.c",
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=extra_define_macros,
        ),
    ],
    version=version,
)
