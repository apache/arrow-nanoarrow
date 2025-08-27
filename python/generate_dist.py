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
import pathlib
import shutil


def main():
    src_dir = pathlib.Path(os.environ["MESON_SOURCE_ROOT"]).parent.resolve()
    dist_dir = pathlib.Path(os.environ["MESON_DIST_ROOT"]).resolve()
    subproj_dir = dist_dir / "subprojects" / "arrow-nanoarrow"

    if subproj_dir.is_symlink():
        subproj_dir.unlink()

    subproj_dir.mkdir(parents=True)
    shutil.copy(src_dir / "meson.build", subproj_dir / "meson.build")
    shutil.copy(src_dir / "meson_options.txt", subproj_dir / "meson_options.txt")

    # Copy over any subproject dependency / wrap files
    subproj_subproj_dir = subproj_dir / "subprojects"
    subproj_subproj_dir.mkdir()
    for f in (src_dir / "subprojects").glob("*.wrap"):
        shutil.copy(f, subproj_subproj_dir / f.name)
    shutil.copytree(
        src_dir / "subprojects" / "packagefiles", subproj_subproj_dir / "packagefiles"
    )

    target_src_dir = subproj_dir / "src"
    shutil.copytree(src_dir / "src", target_src_dir)

    # CMake isn't actually required for building, but the bundle.py script reads from
    # its configuration
    shutil.copy(src_dir / "CMakeLists.txt", subproj_dir / "CMakeLists.txt")

    subproj_ci_scripts_dir = subproj_dir / "ci" / "scripts"
    subproj_ci_scripts_dir.mkdir(parents=True)
    shutil.copy(
        src_dir / "ci" / "scripts" / "bundle.py", subproj_ci_scripts_dir / "bundle.py"
    )


if __name__ == "__main__":
    main()
