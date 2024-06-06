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

import io
import pathlib
import re


def read_content(path_or_content):
    if isinstance(path_or_content, pathlib.Path):
        with open(path_or_content) as f:
            return f.read()
    else:
        return str(path_or_content)


def configure_content(paths_or_content, args):
    content = read_content(paths_or_content)

    for key, value in args.items():
        replace_key = f"@{key}@"
        if content.count(replace_key) != 1:
            raise ValueError(
                f"Expected exactly one occurrence of '{replace_key}' in '{paths_or_content}'"
            )

        content = content.replace(replace_key, value)

    return content


def concatenate_content(paths_or_content):
    out = io.StringIO()

    for path in paths_or_content:
        out.write(read_content(path))

    return out.getvalue()


def cmakelist_version(path_or_content):
    content = read_content(path_or_content)
    version_match = re.search(r'set\(NANOARROW_VERSION "(.*?)"\)', content)
    if version_match is None:
        raise ValueError(f"Can't find NANOARROW_VERSION in '{path_or_content}'")

    version = version_match.group(1)

    component_match = re.search(r"^([0-9]+)\.([0-9]+)\.([0-9]+)", version)
    return (version,) + tuple(int(component) for component in component_match.groups())
