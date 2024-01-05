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
#
# This file is part of 'miniver': https://github.com/jbweston/miniver

import os

# No public API
__all__ = []

package_root = os.path.dirname(os.path.realpath(__file__))
package_name = os.path.basename(package_root)

STATIC_VERSION_FILE = "_static_version.py"


def get_version(version_file=STATIC_VERSION_FILE):
    override = os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION")
    if override is not None and override != "":
        return override
    version_info = get_static_version_info(version_file)
    version = version_info["version"]
    return version


def get_static_version_info(version_file=STATIC_VERSION_FILE):
    version_info = {}
    with open(os.path.join(package_root, version_file), "rb") as f:
        exec(f.read(), {}, version_info)
    return version_info


__version__ = get_version()

if __name__ == "__main__":
    print("Version: ", get_version())
