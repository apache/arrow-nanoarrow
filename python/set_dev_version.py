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

"""Set development version from git

This is used to set the Python package version before building
nightly wheels, since at least some tools do not allow
duplicate versions when uploading sdist/wheels. This requires
that the repo has a full checkout with tags.

The Python package version is hard-coded in a _static_version.py
file. This script calculates the number of commits since the last
apache-arrow-nanoarrow-x.x.x.dev tag and updates the static version
accordingly.
"""

import os
import re
import subprocess


def git(*args):
    out = subprocess.run(["git"] + list(args), stdout=subprocess.PIPE)
    return out.stdout.decode("UTF-8").strip().splitlines()


if __name__ == "__main__":
    last_dev_tag = git(
        "describe", "--match", "apache-arrow-nanoarrow-*.dev", "--tags", "--abbrev=0"
    )[0]
    lines = git("log", "--pretty=oneline", f"{last_dev_tag}..HEAD")
    dev_distance = len(lines)

    this_dir = os.path.dirname(__file__)
    version_file = os.path.join(this_dir, "src", "nanoarrow", "_static_version.py")

    with open(version_file) as f:
        version_content = f.read()

    version_content = re.sub(
        r'"([0-9]+\.[0-9]+\.[0-9]+)\.dev[0-9]+"',
        f'"\\1.dev{dev_distance}"',
        version_content,
    )

    with open(version_file, "w") as f:
        f.write(version_content)
