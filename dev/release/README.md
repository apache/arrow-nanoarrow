<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Release process for Apache Arrow nanoarrow

## Verifying a nanoarrow release candidate

Release candidates for nanoarrow are uploaded to https://dist.apache.org/repos/dist/dev/arrow/
prior to a release vote being called on the
[Apache Arrow developer mailing list](https://lists.apache.org/list.html?dev@arrow.apache.org).
A script (`verify-release-candidate.sh`) is provided to verify such a release candidate.
For example, to verify nanoarrow 0.1.0-rc0, one could run:

```
git clone https://github.com/apache/arrow-nanoarrow.git arrow-nanoarrow
cd arrow-nanoarrow/dev/release
./verify-release-candidate.sh 0.1.0 0
```

Full verification requires:

- bash
- curl
- gnupg
- cmake
- Arrow C++
- R

Of these, `cmake` and `R` must be available on `$PATH` and Arrow C++ must be findable by `cmake`.
You can set the `NANOARROW_CMAKE_OPTIONS` environment variable to help CMake find the Arrow
install directory.

### MacOS

On MacOS you can install all requirements except R using [Homebrew](https://brew.sh):

```bash
brew install cmake gnupg apache-arrow
```

On MacOS versions where Homebrew is not available or no longer supported, you must
install [CMake directly from Kitware](https://cmake.org/download/). Building Arrow
C++ on older MacOS may require building OpenSSL from source. Due to CMake version
support, this is may only be possible for MacOS 10.13 or later.

You can install R using the instructions provided on the
[CRAN Download page](https://cloud.r-project.org/bin/macosx/).

### Debian/Ubuntu

On Debian/Ubuntu (e.g., `docker run ubuntu:latest`) you can install prerequisites using `apt`:

```
sudo apt update
sudo apt install -y cmake r-base gnupg curl

# For Arrow C++
sudo apt install -y -V ca-certificates lsb-release wget
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt update
sudo apt install -y -V libarrow-dev
```

### Fedora

### Windows

- Install MSys2/bash
- Install CMake
- Build Arrow from source
- `export NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=.../dist/lib/cmake/Arrow"`

### Conda
