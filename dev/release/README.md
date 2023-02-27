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

Full verification requires [CMake](https://cmake.org/download/) to build and run the test
suite. The test suite currently depends on an Arrow C++ installation that is discoverable
by CMake (e.g., using one of the methods described in the
[Arrow installation instructions](https://arrow.apache.org/install/)). For environments
where binary packages are not provided, building and installing Arrow C++ from source
may be required. You can provide the `NANOARROW_CMAKE_OPTIONS` environment variable to
pass extra arguments to `cmake` (e.g., `-DArrow_DIR=path/to/arrow/lib/cmake/Arrow` or
`-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake`).

Verification of the R package requires an
[R installation](https://cloud.r-project.org/) and a C/C++ compiler (e.g.,
[RTools](https://cloud.r-project.org/bin/windows/Rtools/) on Windows or XCode Command
Line Tools). You can set the `R_HOME` environment variable or
`export PATH="$PATH:/path/to/R/bin"` to point to a specific R installation.

The verification script itself is written in `bash` and requires the `curl`, `gpg`, and
`shasum`/`sha512sum` commands. These are typically available from a packaage
manager except on Windows (see below).

### MacOS

On MacOS you can install all requirements except R using [Homebrew](https://brew.sh):

```bash
brew install cmake gnupg apache-arrow
```

You can install R using the instructions provided on the
[R Project Download page](https://cloud.r-project.org/bin/macosx/).

### Debian/Ubuntu

On Debian/Ubuntu (e.g., `docker run ubuntu:latest`) you can install prerequisites using `apt`:

```bash
apt update
sudo apt install -y git cmake r-base gnupg curl

# For Arrow C++
apt install -y -V ca-certificates lsb-release wget
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt update
sudo apt install -y -V libarrow-dev
```

### Fedora

On recent Fedora (e.g., `docker run --rm -it fedora:latest`), you can install all prerequisites
using `dnf`:

```bash
dnf install -y git cmake R gnupg curl libarrow-devel
```

### Alpine Linux

On Alpine Linux (e.g., `docker run --rm -it alpine:latest`), all prerequisites are available
using `apk add` except Arrow C++ which must be built and installed from source.

```bash
apk add bash linux-headers git cmake R-dev g++ gnupg curl

# Build Arrow C++ from source
curl https://dlcdn.apache.org/arrow/arrow-11.0.0/apache-arrow-11.0.0.tar.gz | \
  tar -zxf -
mkdir arrow-build && cd arrow-build
cmake ../apache-arrow-11.0.0/cpp \
    -DARROW_JEMALLOC=OFF -DARROW_SIMD_LEVEL=NONE -DCMAKE_INSTALL_PREFIX=../arrow
cmake --build .
cmake --install . --prefix=../arrow
cd ..
```

### Windows

To verify the C library currently you will need to either use Conda or
[build Arrow from source](https://arrow.apache.org/docs/dev/developers/cpp/windows.html).
You can verify the R package only using the MSys2 bash shell (?).

- Install MSys2/bash
- Install CMake
- Build Arrow from source
- `export NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=.../dist/lib/cmake/Arrow"`

### Conda
