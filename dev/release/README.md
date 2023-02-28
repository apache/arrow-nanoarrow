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
`export PATH="$PATH:/path/to/R"` (where `$R_HOME/bin/R` is the R executable)
to point to a specific R installation.

The verification script itself is written in `bash` and requires the `curl`, `gpg`, and
`shasum`/`sha512sum` commands. These are typically available from a package
manager except on Windows (see below).

To run only C library verification (requires CMake and Arrow C++ but not R):

```bash
TEST_DEFAULT=0 TEST_C=1 TEST_C_BUNDLED=1 ./verify-release-candidate.sh 0.1.0 0
```

To run only R package verification (requires R but not CMake or Arrow C++):

```bash
TEST_DEFAULT=0 TEST_R=1 ./verify-release-candidate.sh 0.1.0 0
```

### MacOS

On MacOS you can install all requirements except R using [Homebrew](https://brew.sh):

```bash
brew install cmake gnupg apache-arrow
```

For older MacOS or MacOS without Homebrew, you will have to install the XCode
Command Line Tools (i.e., `xcode-select --install`),
[install GnuPG](https://gnupg.org/download/),
[install CMake](https://cmake.org/download/), and build Arrow C++ from source.

```bash
# Download + build Arrow C++
curl https://dlcdn.apache.org/arrow/arrow-11.0.0/apache-arrow-11.0.0.tar.gz | \
  tar -zxf -
mkdir arrow-build && cd arrow-build
cmake ../apache-arrow-11.0.0/cpp \
    -DARROW_JEMALLOC=OFF -DARROW_SIMD_LEVEL=NONE \
    # Required for Arrow on old MacOS
    -DCMAKE_CXX_FLAGS="-D_LIBCPP_DISABLE_AVAILABILITY" \
    -DCMAKE_INSTALL_PREFIX=../arrow
cmake --build .
cmake --install . --prefix=../arrow
cd ..

# Pass location of install to the release verification script
export NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=$(pwd)/arrow/lib/cmake/Arrow -DCMAKE_CXX_FLAGS=-D_LIBCPP_DISABLE_AVAILABILITY"
```

You can install R using the instructions provided on the
[R Project Download page](https://cloud.r-project.org/bin/macosx/).

### Conda (Linux and MacOS)

Using `conda`, one can install all requirements needed for verification on Linux
or MacOS:

```bash
conda create --name nanoarrow-verify-rc
conda activate nanoarrow-verify-rc
conda config --set channel_priority strict

conda install -c conda-forge git cmake gnupg arrow-cpp
```

### Windows

On Windows, prerequisites can be installed using officially provided
installers:
[Visual Studio](https://visualstudio.microsoft.com/vs/),
[CMake](https://cmake.org/download/), and
[Git](https://git-scm.com/downloads) should provide the prerequisties
to verify the C library; R and Rtools can be installed using the
[official R-project installer](https://cloud.r-project.org/bin/windows/).
Arrow C++ can be built from source. The version of bash provided with
Git for Windows can be used to execute the Arrow C++ build commands and
the verification script.

```bash
# Build Arrow C++ from source
curl https://dlcdn.apache.org/arrow/arrow-11.0.0/apache-arrow-11.0.0.tar.gz | \
  tar -zxf -
mkdir arrow-build && cd arrow-build
cmake ../apache-arrow-11.0.0/cpp -DCMAKE_INSTALL_PREFIX=../arrow
cmake --build .
cmake --install . --prefix=../arrow --config=Debug
cd ..

# Pass location of Arrow and R to the verification script
export NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=$(pwd -W)/arrow/lib/cmake/Arrow"
export R_HOME="/c/Program Files/R/R-4.2.2"
```

### Debian/Ubuntu

On Debian/Ubuntu (e.g., `docker run --rm -it ubuntu:latest`) you can install prerequisites using `apt`.

```bash
apt-get update && apt-get install -y git cmake r-base gnupg curl

# For Arrow C++
apt-get install -y -V ca-certificates lsb-release wget
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt-get install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt-get update
apt-get install -y -V libarrow-dev
```

### Fedora

On recent Fedora (e.g., `docker run --rm -it fedora:latest`), you can install all prerequisites
using `dnf`:

```bash
dnf install -y git cmake R gnupg curl libarrow-devel
```

### Arch Linux

On Arch Linux (e.g., `docker run --rm -it archlinux:latest`, you can install all prerequisites
using `pacman`):

```bash
pacman -S git g++ make cmake r-base gnupg curl arrow
```

### Alpine Linux

On Alpine Linux (e.g., `docker run --rm -it alpine:latest`), most prerequisites are available
using `apk add` except Arrow C++ which must be built and installed from source.

```bash
apk add bash linux-headers git cmake R R-dev g++ gnupg curl

# Build Arrow C++ from source
curl https://dlcdn.apache.org/arrow/arrow-11.0.0/apache-arrow-11.0.0.tar.gz | \
  tar -zxf -
mkdir arrow-build && cd arrow-build
cmake ../apache-arrow-11.0.0/cpp \
    -DARROW_JEMALLOC=OFF -DARROW_SIMD_LEVEL=NONE -DCMAKE_INSTALL_PREFIX=../arrow
cmake --build .
cmake --install . --prefix=../arrow
cd ..

# Pass location of Arrow to the verification script
export NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=$(pwd)/arrow/lib/cmake/Arrow"
```

### Centos7

On Centos7 (e.g., `docker run --rm -it centos:7`), most prerequisites are
available via `yum install` except Arrow C++, which must be built from
source. Arrow C++ 9.0.0 was the last version to support the default system
compiler (gcc 4.8).

```bash
yum install epel-release # needed to install R
yum install git gnupg curl R gcc-c++ cmake3

# Needed for R CMD check if the en_US.UTF-8 locale is not defined
# (e.g., in the centos:7 docker image)
# localedef -c -f UTF-8 -i en_US en_US.UTF-8
# export LC_ALL=en_US.UTF-8

# Build Arrow C++ 9.0.0 from source
curl https://dlcdn.apache.org/arrow/arrow-9.0.0/apache-arrow-9.0.0.tar.gz | \
  tar -zxf -
mkdir arrow-build && cd arrow-build
cmake3 ../apache-arrow-9.0.0/cpp \
    -DARROW_JEMALLOC=OFF -DARROW_SIMD_LEVEL=NONE -DCMAKE_INSTALL_PREFIX=../arrow
cmake3 --build .
make install
cd ..

# Pass location of Arrow, cmake, and ctest to the verification script
export NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=$(pwd)/arrow/lib/cmake/Arrow"
export CMAKE_BIN=cmake3
export CTEST_BIN=ctest3
```

### Big endian

One can verify a nanoarrow release candidate on big endian by setting
`DOCKER_DEFAULT_PLATFORM=linux/s390x` and following the instructions for
[Alpine Linux](#alpine-linux).

## Creating a release candidate

The first step to creating a nanoarrow release is to create a `maint-VERSION` branch
and run
[01-prepare.R](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/01-prepare.sh):

```bash
# from the repository root
# 01-prepare.sh <nanoarrow-dir> <prev_veresion> <version> <next_version> <rc-num>
dev/release/01-prepare.sh . 0.0.0 0.1.0 0.2.0 0
```

This will update version numbers, the changelong, and create the git tag
`apache-arrow-nanoarrow-0.1.0-rc0`. Check to make sure that the changelog
and versions are what you expect them to be before pushing the tag (you
may wish to do this by opening a dummy PR to run CI and look at the diff
from the main branch). When you are satisfied that the code at this tag
is release-candidate worthy, `git push` the tag to the `upstream` repository
(or whatever your remote name is for the `apache/arrow-nanoarrow` repo).
This will kick off a
[packaging workflow](https://github.com/apache/arrow-nanoarrow/blob/main/.github/workflows/packaging.yaml)
that will create a GitHub release and upload assets that are required for
later steps. This step can be done by any Arrow comitter.

Next, all assets need to be signed by somebody whose GPG key is listed in the
[Arrow developers KEYS file](https://dist.apache.org/repos/dist/dev/arrow/KEYS)
by calling
[02-sign.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/02-sign.sh)
The caller of the script does not need to be on any particular branch to call
the script but *does* need the
[dev/release/.env](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/.env.example)
file to exist setting the appropriate `GPG_KEY_ID` environment variable.

```
# 02-sign.sh <version> <rc-num>
dev/release/02-sign.sh . 0.1.0 0
```

Finally, run
[03-source.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/03-source.sh).
This step can be done by any Arrow comitter. The caller of this script does not need to
be on any particular branch but *does* need the
[dev/release/.env](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/.env.example)
file to exist setting the appropriate `APACHE_USERNAME` variable set.
