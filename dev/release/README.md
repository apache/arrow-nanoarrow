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
For example, to verify nanoarrow 0.1.0-rc1, one could run:

```
git clone https://github.com/apache/arrow-nanoarrow.git arrow-nanoarrow
cd arrow-nanoarrow/dev/release
./verify-release-candidate.sh 0.1.0 1
```

Full verification requires [CMake](https://cmake.org/download/) to build and run the test
suite. The test suite currently depends on an Arrow C++ installation that is discoverable
by CMake (e.g., using one of the methods described in the
[Arrow installation instructions](https://arrow.apache.org/install/)). For environments
where binary packages are not provided, building and installing Arrow C++ from source
may be required. You can provide the `NANOARROW_CMAKE_OPTIONS` environment variable to
pass extra arguments to `cmake` (e.g., `-DArrow_DIR=<path/to/arrow>/lib/cmake/Arrow` or
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
TEST_DEFAULT=0 TEST_C=1 TEST_C_BUNDLED=1 ./verify-release-candidate.sh 0.1.0 1
```

To run only R package verification (requires R but not CMake or Arrow C++):

```bash
TEST_DEFAULT=0 TEST_R=1 ./verify-release-candidate.sh 0.1.0 1
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

conda install -c conda-forge compilers git cmake gnupg arrow-cpp gtest
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
pacman -S git gcc make cmake r-base gnupg curl arrow
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
dev/release/01-prepare.sh . 0.0.0 0.1.0 0.2.0 1
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
dev/release/02-sign.sh 0.1.0 1
```

Finally, run
[03-source.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/03-source.sh).
This step can be done by any Arrow comitter. The caller of this script does not need to
be on any particular branch but *does* need the
[dev/release/.env](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/.env.example)
file to exist setting the appropriate `APACHE_USERNAME` environment variable.

At this point the release candidate is suitable for a vote on the Apache Arrow developer mailing list.

```
Hello,

I would like to propose the following release candidate (RC{rc_num}) of Apache Arrow nanoarrow [0] version {version}. This is an initial release consisting of {num_resolved_issues} resolved GitHub issues [1].

This release candidate is based on commit: {rc_commit} [2]

The source release rc{rc_num} is hosted at [3].
The changelog is located at [4].

Please download, verify checksums and signatures, run the unit tests, and vote on the release. See [5] for how to validate a release candidate.

The vote will be open for at least 72 hours.

[ ] +1 Release this as Apache Arrow nanoarrow {version}
[ ] +0
[ ] -1 Do not release this as Apache Arrow nanoarrow {version} because...

[0] https://github.com/apache/arrow-nanoarrow
[1] https://github.com/apache/arrow-nanoarrow/milestone/{milestone}?closed=1
[2] https://github.com/apache/arrow-nanoarrow/tree/apache-arrow-nanoarrow-{version}-rc{rc_num}
[3] https://dist.apache.org/repos/dist/dev/arrow/apache-arrow-nanoarrow-{version}-rc{rc_num}/
[4] https://github.com/apache/arrow-nanoarrow/blob/apache-arrow-nanoarrow-{version}-rc{rc_num}/CHANGELOG.md
[5] https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/README.md
```

## Post-release

After a passing release vote, the following tasks must be completed:

```
[ ] Closed GitHub milestone
[ ] Added release to Apache Reporter System
[ ] Uploaded artifacts to Subversion
[ ] Created GitHub release
[ ] Submit R package to CRAN
[ ] Sent announcement to announce@apache.org
[ ] Release blog post at https://github.com/apache/arrow-site/pull/288
[ ] Removed old artifacts from SVN
[ ] Bumped versions on main
```

Template for the email to announce@apache.org:

```
[ANNOUNCE] Apache Arrow nanoarrow 0.1.0 Released

The Apache Arrow community is pleased to announce the 0.1.0 release of Apache Arrow nanoarrow. This initial release covers 31 resolved issues from 6 contributors[1].

The release is available now from [2].

Release notes are available at:
https://github.com/apache/arrow-nanoarrow/blob/apache-arrow-nanoarrow-0.1.0/CHANGELOG.md

What is Apache Arrow?
---------------------
Apache Arrow is a columnar in-memory analytics layer designed to accelerate big data. It houses a set of canonical in-memory representations of flat and hierarchical data along with multiple language-bindings for structure manipulation. It also provides low-overhead streaming and batch messaging, zero-copy interprocess communication (IPC), and vectorized in-memory analytics libraries. Languages currently supported include C, C++, C#, Go, Java, JavaScript, Julia, MATLAB, Python, R, Ruby, and Rust.

What is Apache Arrow nanoarrow?
--------------------------
Apache Arrow nanoarrow is a small C library for building and interpreting Arrow C Data interface structures with bindings for users of the R programming language. The vision of nanoarrow is that it should be trivial for a library or application to implement an Arrow-based interface. The library provides helpers to create types, schemas, and metadata, an API for building arrays element-wise,
and an API to extract elements element-wise from an array. For a more detailed description of the features nanoarrow provides and motivation for its development, see [3].

Please report any feedback to the mailing lists ([4], [5]).

Regards,
The Apache Arrow Community

[1]: https://github.com/apache/arrow-nanoarrow/issues?q=is%3Aissue+milestone%3A%22nanoarrow+0.1.0%22+is%3Aclosed
[2]: https://www.apache.org/dyn/closer.cgi/arrow/apache-arrow-nanoarrow-0.1.0
[3]: https://github.com/apache/arrow-nanoarrow
[4]: https://lists.apache.org/list.html?user@arrow.apache.org
[5]: https://lists.apache.org/list.html?dev@arrow.apache.org
```
