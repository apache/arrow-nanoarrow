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
For example, to verify nanoarrow 0.7.0-rc0, one could run:

```bash
git clone https://github.com/apache/arrow-nanoarrow.git arrow-nanoarrow
cd arrow-nanoarrow/dev/release
./verify-release-candidate.sh 0.7.0 0
```

The verification script itself is written in `bash` and requires the `curl`, `gpg`, and
`shasum`/`sha512sum` commands. These are typically available from a package
manager except on Windows (see below). [CMake](https://cmake.org/download/),
Python (>=3.8), and  a C/C++ compiler are required to verify the C libraries;
Python (>=3.8) is required to verify the Python bindings; and R (>= 4.0) is
required to verify the R bindings. See below for platform-specific direction
for how to obtain verification dependencies.

To run only C library verification (requires CMake and Arrow C++ but not R or Python):
Options are passed to the verification script using environment variables.
For example, to run only C library verification (requires CMake and Python but not R):

To run only C library verification (requires CMake but not R or Python):

```bash
TEST_DEFAULT=0 TEST_C=1 TEST_C_BUNDLED=1 ./verify-release-candidate.sh 0.7.0 0
```

To run only R package verification (requires R but not CMake or Arrow C++):

```bash
TEST_DEFAULT=0 TEST_R=1 ./verify-release-candidate.sh 0.7.0 0
```

To run only Python verification (requires Python but not CMake or Arrow C++):

```bash
TEST_DEFAULT=0 TEST_PYTHON=1 ./verify-release-candidate.sh 0.7.0 0
```

### MacOS

On MacOS you can install a modern C/C++ toolchain via the XCode Command Line Tools (i.e.,
`xcode-select --install`). Other dependencies are available via [Homebrew](https://brew.sh):

```bash
brew install cmake gnupg
```

You can install R using the instructions provided on the
[R Project Download page](https://cloud.r-project.org/bin/macosx/);
the system `python3` provided by MacOS is sufficient to verify the release
candidate.

For older MacOS or MacOS without Homebrew, you can
[install GnuPG](https://gnupg.org/download/) and
[install CMake](https://cmake.org/download/) separately.

### Conda (Linux and MacOS)

Using `conda`, one can install all requirements needed for verification on Linux
or MacOS. Users are recommended to install `gnupg` using
a system installer because of interactions with other installations that
may cause a crash.

```bash
conda create --name nanoarrow-verify-rc
conda activate nanoarrow-verify-rc
conda config --set channel_priority strict

conda install -c conda-forge compilers git cmake
# For R (see below about potential interactions with system R
# before installing via conda on MacOS)
conda install -c conda-forge r-testthat r-hms r-blob r-pkgbuild r-bit64
```

Note that using conda-provided R when there is also a system install of R
on MacOS is unlikely to work.

### Windows

On Windows, prerequisites can be installed using officially provided installers:
[Visual Studio](https://visualstudio.microsoft.com/vs/),
[CMake](https://cmake.org/download/), and
[Git](https://git-scm.com/downloads) should provide the prerequisties
to verify the C library; R and Rtools can be installed using the
[official R-project installer](https://cloud.r-project.org/bin/windows/).

```bash
# Pass location of R to the verification script
export NANOARROW_CMAKE_OPTIONS="-Dgtest_force_shared_crt=ON -DNANOARROW_ARROW_STATIC=ON"
export R_HOME="/c/Program Files/R/R-4.5.0"
```

### Debian/Ubuntu

On Debian/Ubuntu (e.g., `docker run --rm -it ubuntu:latest`) you can install prerequisites using `apt`.

```bash
apt-get update && apt-get install -y git g++ cmake r-base gnupg curl python3-dev python3-venv
```

If you have never installed an R package before, R verification will fail when it
tries to install any missing dependencies. Because of how R is configured by
default, you must install your first package in an interactive session and select
`yes` when it asks if you would like to create a user-specific directory.

### Fedora

On recent Fedora (e.g., `docker run --rm -it fedora:latest`), you can install all prerequisites
using `dnf`:

```bash
dnf install -y git cmake R gnupg curl python3-devel python3-virtualenv
```

### Arch Linux

On Arch Linux (e.g., `docker run --rm -it archlinux:latest`, you can install all prerequisites
using `pacman`):

```bash
pacman -Sy git gcc make cmake r-base gnupg curl python
```

### Alpine Linux

On Alpine Linux (e.g., `docker run --rm -it alpine:latest`), all prerequisites are available using `apk add` except for Arrow C++ which requires enabling the
community repository.

```bash

apk add bash linux-headers git cmake R R-dev g++ gnupg curl python3-dev
```

### Big endian

One can verify a nanoarrow release candidate on big endian by setting
`DOCKER_DEFAULT_PLATFORM=linux/s390x` and following the instructions for
[Alpine Linux](#alpine-linux), [Fedora](#fedora), or [Debian/Ubuntu](#debianubuntu).

## Creating a release candidate

The first step to creating a nanoarrow release is to create a `maint-VERSION` branch
(e.g., `usethis::pr_init("maint-0.7.0")`) and push the branch to `upstream`. This is
a good opportunity to run though the above instructions to make sure the verification
script and instructions are up-to-date.
targeting the maint-XX branch that was just pushed.

This is a good time to run other final checks such as:

- Run through some R packaging release checks (e.g., urlchecker, winbuilder)
- Manually dispatch the [Verification workflow](https://github.com/apache/arrow-nanoarrow/actions/workflows/verify.yaml).
- Manually dispatch the [Python wheels workflow](https://github.com/apache/arrow-nanoarrow/actions/workflows/python-wheels.yaml).
- Create a draft [PR into WrapDB](#update-the-wrapdb-entry) to make sure tests pass in their CI
- Create a draft [PR into vcpkg](#update-the-vcpkg-entry) to make sure tests pass in their CI
- Draft a release blog post and make a draft PR into [arrow-site](https://github.com/apache/arrow-site).
- Review [nanoarrow dev documentation](https://arrow.apache.org/nanoarrow/main/) for obvious holes/typos.

When these steps are complete, run
[01-prepare.R](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/01-prepare.sh):

```bash
# from the repository root
# 01-prepare.sh <nanoarrow-dir> <prev_veresion> <version> <next_version> <rc-num>
dev/release/01-prepare.sh . 0.6.0 0.7.0 0.8.0 0
```

This will update version numbers, the changelong, and create the git tag
`apache-arrow-nanoarrow-0.7.0-rc0`. Check to make sure that the changelog
and versions are what you expect them to be before pushing the tag (you
may wish to do this by opening a dummy PR to run CI and look at the diff
from the main branch).

When you are satisfied that the code at this tag
is release-candidate worthy, `git push` the tag to the `upstream` repository
(or whatever your remote name is for the `apache/arrow-nanoarrow` repo).
This will kick off a
[packaging workflow](https://github.com/apache/arrow-nanoarrow/blob/main/.github/workflows/packaging.yaml)
that will create a GitHub release and upload assets that are required for
later steps. This step can be done by any Arrow committer.

Next, all assets need to be signed by somebody whose GPG key is listed in the
[Arrow developers KEYS file](https://dist.apache.org/repos/dist/dev/arrow/KEYS)
by calling
[02-sign.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/02-sign.sh)
The caller of the script does not need to be on any particular branch to call
the script but *does* need the
[dev/release/.env](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/.env.example)
file to exist setting the appropriate `GPG_KEY_ID` environment variable.

```bash
# 02-sign.sh <version> <rc-num>
dev/release/02-sign.sh 0.7.0 0
```

Finally, run
[03-source.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/03-source.sh).
This step can be done by any Arrow committer. The caller of this script does not need to
be on any particular branch but *does* need the
[dev/release/.env](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/.env.example)
file to exist setting the appropriate `APACHE_USERNAME` environment variable.

```
# 03-source.sh $0 <version> <rc-num>
dev/release/03-source.sh 0.7.0 0
```

You should check that the release verification runs locally and/or
start a
[Verification workflow](https://github.com/apache/arrow-nanoarrow/actions/workflows/verify.yaml) and wait for it to complete.

At this point the release candidate is suitable for a vote on the Apache Arrow developer mailing list.

```
[VOTE] Release nanoarrow 0.7.0

Hello,

I would like to propose the following release candidate (rc0) of Apache Arrow nanoarrow [0] version 0.7.0. This is an initial release consisting of 44 resolved GitHub issues from 5 contributors [1].

This release candidate is based on commit: {rc_commit} [2]

The source release rc0 is hosted at [3].
The changelog is located at [4].

Please download, verify checksums and signatures, run the unit tests, and vote on the release. See [5] for how to validate a release candidate.

The vote will be open for at least 72 hours.

[ ] +1 Release this as Apache Arrow nanoarrow 0.7.0
[ ] +0
[ ] -1 Do not release this as Apache Arrow nanoarrow 0.7.0 because...

[0] https://github.com/apache/arrow-nanoarrow
[1] https://github.com/apache/arrow-nanoarrow/milestone/4?closed=1
[2] https://github.com/apache/arrow-nanoarrow/tree/apache-arrow-nanoarrow-0.7.0-rc0
[3] https://dist.apache.org/repos/dist/dev/arrow/apache-arrow-nanoarrow-0.7.0-rc0/
[4] https://github.com/apache/arrow-nanoarrow/blob/apache-arrow-nanoarrow-0.7.0-rc0/CHANGELOG.md
[5] https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/README.md
```

## Post-release

After a passing release vote, the following tasks must be completed:

```
[ ] Closed GitHub milestone
[ ] Added release to the Apache Reporter System
[ ] Uploaded artifacts to Subversion
[ ] Created GitHub release
[ ] Submit R package to CRAN
[ ] Submit Python package to PyPI
[ ] Update Python package on conda-forge
[ ] Update the WrapDB entry
[ ] Update release documentation
[ ] Release blog post at https://github.com/apache/arrow-site/pull/288
[ ] Sent announcement to announce@apache.org
[ ] Removed old artifacts from SVN
[ ] Bumped versions on main
```

### Close GitHub milestone

Find the appropriate entry in https://github.com/apache/arrow-nanoarrow/milestones/
and mark it as closed.

### Add release to the Apache Reporter System

The reporter system for Arrow can be found at
<https://reporter.apache.org/addrelease.html?arrow>. To add a release, a
PMC member must log in with their Apache username/password. The release
names are in the form `NANOARROW-0.7.0`.

### Upload artifacts to Subversion / Create GitHub Release

These are both handled by
[post-01-upload.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/post-01-upload.sh).
This script must be run by a PMC member whose `APACHE_USERNAME` environment variable
has been set in `.env`.

```bash
dev/release/post-01-upload.sh 0.7.0 0
```

### Submit R package to CRAN

The R package submission occurs from a separate branch to facilitate including
any small changes requested by a member of the CRAN team; however, these
updates are usually automatic and do not require additional changes.
Before a release candidate is created, the first section of
`usethis::use_release_issue()` should all be completed (i.e., any changes
after release should be minor tweaks). The steps are:

- Ensure you are on the release branch (i.e., `git switch maint-0.7.0`)
- Run `usethis::pr_init("r-cran-maint-0.7.0")` and push the branch to your
  fork.
- Ensure `cran_comments.md` is up-to-date.
- Run `devtools::check()` locally and verify that the package version is correct
- Run `urlchecker::url_check()`
- Run `devtools::check_win_devel()` and wait for the response
- Run `devtools::submit_cran()`
- Confirm submission email

Any changes required at this stage should be made as a PR into `main` and
cherry-picked into the `r-cran-maint-XXX` packaging branch. (i.e.,
`git cherry-pick 01234abcdef`). If any changes
to the source are required, bump the "tweak" version (e.g., `Version: 0.7.0.1`
in `DESCRIPTION`).

### Submit Python package to PyPI

The Python package source distribution and wheels are built using the [Build Python Wheels](https://github.com/apache/arrow-nanoarrow/actions/workflows/python-wheels.yaml) action on the `maint-0.7.0` branch after cutting the release candidate.

To submit these to PyPI, download all assets from the run into a folder (e.g., `python/dist`) and run:

```shell
# pip install twine
twine upload python/dist/*.tar.gz python/dist/*.whl
```

You will need to enter a token with "upload packages" permissions for the [nanoarrow PyPI project](https://pypi.org/project/nanoarrow/).

This can/should be automated for future releases using one or more GitHub API calls.

### Update Python package on conda-forge

The [conda-forge feedstock](https://github.com/conda-forge/nanoarrow-feedstock) is updated automatically by a conda-forge bot after the source distribution has been uploaded to PyPI (typically this takes several hours). This will also start a CI run to ensure that the updated version will build on PyPI.

### Update the WrapDB Entry

The nanoarrow C library is available for users of the [Meson build system](https://mesonbuild.com/) via [WrapDB](https://mesonbuild.com/Wrapdb-projects.html). When a new release is added, PR into the [WrapDB repository](https://github.com/mesonbuild/wrapdb) is required to make the new version available to users. See https://github.com/mesonbuild/wrapdb/pull/1536 for a template PR. It is also a good idea to do this step before the release candidate is cut to catch packaging issues before finalizing the content of the version.

### Update the vcpkg Entry

The nanoarrow C library is available on [vcpkg](https://github.com/microsoft/vcpkg). When a new release is added, PR into the vcpkg repository to make the new version available to users. See https://github.com/microsoft/vcpkg/pull/46029 for a template PR. It is a good idea to do this step before merging a release to catch packaging issues before finalizing the content of the version.

### Update release documentation

The [nanoarrow documentation](https://arrow.apache.org/nanoarrow) is populated from the [asf-site branch](https://github.com/apache/arrow-nanoarrow/tree/asf-site) of this repository. To update the documentation, first clone just the asf-site branch:

```shell
git clone -b asf-site --single-branch https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow
```

Download the [0.7.0 documentation](https://github.com/apache/arrow-nanoarrow/releases/download/apache-arrow-nanoarrow-0.7.0/docs.tgz):

```shell
curl -L https://github.com/apache/arrow-nanoarrow/releases/download/apache-arrow-nanoarrow-0.7.0/docs.tgz \
  -o docs.tgz
```

Extract the documentation and rename the directory to `0.7.0`:

```shell
tar -xvzf docs.tgz
mv nanoarrow-docs 0.7.0
```

Then remove the existing `latest` directory and run the extraction again, renaming to `latest` instead:

```shell
rm -rf latest
tar -xvzf docs.tgz
mv nanoarrow-docs latest
```

Finally, update `switcher.json` with entries pointing `/latest/` and `/0.7.0/` to `"version": "0.7.0"`:

```json
[
    {
        "version": "dev",
        "url": "https://arrow.apache.org/nanoarrow/main/"
    },
    {
        "version": "0.7.0",
        "url": "https://arrow.apache.org/nanoarrow/latest/"
    },
    {
        "version": "0.7.0",
        "url": "https://arrow.apache.org/nanoarrow/0.7.0/"
    },
    {
        "version": "0.6.0",
        "url": "https://arrow.apache.org/nanoarrow/0.6.0/"
    },
    ...
]
```

This can/should be automated for future releases.

### Release blog post

Final review + merge of the blog post that was drafted prior to preparation of
the release candidate.

### Send announcement

This email should be sent to `announce@apache.org` and `dev@arrow.apache.org`. It
**must** be sent from your Apache email address and **must** be sent through
the `mail-relay.apache.org` outgoing server.

Email template:

```
[ANNOUNCE] Apache Arrow nanoarrow 0.7.0 Released

The Apache Arrow community is pleased to announce the 0.7.0 release of
Apache Arrow nanoarrow. This initial release covers 79 resolved issues
from 9 contributors[1].

The release is available now from [2], release notes are available at
[3], and a blog post highlighting new features and breaking changes is
available at [4].

What is Apache Arrow?
---------------------
Apache Arrow is a columnar in-memory analytics layer designed to
accelerate big data. It houses a set of canonical in-memory
representations of flat and hierarchical data along with multiple
language-bindings for structure manipulation. It also provides
low-overhead streaming and batch messaging, zero-copy interprocess
communication (IPC), and vectorized in-memory analytics libraries.
Languages currently supported include C, C++, C#, Go, Java,
JavaScript, Julia, MATLAB, Python, R, Ruby, and Rust.

What is Apache Arrow nanoarrow?
--------------------------
Apache Arrow nanoarrow is a C library for building and interpreting
Arrow C Data interface structures with bindings for users of R and
Python. The vision of nanoarrow is that it should be trivial for a
library or application to implement an Arrow-based interface. The
library provides helpers to create types, schemas, and metadata, an
API for building arrays element-wise,
and an API to extract elements element-wise from an array. For a more
detailed description of the features nanoarrow provides and motivation
for its development, see [5].

Please report any feedback to the mailing lists ([6], [7]).

Regards,
The Apache Arrow Community

[1] https://github.com/apache/arrow-nanoarrow/issues?q=milestone%3A%22nanoarrow+0.7.0%22+is%3Aclosed
[2] https://www.apache.org/dyn/closer.cgi/arrow/apache-arrow-nanoarrow-0.7.0
[3] https://github.com/apache/arrow-nanoarrow/blob/apache-arrow-nanoarrow-0.7.0/CHANGELOG.md
[4] https://arrow.apache.org/blog/2024/05/27/nanoarrow-0.7.0-release/
[5] https://arrow.apache.org/nanoarrow/
[6] https://lists.apache.org/list.html?user@arrow.apache.org
[7] https://lists.apache.org/list.html?dev@arrow.apache.org
```

### Remove old artifacts from SVN

These artifacts include any release candidates that were uploaded to
<https://dist.apache.org/repos/dist/dev/arrow/> and old releases that
were upload to
<https://dist.apache.org/repos/dist/release/arrow/>. You can remove
them using:

```
# Once
export APACHE_USERNAME=xxx
# Once for every release candidate
svn rm --username=$APACHE_USERNAME -m "Clean up svn artifacts" https://dist.apache.org/repos/dist/dev/arrow/apache-arrow-nanoarrow-0.7.0-rc0/
```

### Bumped versions on main

This is handled by
[post-03-bump-versions.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/post-03-bump-versions.sh). Create a branch and then run:

```bash
dev/release/post-03-bump-versions.sh . 0.7.0 0.8.0
```
