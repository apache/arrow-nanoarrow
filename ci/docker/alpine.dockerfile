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

ARG NANOARROW_ARCH

FROM --platform=linux/${NANOARROW_ARCH} alpine:latest

RUN apk add bash linux-headers git cmake R R-dev g++ gfortran gnupg curl py3-virtualenv python3-dev

# For Arrow C++
COPY ci/scripts/build-arrow-cpp-minimal.sh /
RUN /build-arrow-cpp-minimal.sh 15.0.2 /arrow

# There's a missing define that numpy's build needs on s390x and there is no wheel
RUN (grep -e "S390" /usr/include/bits/hwcap.h && echo "#define HWCAP_S390_VX HWCAP_S390_VXRS" >> /usr/include/bits/hwcap.h) || true
RUN virtualenv -v --download /venv
RUN source /venv/bin/activate && pip install build Cython pytest pytest-cython numpy
ENV NANOARROW_PYTHON_VENV "/venv"

# For R. Note that arrow is not installed (takes too long).
RUN mkdir ~/.R && echo "MAKEFLAGS = -j$(nproc)" > ~/.R/Makevars
RUN R -e 'install.packages("desc", repos = "https://cloud.r-project.org")' && mkdir /tmp/rdeps
COPY r/DESCRIPTION /tmp/rdeps
RUN R -e 'install.packages(setdiff(desc::desc("/tmp/rdeps")$get_deps()$package, "arrow"), repos = "https://cloud.r-project.org")'
RUN rm -f ~/.R/Makevars

ENV NANOARROW_CMAKE_OPTIONS -DArrow_DIR=/arrow/lib/cmake/Arrow
