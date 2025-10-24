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

FROM --platform=linux/${NANOARROW_ARCH} tgagor/centos:9

RUN dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
RUN dnf install -y git gnupg gcc-c++ gcc-gfortran cmake python3-devel

# Install R
# https://docs.posit.co/resources/install-r.html
ENV R_VERSION=4.5.1
RUN curl -O https://cdn.posit.co/r/rhel-9/pkgs/R-${R_VERSION}-1-1.$(arch).rpm && \
    dnf install -y R-${R_VERSION}-1-1.$(arch).rpm
RUN ln -s /opt/R/${R_VERSION}/bin/R /usr/local/bin/R && \
    ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript

# For Arrow C++
COPY ci/scripts/build-arrow-cpp-minimal.sh /
RUN /build-arrow-cpp-minimal.sh 21.0.0 /arrow

RUN python3 -m venv /venv
RUN source /venv/bin/activate && \
    pip install --upgrade pip && \
    pip install build Cython pytest pytest-cython numpy
ENV NANOARROW_PYTHON_VENV "/venv"

# For R. Note that arrow is not installed (takes too long).
RUN mkdir ~/.R && echo "MAKEFLAGS = -j$(nproc)" > ~/.R/Makevars
RUN R -e 'install.packages("desc", repos = "https://cloud.r-project.org")' && mkdir /tmp/rdeps
COPY r/DESCRIPTION /tmp/rdeps
RUN R -e 'install.packages(setdiff(desc::desc("/tmp/rdeps")$get_deps()$package, "arrow"), repos = "https://cloud.r-project.org")'
RUN rm -f ~/.R/Makevars

ENV NANOARROW_CMAKE_OPTIONS "-DNANOARROW_BUILD_TESTS_WITH_ARROW=ON -DArrow_DIR=/arrow/lib64/cmake/Arrow"
