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

FROM --platform=linux/${NANOARROW_ARCH} centos:7

RUN yum install -y epel-release
RUN yum install -y git gnupg curl R gcc-c++ gcc-gfortran cmake3 python3-devel

# For Arrow C++. Use 9.0.0 because this version works fine with the default gcc
RUN curl -L https://github.com/apache/arrow/archive/refs/tags/apache-arrow-9.0.0.tar.gz | tar -zxf - && \
    mkdir /arrow-build && \
    cd /arrow-build && \
    cmake3 ../arrow-apache-arrow-9.0.0/cpp \
        -DARROW_JEMALLOC=OFF \
        -DARROW_SIMD_LEVEL=NONE \
        -DARROW_WITH_ZLIB=ON \
        -DCMAKE_INSTALL_PREFIX=../arrow && \
    cmake3 --build . && \
    make install

RUN python3 -m venv /venv
RUN source /venv/bin/activate && \
    pip install --upgrade pip && \
    pip install build Cython pytest pytest-cython numpy
ENV NANOARROW_PYTHON_VENV "/venv"

# Locale required for R CMD check
RUN localedef -c -f UTF-8 -i en_US en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# For R. Note that arrow is not installed (takes too long).
RUN mkdir ~/.R && echo "MAKEFLAGS = -j$(nproc)" > ~/.R/Makevars
RUN R -e 'install.packages("desc", repos = "https://cloud.r-project.org")' && mkdir /tmp/rdeps
COPY r/DESCRIPTION /tmp/rdeps
RUN R -e 'install.packages(setdiff(desc::desc("/tmp/rdeps")$get_deps()$package, "arrow"), repos = "https://cloud.r-project.org")'
RUN rm -f ~/.R/Makevars

ENV NANOARROW_CMAKE_OPTIONS -DArrow_DIR=/arrow/lib/cmake/Arrow
ENV CMAKE_BIN cmake3
ENV CTEST_BIN ctest3
