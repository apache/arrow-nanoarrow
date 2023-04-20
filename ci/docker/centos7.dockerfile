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
RUN yum install -y git gnupg curl R gcc-c++ cmake3

RUN localedef -c -f UTF-8 -i en_US en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# For Arrow C++. Use 9.0.0 because this version works fine with the default gcc
RUN curl -L https://github.com/apache/arrow/archive/refs/tags/apache-arrow-9.0.0.tar.gz | tar -zxf - && \
    mkdir /arrow-build && \
    cd /arrow-build && \
    cmake3 ../arrow-apache-arrow-9.0.0/cpp \
        -DARROW_JEMALLOC=OFF \
        -DARROW_SIMD_LEVEL=NONE \
        -DCMAKE_INSTALL_PREFIX=../arrow && \
    cmake3 --build . && \
    make install

# For R. Note that arrow is not installed (takes too long).
RUN R -e 'install.packages(c("blob", "hms", "tibble", "rlang", "testthat", "tibble", "vctrs", "withr"), repos = "https://cloud.r-project.org")'

ENV NANOARROW_CMAKE_OPTIONS -DArrow_DIR=/arrow/lib/cmake/Arrow
ENV CMAKE_BIN cmake3
ENV CTEST_BIN ctest3
