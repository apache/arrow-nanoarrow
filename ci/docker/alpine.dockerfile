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

FROM alpine:latest

RUN apk add bash linux-headers git cmake R R-dev g++ gnupg curl

# For Arrow C++
RUN curl https://dlcdn.apache.org/arrow/arrow-11.0.0/apache-arrow-11.0.0.tar.gz | tar -zxf - && \
    mkdir /arrow-build && \
    cd /arrow-build && \
    cmake ../apache-arrow-11.0.0/cpp \
        -DARROW_JEMALLOC=OFF \
        -DARROW_SIMD_LEVEL=NONE \
        -DCMAKE_INSTALL_PREFIX=../arrow && \
    cmake --build . && \
    make --install . --prefix=../arrow

# For R
RUN R -e 'install.packages(c("blob", "hms", "tibble", "rlang", "testthat", "tibble", "vctrs", "withr")), repos = "https://cloud.r-project.org")'
RUN R -e 'Sys.setenv(NOT_CRAN = "true", ARROW_USE_PKG_CONFIG = "false"); install.packages("arrow", repos = "https://cloud.r-project.org")'

ENV NANOARROW_CMAKE_OPTIONS -DArrow_DIR=/arrow/lib/cmake/Arrow
