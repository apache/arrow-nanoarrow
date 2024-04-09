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

FROM --platform=linux/${NANOARROW_ARCH} ubuntu:latest

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    locales git cmake r-base gnupg curl valgrind gfortran python3-venv python3-dev doxygen pandoc lcov \
    libxml2-dev libfontconfig1-dev libfreetype6-dev libfribidi-dev libharfbuzz-dev \
    libjpeg-dev libpng-dev libtiff-dev

# For Arrow C++
RUN apt-get install -y -V ca-certificates lsb-release wget && \
    wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt-get install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt-get update && \
    apt-get install -y -V libarrow-dev

# For documentation build + Python build
RUN python3 -m venv --upgrade-deps /venv
RUN echo "source /venv/bin/activate && pip install pydata-sphinx-theme sphinx breathe build Cython numpy pytest pytest-cython pytest-cov pyarrow" | bash
ENV NANOARROW_PYTHON_VENV "/venv"

# Locale required for R CMD check
RUN locale-gen en_US.UTF-8 && update-locale en_US.UTF-8

# For R. The Ubuntu image installs a few extras for coverage + documentation
RUN mkdir ~/.R && echo "MAKEFLAGS = -j$(nproc)" > ~/.R/Makevars
RUN R -e 'install.packages(c("desc", "covr", "pkgdown"), repos = "https://cloud.r-project.org")' && mkdir /tmp/rdeps
COPY r/DESCRIPTION /tmp/rdeps
RUN R -e 'install.packages(setdiff(desc::desc("/tmp/rdeps")$get_deps()$package, "arrow"), repos = "https://cloud.r-project.org")'

# Install arrow here so that the integration tests for R run in at least one test image.
# -fPIC required for this to work on MacOS/arm64
RUN echo "CXX17FLAGS += -fPIC" >> ~/.R/Makevars
RUN ARROW_USE_PKG_CONFIG=false ARROW_R_DEV=true R -e 'install.packages("arrow", repos = "https://cloud.r-project.org"); library(arrow)'
RUN rm -f ~/.R/Makevars
