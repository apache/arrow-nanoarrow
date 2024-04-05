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

FROM --platform=linux/${NANOARROW_ARCH} fedora:latest

RUN dnf install -y git cmake R gnupg curl libarrow-devel glibc-langpack-en \
   python3-devel python3-virtualenv

# For Python
RUN python3 -m venv --upgrade-deps /venv
RUN source /venv/bin/activate && pip install build Cython pytest pytest-cython numpy pyarrow
ENV NANOARROW_PYTHON_VENV "/venv"

# For R. Note that arrow is not installed (takes too long).
RUN mkdir ~/.R && echo "MAKEFLAGS = -j$(nproc)" > ~/.R/Makevars
RUN R -e 'install.packages("desc", repos = "https://cloud.r-project.org")' && mkdir /tmp/rdeps
COPY r/DESCRIPTION /tmp/rdeps
RUN R -e 'install.packages(setdiff(desc::desc("/tmp/rdeps")$get_deps()$package, "arrow"), repos = "https://cloud.r-project.org")'
RUN rm -f ~/.R/Makevars
