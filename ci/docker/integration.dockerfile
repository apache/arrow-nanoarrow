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

FROM apache/arrow-dev:amd64-conda-integration

ENV ARROW_USE_CCACHE=OFF \
    ARROW_CPP_EXE_PATH=/build/cpp/debug \
    ARROW_NANOARROW_PATH=/build/nanoarrow \
    ARROW_RUST_EXE_PATH=/build/rust/debug \
    BUILD_DOCS_CPP=OFF \
    ARROW_INTEGRATION_CPP=ON \
    ARROW_INTEGRATION_CSHARP=ON \
    ARROW_INTEGRATION_GO=ON \
    ARROW_INTEGRATION_JAVA=ON \
    ARROW_INTEGRATION_JS=ON \
    ARCHERY_INTEGRATION_WITH_NANOARROW="1" \
    ARCHERY_INTEGRATION_WITH_RUST="1"

# These are necessary because the github runner overrides $HOME
# https://github.com/actions/runner/issues/863
ENV RUSTUP_HOME=/root/.rustup
ENV CARGO_HOME=/root/.cargo

ENV ARROW_USE_CCACHE=OFF
ENV ARROW_CPP_EXE_PATH=/build/cpp/debug
ENV ARROW_NANOARROW_PATH=/build/nanoarrow
ENV ARROW_RUST_EXE_PATH=/build/rust/debug
ENV BUILD_DOCS_CPP=OFF

# Clone the arrow monorepo
RUN git clone https://github.com/apache/arrow.git /arrow-integration --recurse-submodules

# Clone the arrow-rs repo
RUN git clone https://github.com/apache/arrow-rs /arrow-integration/rust

# Build all the integrations except nanoarrow (since we'll do that ourselves on each run)
RUN ARCHERY_INTEGRATION_WITH_NANOARROW="0" \
    conda run --no-capture-output \
    /arrow-integration/ci/scripts/integration_arrow_build.sh \
    /arrow-integration \
    /build
