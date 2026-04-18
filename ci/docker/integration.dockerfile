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
    ARCHERY_INTEGRATION_WITH_DOTNET=1 \
    ARCHERY_INTEGRATION_WITH_GO=1 \
    ARCHERY_INTEGRATION_WITH_JAVA=1 \
    ARCHERY_INTEGRATION_WITH_JS=1 \
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
RUN git clone https://github.com/apache/arrow.git /arrow-integration --recurse-submodules --depth 1

# Clone the arrow-dotnet repo
RUN git clone https://github.com/apache/arrow-dotnet.git /arrow-integration/dotnet --depth 1

# Clone the arrow-go repo
RUN git clone https://github.com/apache/arrow-go.git /arrow-integration/go --depth 1

# Clone the arrow-java repo
RUN git clone https://github.com/apache/arrow-java.git /arrow-integration/java --depth 1

# Clone the arrow-js repo
RUN git clone https://github.com/apache/arrow-js.git /arrow-integration/js --depth 1

# Clone the arrow-rs repo
RUN git clone https://github.com/apache/arrow-rs.git /arrow-integration/rust --depth 1

# Install missing conda packages that the base image expects but no longer includes.
# The conda environment sets CC, GOROOT, CFLAGS etc. but the packages that provide
# these tools were removed from the base image. Must install into the "arrow" environment.
RUN conda install -n arrow -y compilers go

# Tell zstd-sys to use system libzstd via pkg-config instead of compiling from source
ENV ZSTD_SYS_USE_PKG_CONFIG=1

# Build all the integrations except nanoarrow (since we'll do that ourselves on each run)
# Activate conda environment directly instead of using conda run, which has issues with
# environment variables being inherited from base instead of the target environment.
# Also add cargo/bin back to PATH since conda activation may not include it.
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate arrow && \
    export PATH="/root/.cargo/bin:$PATH" && \
    ARCHERY_INTEGRATION_WITH_NANOARROW="0" \
    /arrow-integration/ci/scripts/integration_arrow_build.sh /arrow-integration /build
