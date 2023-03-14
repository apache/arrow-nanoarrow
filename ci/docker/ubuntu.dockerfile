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

FROM ubuntu:latest

RUN apt-get update && apt-get install -y git cmake r-base gnupg curl

# For Arrow C++
RUN apt-get install -y -V ca-certificates lsb-release wget && \
    wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt-get install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt-get update && \
    apt-get install -y -V libarrow-dev

# For R
RUN R -e 'install.packages(c("blob", "hms", "tibble", "rlang", "testthat", "tibble", "vctrs", "withr")), repos = "https://cloud.r-project.org")'
RUN R -e 'Sys.setenv(NOT_CRAN = "true", ARROW_USE_PKG_CONFIG = "false"); install.packages("arrow", repos = "https://cloud.r-project.org")'
