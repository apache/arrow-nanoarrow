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

# Use ci/scripts/bundle.py to generate source files/headers that are
# easier to use with the default R package build system

python <- function() {
  from_env <- Sys.getenv("PYTHON_BIN", "python3")
  has_from_env <- system(
    paste(from_env, "--version"),
    ignore.stdout = TRUE,
    ignore.stderr = TRUE
  )

  if (has_from_env == 0) {
    from_env
  } else {
    "python"
  }
}

run_bundler <- function() {
  args <- c(
    "--symbol-namespace=RPkg",
    "--header-namespace=",
    "--include-output-dir=src",
    "--source-output-dir=src",
    "--with-ipc",
    "--with-flatcc"
  )
  command <- sprintf(
    "%s ../ci/scripts/bundle.py  %s",
    python(),
    paste(args, collapse = " ")
  )

  exit_code <- system(command)
  message(sprintf("[%d] %s", exit_code, command))
  exit_code == 0
}

cat("Vendoring files from arrow-nanoarrow to src/:\n")
stopifnot(file.exists("../CMakeLists.txt") && run_bundler())

# Post-process headers for CMD check
f <- "src/flatcc/portable/pdiagnostic.h"
lines <- readLines(f)
writeLines(gsub("^#pragma", "/**/#pragma", lines), f)

# Remove unused files
unused_files <- list.files("src", "\\.hpp$", full.names = TRUE)
unlink(unused_files)
