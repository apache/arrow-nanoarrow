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

# If we are building within the repo, the safest way to proceed is to
# run CMake to regenerate the bundled nanoarrow.c/nanoarrow.h.
temp_dir <- tempfile()
on.exit(unlink(temp_dir, recursive = TRUE))
dir.create(temp_dir)

source_dir <- normalizePath("..", winslash = "/")
build_dir <- file.path(temp_dir, "build")
dist_dir <- file.path(temp_dir, "dist")
dir.create(build_dir)
dir.create(dist_dir)

cmake_command <- shQuote(Sys.getenv("CMAKE_BIN", "cmake"))

run_cmake <- function(args, wd = ".") {
  force(args)

  previous_wd <- getwd()
  setwd(dir = wd)
  on.exit(setwd(dir = previous_wd))

  command <- sprintf("%s %s", cmake_command, paste(args, collapse = " "))
  exit_code <- system(command)
  message(sprintf("[%d] %s", exit_code, command))
  exit_code == 0
}

file.exists("../CMakeLists.txt") &&
  run_cmake("--version") &&
  run_cmake(
    sprintf("%s -DNANOARROW_BUNDLE=ON -DNANOARROW_NAMESPACE=RPkg", source_dir),
    wd = build_dir
  ) &&
  run_cmake(sprintf("--build %s", shQuote(build_dir))) &&
  run_cmake(
    sprintf("--install %s --prefix=%s", shQuote(build_dir), shQuote(dist_dir))
  )

# If any of the above failed, we can also copy from ../dist. This is likely for
# for installs via pak or remotes that run pkgbuild::build()
if (!file.exists(file.path(dist_dir, "nanoarrow.h"))) {
  dist_dir <- "../dist"
}

files_to_vendor <- file.path(dist_dir, c("nanoarrow.c", "nanoarrow.h"))

if (all(file.exists(files_to_vendor))) {
  files_dst <- file.path("src", basename(files_to_vendor))

  n_removed <- suppressWarnings(sum(file.remove(files_dst)))
  if (n_removed > 0) {
    cat(sprintf("Removed %d previously vendored files from src/\n", n_removed))
  }

  cat(
    sprintf(
      "Vendoring files from arrow-nanoarrow to src/:\n%s\n",
      paste("-", files_to_vendor, collapse = "\n")
    )
  )

  if (all(file.copy(files_to_vendor, "src"))) {
    cat("All files successfully copied to src/\n")
  } else {
    stop("Failed to vendor all files")
  }
}
