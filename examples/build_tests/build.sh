#!/bin/bash
# This script is used to build the project

# Build (but do not install) nanoarrow.
cmake -S ../.. -B nanoarrow_build/ -DCMAKE_INSTALL_PREFIX=nanoarrow_install/
cmake --build nanoarrow_build/

# Build the project against the built nanoarrow.
cmake -S . -B build/ -Dnanoarrow_ROOT=nanoarrow_build/
cmake --build build/

# Install nanoarrow and build against it.
cmake --install nanoarrow_build/
cmake -S . -B build_against_install/ -Dnanoarrow_ROOT=nanoarrow_install/
cmake --build build_against_install/

# Now try using FetchContent to get nanoarrow from remote.
cmake -S . -B build_against_fetched/ -DFIND_NANOARROW=OFF
cmake --build build_against_fetched/
