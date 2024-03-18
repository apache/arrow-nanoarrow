#!/bin/bash
# This script is used to build the project

# Build (but do not install) nanoarrow.
cmake -S ../.. -B nanoarrow_build/
cmake --build nanoarrow_build/

# Build the project against the built nanoarrow.
cmake -S . -B build/ -Dnanoarrow_ROOT=nanoarrow_build/
cmake --build build/
