#!/bin/bash
# This script is used to build the project

# Build nanoarrow statically.
cmake -S ../.. -B scratch/nanoarrow_build_static/ -DCMAKE_INSTALL_PREFIX=scratch/nanoarrow_install_static/
cmake --build scratch/nanoarrow_build_static/
cmake --install scratch/nanoarrow_build_static/

# Build nanoarrow dynamically.
cmake -S ../.. -B scratch/nanoarrow_build_shared/ -DCMAKE_INSTALL_PREFIX=scratch/nanoarrow_install_shared/ -DBUILD_SHARED_LIBS=ON
cmake --build scratch/nanoarrow_build_shared/
cmake --install scratch/nanoarrow_build_shared/

for nanoarrow_build_type in static shared; do
    # Build the project against the built nanoarrow.
    cmake -S . -B scratch/build_${nanoarrow_build_type}/ -Dnanoarrow_ROOT=scratch/nanoarrow_build_${nanoarrow_build_type}/
    cmake --build scratch/build_${nanoarrow_build_type}/

    # Build the project against the installed nanoarrow.
    cmake -S . -B scratch/build_against_install_${nanoarrow_build_type}/ -Dnanoarrow_ROOT=scratch/nanoarrow_install_${nanoarrow_build_type}/
    cmake --build scratch/build_against_install_${nanoarrow_build_type}/

    # Now try using FetchContent to get nanoarrow from remote.
    cmake -S . -B scratch/build_against_fetched_${nanoarrow_build_type}/ -DFIND_NANOARROW=OFF
    cmake --build scratch/build_against_fetched_${nanoarrow_build_type}/
done
