# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Download a flatcc release
curl -L https://github.com/dvidelabs/flatcc/archive/refs/tags/v0.6.1.zip --output flatcc-src.zip
unzip flatcc-src.zip -d flatcc-src
FLATCC_SRC_DIR=flatcc-src/`ls flatcc-src`

# Remove previous vendored flatcc
rm -rf ../thirdparty/flatcc

# Create the folders we need to exist
mkdir -p ../thirdparty/flatcc/src/runtime
mkdir -p ../thirdparty/flatcc/include/flatcc/portable

# The only part of the runtime we need
cp $FLATCC_SRC_DIR/src/runtime/emitter.c \
  $FLATCC_SRC_DIR/src/runtime/builder.c \
  $FLATCC_SRC_DIR/src/runtime/verifier.c \
  $FLATCC_SRC_DIR/src/runtime/refmap.c \
  ../thirdparty/flatcc/src/runtime/

# We also need the headers for those sources. makedepend
# can get those for us in topological order.
pushd $FLATCC_SRC_DIR/include

# List object dependencies (warns that it can't find system headers, which is OK)
# This list is in topological order and could be used for a single-file include;
# however, this approach is easier to support alongside a previous installation
# of the flatcc runtime.
makedepend -s#: -f- -- -I. -DFLATCC_PORTABLE -- 2>/dev/null \
  ../src/runtime/emitter.c \
  ../src/runtime/builder.c \
  ../src/runtime/verifier.c \
  ../src/runtime/refmap.c \
  ../src/nanoarrow/nanoarrow_ipc.c | \
  # Remove the '<src file>.o: ' prefix
  sed 's/[^:]*: *//' | \
  # Spaces to new lines
  sed 's/ /\n/' | \
  # Only unique lines (but don't sort to preserve topological order)
  awk '!x[$0]++' | \
  # Remove blank lines
  sed '/^$/d' | \
  # Or anything having to do with nanoarrow
  sed '/nanoarrow/d' > ../../../flatcc-headers.txt

popd

# Copy the headers we need. Using loops because cp --parents is not portable.
for file in $(cat flatcc-headers.txt | sed /portable/d); do
    cp "$FLATCC_SRC_DIR/include/$file" ../thirdparty/flatcc/include/flatcc
done
for file in $(cat flatcc-headers.txt | sed -n /portable/p); do
    cp "$FLATCC_SRC_DIR/include/$file" ../thirdparty/flatcc/include/flatcc/portable;
done

# And the license
cp $FLATCC_SRC_DIR/LICENSE ../thirdparty/flatcc

# clean up
rm -rf flatcc-src
rm flatcc-src.zip
rm flatcc-headers.txt
