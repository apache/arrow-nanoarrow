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

REPO=`git rev-parse --show-toplevel`
if [ -z "$REPO" ]
then exit 1
fi

# Download flatcc
git clone https://github.com/dvidelabs/flatcc.git
git -C flatcc checkout bf4f67a16d85541e474f1d67b8fb64913ba72bc7

# Remove previous vendored flatcc
rm -rf $REPO/thirdparty/flatcc

# Create the folders we need to exist
mkdir -p $REPO/thirdparty/flatcc/src/runtime
mkdir -p $REPO/thirdparty/flatcc/include/flatcc/portable

# The only part of the runtime we need
cp flatcc/src/runtime/emitter.c \
  flatcc/src/runtime/builder.c \
  flatcc/src/runtime/verifier.c \
  flatcc/src/runtime/refmap.c \
  $REPO/thirdparty/flatcc/src/runtime/

# We also need the headers for those sources. makedepend
# can get those for us in topological order.

# List object dependencies (warns that it can't find system headers, which is OK)
# This list is in topological order and could be used for a single-file include;
# however, this approach is easier to support alongside a previous installation
# of the flatcc runtime.
makedepend -s#: -f- -- -Iflatcc/include -I$REPO/src -DFLATCC_PORTABLE -- 2>/dev/null \
  `ls $REPO/thirdparty/flatcc/src/runtime/*.c` `ls $REPO/src/nanoarrow/ipc/*.c` | \
  # Remove the '<src file>.o: ' prefix
  sed 's/[^:]*: *//' | \
  # Spaces to new lines
  sed 's/ /\n/' | \
  # Remove system headers
  sed '/^\//d' | \
  # Only unique lines (but don't sort to preserve topological order)
  awk '!x[$0]++' | \
  # Remove anything having to do with nanoarrow
  sed '/nanoarrow/d' | \
  # Remove blank lines
  sed '/^$/d' > flatcc-headers.txt

cat flatcc-headers.txt

# Copy the headers we need. Using loops because cp --parents is not portable.
for file in $(cat flatcc-headers.txt); do
    cp "$file" "$REPO/thirdparty/$file"
done

# And the license
cp flatcc/LICENSE $REPO/thirdparty/flatcc/LICENSE

# clean up
rm -rf flatcc
rm flatcc-headers.txt
