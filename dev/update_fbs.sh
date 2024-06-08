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

# get the .fbs files from the arrow repo
mkdir format && cd format
curl -L https://github.com/apache/arrow/raw/main/format/Schema.fbs --output Schema.fbs
curl -L https://github.com/apache/arrow/raw/main/format/Tensor.fbs --output Tensor.fbs
curl -L https://github.com/apache/arrow/raw/main/format/SparseTensor.fbs --output SparseTensor.fbs
curl -L https://github.com/apache/arrow/raw/main/format/Message.fbs --output Message.fbs
curl -L https://github.com/apache/arrow/raw/main/format/File.fbs --output File.fbs

# compile using flatcc
flatcc --common --reader --builder --verifier --recursive --outfile ../src/nanoarrow/nanoarrow_ipc_flatcc_generated.h *.fbs

# clean up
cd ..
rm -rf format
