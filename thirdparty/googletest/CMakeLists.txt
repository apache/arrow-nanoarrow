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

# Use an old version of googletest if we have to to support gcc 4.8
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION
                                               VERSION_GREATER_EQUAL "5.0.0")
  fetchcontent_declare(googletest
                       URL https://github.com/google/googletest/archive/release-1.11.0.zip
                       URL_HASH SHA256=353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a
  )
else()
  fetchcontent_declare(googletest
                       URL https://github.com/google/googletest/archive/release-1.10.0.zip
                       URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
  )
endif()

fetchcontent_makeavailable(googletest)
