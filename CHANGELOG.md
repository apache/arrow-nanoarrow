<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# nanoarrow Changelog

## nanoarrow 0.1.0 (2023-03-01)

### Feat

- **extensions/nanoarrow_ipc**: Improve type coverage of schema field decode (#115)
- **r**: Add `as_nanoarrow_array()` implementation that does not fall back on `arrow::as_arrow_array()` everywhere (#108)
- **r**: Create nanoarrow_array objects from buffers (#105)
- **r**: Implement infer schema methods (#104)
- **r**: Create and modify nanoarrow_schema objects (#101)

### Fix

- Correct storage type for timestamp and duration types (#116)
- **extensions/nanoarrow_ipc**: Remove extra copy of flatcc from dist/ (#113)
- make sure bundled nanoarrow is also valid C++ (#112)
- **extensions/nanoarrow_ipc**: Tweak draft interface (#111)
- set map entries/key to non-nullable (#107)
- **ci**: Actually commit bundled IPC extension to dist/ (#96)
