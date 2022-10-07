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

# Building nanoarrow documentation

Building the nanoarrow documentation requires [Python](https://python.org), [R](https://r-project.org), [Doxygen](https://doxygen.nl), and [pandoc](https://pandoc.org/). In addition, several Python and R packages are required. You can install the Python dependencies using `pip install -r requirements.txt` in this directory; you can install the R dependencies using `R -e 'install.packages("pkgdown")`.

```bash
git clone https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow/docs

# run doxygen for the C API
pushd ../src/apidoc
doxygen
popd

# copy the readme into rst so that we can include it from sphinx
pandoc ../README.md --from markdown --to rst -s -o source/README_generated.rst

# Run sphinx to generate the main site
sphinx-build source _build/html

# Run pkgdown to generate R package documentation
R -e 'pkgdown::build_site("../r")'
```
