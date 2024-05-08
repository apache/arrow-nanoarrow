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

The `ci/scripts/build-docs.sh` script (or the `docker compose run --rm docs` compose service) can be used to run all steps at once, after which `sphinx-build source _build/html` can be used to iterate on changes.

```bash
git clone https://github.com/apache/arrow-nanoarrow.git

# Usually easiest to start with one of the docs build scripts
docker compose run --rm docs
# or install prerequisites and run
ci/scripts/build-docs.sh

# Iterate on Sphinx documentation
cd docs
sphinx-build source _build/html
```
