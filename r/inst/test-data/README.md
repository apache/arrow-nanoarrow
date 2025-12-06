<!--
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

# Extra test files for R

## complex-map.arrows

Sourced from the [Overture Maps Foundation](https://overturemaps.org) "divisions_area" table using
[SedonaDB](https://sedona.apache.org/sedonadb).

```r
library(sedonadb)

sd_sql("SET datafusion.execution.batch_size = 1024")

sd_read_parquet("/Volumes/data/overture/data/theme=divisions/type=division_area/") |> 
  sd_to_view("division_area", overwrite = TRUE)

sd_sql("SELECT ROW_NUMBER() OVER (ORDER BY names.primary) as idx, names FROM division_area WHERE names.common IS NOT NULL") |> 
  sd_compute() |> sd_to_view("names_with_common", overwrite = TRUE)

sd_sql("SELECT * FROM names_with_common WHERE (idx % 100) = 0 ORDER BY idx") |> 
  nanoarrow::write_nanoarrow("inst/test-data/complex-map.arrows")
```
