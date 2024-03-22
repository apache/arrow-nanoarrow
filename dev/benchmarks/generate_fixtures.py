# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os

import numpy as np
import pyarrow as pa
from pyarrow import ipc


def write_fixture(schema, batch_generator, fixture_name, fixtures_dir=None):
    if fixtures_dir is None:
        fixtures_dir = os.getcwd()

    with ipc.new_stream(os.path.join(fixtures_dir, fixture_name), schema) as out:
        for batch in batch_generator:
            out.write_batch(batch)


def write_fixture_float64(
    fixture_name,
    num_cols=10,
    num_batches=2,
    batch_size=65536,
    seed=1938,
    fixtures_dir=None,
):
    """
    Writes a fixture containing random float64 columns in various configurations.
    """
    generator = np.random.default_rng(seed=seed)

    schema = pa.schema({f"col{i}": pa.float64() for i in range(num_cols)})

    def gen_batches():
        for _ in range(num_batches):
            arrays = [np.array(generator.random(batch_size)) for _ in range(num_cols)]
            yield pa.record_batch(arrays, names=[f"col{i}" for i in range(num_cols)])

    write_fixture(schema, gen_batches(), fixture_name, fixtures_dir=fixtures_dir)


if __name__ == "__main__":
    this_dir = os.path.dirname(__file__)
    fixtures_dir = os.path.join(this_dir, "fixtures")

    if not os.path.isdir(fixtures_dir):
        os.mkdir(fixtures_dir)

    write_fixture_float64(
        "float64_basic.arrows",
        num_cols=10,
        num_batches=2,
        batch_size=65536,
        fixtures_dir=fixtures_dir,
    )
    write_fixture_float64(
        "float64_long.arrows",
        num_cols=1,
        num_batches=20,
        batch_size=65536,
        fixtures_dir=fixtures_dir,
    )
    write_fixture_float64(
        "float64_wide.arrows",
        num_cols=1280,
        num_batches=1,
        batch_size=1024,
        fixtures_dir=fixtures_dir,
    )
