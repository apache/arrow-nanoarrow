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

this_dir = os.path.dirname(__file__)
fixtures_dir = os.path.join(this_dir, "fixtures")


def write_fixture(schema, batch_generator, fixture_name):
    with ipc.new_stream(os.path.join(fixtures_dir, fixture_name), schema) as out:
        for batch in batch_generator:
            out.write_batch(batch)


def write_fixture_many_batches():
    """
    A fixture with 10,000 batches, each of which has 5 rows. This is designed to
    benchmark per-batch overhead.
    """
    generator = np.random.default_rng(seed=1938)

    num_batches_pretty_big = int(1e4)
    batch_size = 5
    schema = pa.schema({"col1": pa.float64()})

    def gen_batches():
        for _ in range(num_batches_pretty_big):
            array = np.array(generator.random(batch_size))
            yield pa.record_batch([array], names=["col1"])

    write_fixture(schema, gen_batches(), "many_batches.arrows")
