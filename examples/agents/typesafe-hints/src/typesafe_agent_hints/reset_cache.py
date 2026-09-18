# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reset one exclusive, idle worker via Dynamo's supported request-plane RPC.

Run only inside the Dynamo environment with the worker's DYN_FILE_KV directory.
This optional helper is not imported by the proxy or ordinary CPU-only tests.
"""

import argparse
import asyncio
import json

from dynamo.runtime import DistributedRuntime


async def main_async(endpoint):
    runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
    try:
        async with asyncio.timeout(30):
            client = await runtime.endpoint(endpoint).client()
            await client.wait_for_instances()
            instances = client.instance_ids()
            if len(instances) != 1:
                raise RuntimeError("Cache reset requires exactly one exclusive worker")
            replies = []
            stream = await client.direct({}, instances[0])
            async for response in stream:
                if response.is_error():
                    raise RuntimeError("Cache reset RPC returned an error")
                data = response.data()
                if isinstance(data, str):
                    data = json.loads(data)
                if data is not None:
                    replies.append(data)
            if len(replies) != 1 or replies[0].get("status") != "success":
                raise RuntimeError(f"Cache reset did not succeed: {replies!r}")
    finally:
        runtime.shutdown()
    print("CACHE_RESET_OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args.endpoint))


if __name__ == "__main__":
    main()
