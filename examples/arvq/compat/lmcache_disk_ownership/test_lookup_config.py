# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the explicit TP4/DCP4 all-shard lookup configuration."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.factory import LookupClientFactory
from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient


class Tests(unittest.TestCase):
    def test_real_environment_config_and_server_selection(self):
        with patch.dict(os.environ, {"LMCACHE_LOOKUP_SERVER_WORKER_IDS": "0,1,2,3"}):
            config = LMCacheEngineConfig.from_env()
        self.assertEqual(config.get_lookup_server_worker_ids(True, 4), [0, 1, 2, 3])
        engine = SimpleNamespace(config=config)
        with (
            patch.object(LookupClientFactory, "_create_zmq_server_transport"),
            patch(
                "lmcache.v1.lookup_client.lmcache_lookup_client.LMCacheLookupServer"
            ) as server,
        ):
            for rank in range(4):
                metadata = SimpleNamespace(use_mla=True, world_size=4, worker_id=rank)
                LookupClientFactory.create_lookup_server(engine, metadata)
            self.assertEqual(server.call_count, 4)

    def test_existing_client_uses_minimum_shard_prefix(self):
        for replies, expected in (
            ([8192, 8192, 0, 8192], 0),
            ([8192, 4096, 8192, 8192], 4096),
            ([8192] * 4, 8192),
        ):
            client = object.__new__(LMCacheLookupClient)
            client.enable_blending = False
            client.token_database = SimpleNamespace(
                process_tokens=lambda *args, **kwargs: [(0, 8192, 123)]
            )
            client.transport = SimpleNamespace(
                world_size=4,
                send_and_recv_all=lambda _, values=replies: [
                    x.to_bytes(8, "big") for x in values
                ],
            )
            client.reqs_status = {}
            self.assertEqual(client.lookup([1] * 8192, "fixture"), expected)


if __name__ == "__main__":
    unittest.main()
