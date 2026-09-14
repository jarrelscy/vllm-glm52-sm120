# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary-filesystem regression using real LMCache dependencies, CPU only."""

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from lmcache.utils import CacheEngineKey

TARGET = Path(
    os.environ.get(
        "LMCACHE_DISK_TEST_TARGET",
        (
            "/opt/vllm/.venv/lib/python3.12/site-packages/"
            "lmcache/v1/storage_backend/local_disk_backend.py"
        ),
    )
)
spec = importlib.util.spec_from_file_location("disk_fix_test_target", TARGET)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def Key(model_name, world_size, worker_id, chunk_hash="abcdef"):
    return CacheEngineKey(
        model_name, world_size, worker_id, int(chunk_hash, 16), torch.uint8
    )


class Meta:
    def __init__(self, path, size):
        self.path, self.size = path, size


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.b = object.__new__(module.LocalDiskBackend)
        self.b.path = self.tmp.name
        self.b.metadata = SimpleNamespace(
            model_name="org/model", world_size=4, worker_id=1
        )
        self.b.dict = {}
        self.b.disk_lock = threading.Lock()
        self.b.current_cache_size = self.b.usage = 0
        self.b.stats_monitor = SimpleNamespace(
            update_local_storage_usage=lambda _: None
        )
        self.evicted = []
        self.b.cache_policy = SimpleNamespace(update_on_force_evict=self.evicted.append)
        self.b.batched_msg_sender = None

    def seed(self, key, data=True):
        path = Path(self.b._key_to_path(key))
        if data:
            path.write_bytes(b"12345678")
        Path(str(path) + ".meta").write_text(
            json.dumps(
                {
                    "lmcache_version": 2,
                    "shape": [8],
                    "dtype": "uint8",
                    "fmt": 4,
                    "size": 8,
                }
            )
        )
        return path

    def test_rebuild_only_own_rank_world_model(self):
        keys = [Key("org/model", 4, r) for r in range(4)] + [
            Key("org/model", 8, 1),
            Key("other/model", 4, 1),
        ]
        for k in keys:
            self.seed(k)
        self.b._rebuild_index_sync()
        self.assertEqual(set(self.b.dict), {keys[1]})
        self.assertEqual((self.b.usage, self.b.current_cache_size), (8, 8))
        self.b._rebuild_index_sync()
        self.assertEqual(self.b.usage, 8)
        self.assertEqual(len(list(Path(self.tmp.name).iterdir())), 12)

    def test_foreign_orphan_preserved(self):
        p = self.seed(Key("org/model", 4, 2), data=False)
        self.b._rebuild_index_sync()
        self.assertTrue(Path(str(p) + ".meta").exists())

    def test_lazy_own_only(self):
        own, foreign = Key("org/model", 4, 1), Key("org/model", 4, 2)
        self.seed(own)
        self.seed(foreign)
        self.assertFalse(self.b._try_lazy_load_metadata(foreign))
        self.assertTrue(self.b._try_lazy_load_metadata(own))
        self.assertEqual(self.b.usage, 8)

    def test_missing_eviction_idempotent(self):
        k = Key("org/model", 4, 1)
        p = self.seed(k)
        self.b._rebuild_index_sync()
        p.unlink()
        self.assertTrue(self.b.remove(k))
        self.assertFalse(self.b.remove(k))
        self.assertEqual(self.b.usage, 0)
        self.assertFalse(Path(str(p) + ".meta").exists())
        self.assertEqual(self.evicted, [k])

    def test_foreign_remove_refused(self):
        k = Key("org/model", 4, 2)
        p = self.seed(k)
        self.b.dict[k] = Meta(str(p), 8)
        self.assertFalse(self.b.remove(k))
        self.assertTrue(p.exists())

    def test_permission_error_not_hidden(self):
        k = Key("org/model", 4, 1)
        self.seed(k)
        self.b._rebuild_index_sync()
        with (
            patch.object(os, "remove", side_effect=PermissionError("denied")),
            self.assertRaises(PermissionError),
        ):
            self.b.remove(k)

    def test_metadata_free_explicit_key_unchanged(self):
        self.b.metadata = None
        self.assertTrue(self.b._owns_persisted_key(Key("any/model", 8, 7)))


if __name__ == "__main__":
    unittest.main()
