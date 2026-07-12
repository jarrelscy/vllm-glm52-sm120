#!/usr/bin/env python3
"""Probe SM120 capabilities relevant to DECODE-K ideas (cluster launch,
smem sizes, L2 size + persisting window max)."""
import ctypes

import torch

torch.cuda.init()
dev = 0
props = torch.cuda.get_device_properties(dev)
print("name:", props.name, "cc:", props.major, props.minor,
      "SMs:", props.multi_processor_count)
print("smem/block (default):", props.shared_memory_per_block)
print("smem/block optin:", getattr(props, "shared_memory_per_block_optin", "?"))
print("smem/SM:", props.shared_memory_per_multiprocessor)
print("L2 size MB:", props.L2_cache_size / 1e6)

cuda = ctypes.CDLL("libcudart.so")
val = ctypes.c_int(0)


def attr(n, aid):
    r = cuda.cudaDeviceGetAttribute(ctypes.byref(val), aid, dev)
    print(f"{n} (attr {aid}): {val.value}   (rc={r})")


attr("cudaDevAttrClusterLaunch", 120)
attr("cudaDevAttrMaxPersistingL2CacheSize", 108)
attr("cudaDevAttrMaxAccessPolicyWindowSize", 109)
attr("cudaDevAttrMaxSharedMemoryPerBlockOptin", 97)
attr("cudaDevAttrMaxBlocksPerMultiprocessor", 106)
