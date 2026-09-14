# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Published PV name and legacy hybrid envelopes select the same ARVQ loader."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm.config.model import ModelConfig
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.nvfp4_aqlm_hybrid import (
    NvFp4AqlmHybridConfig,
)
from vllm.model_executor.layers.quantization.nvfp4_arvq_hybrid import (
    NvFp4ArvqHybridConfig,
)


def quant_config(name, version):
    return {
        "quant_method": name,
        "arvq": {
            "format": "rvq256_256x8",
            "version": version,
            "activation_planes": 4,
            "weight_scale_group": 128,
        },
        "aqlm_layer_books": {"3": {"n_nvfp4": 61, "n_base": 0, "n_cold": 195}},
        "nvfp4": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "weights": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "targets": ["Linear"],
                }
            },
            "ignore": [],
        },
    }


@pytest.mark.parametrize(
    "name,version", [("nvfp4_arvq_hybrid", 2), ("nvfp4_aqlm_hybrid", 1)]
)
def test_registry_and_from_config(name, version):
    data = quant_config(name, version)
    original = deepcopy(data)
    selected = get_quantization_config(name)
    result = selected.from_config(data)
    assert isinstance(result, NvFp4ArvqHybridConfig)
    assert result.arvq_format == "rvq256_256x8"
    assert data == original
    assert selected.override_quantization_method(data, None) is None
    assert NvFp4ArvqHybridConfig.get_name() == "nvfp4_arvq_hybrid"
    assert NvFp4AqlmHybridConfig.get_name() == "nvfp4_aqlm_hybrid"


@pytest.mark.parametrize(
    "name,version", [("nvfp4_arvq_hybrid", 2), ("nvfp4_aqlm_hybrid", 1)]
)
@pytest.mark.parametrize("explicit", [False, True])
def test_model_config_quantization_validation(monkeypatch, name, version, explicit):
    import vllm.config.model as model_module

    monkeypatch.setattr(
        model_module.current_platform, "verify_quantization", lambda _: None
    )
    state = SimpleNamespace(
        quantization=name if explicit else None,
        model_arch_config=SimpleNamespace(
            quantization_config=quant_config(name, version)
        ),
        hf_config=SimpleNamespace(),
        allow_deprecated_quantization=False,
    )
    ModelConfig._verify_quantization(state)
    assert state.quantization == name


def test_legacy_aqlm_without_arvq_marker_unchanged():
    data = quant_config("nvfp4_aqlm_hybrid", 1)
    del data["arvq"]
    result = get_quantization_config("nvfp4_aqlm_hybrid").from_config(data)
    assert type(result) is NvFp4AqlmHybridConfig
    assert result.get_name() == "nvfp4_aqlm_hybrid"


def test_legacy_8x7_marker_unchanged():
    data = quant_config("nvfp4_aqlm_hybrid", 1)
    data["arvq"]["format"] = "rvq256_128x8"
    result = get_quantization_config("nvfp4_aqlm_hybrid").from_config(data)
    assert isinstance(result, NvFp4ArvqHybridConfig)
    assert result.arvq_format == "rvq256_128x8"
