# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import deepseek_v2 as model
from vllm.sequence import IntermediateTensors


@pytest.mark.parametrize("index_topk", [None, 8])
def test_pipeline_schema_preserves_index_dtype_and_aux_states(index_topk):
    factory = model._make_dsa_intermediate_tensors_factory(
        ["hidden_states", "residual", "aux_0"], 16, index_topk
    )
    tensors = factory(3, torch.bfloat16, torch.device("cpu"))
    assert tensors["aux_0"].shape == (3, 16)
    assert tensors["hidden_states"].dtype == torch.bfloat16
    if index_topk is None:
        assert "indexer_topk" not in tensors.tensors
    else:
        assert tensors["indexer_topk"].shape == (3, index_topk)
        assert tensors["indexer_topk"].dtype == torch.int32
        assert (tensors["indexer_topk"] == -1).all()


@pytest.mark.parametrize("aux_layers", [(), (1,)])
def test_stage_starting_with_shared_indexer_receives_current_batch(aux_layers):
    # Execute the production forward body with tiny CPU stages, bypassing
    # model construction and the class's compilation wrapper.
    tree = ast.parse(Path(model.__file__).read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2Model"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    group = SimpleNamespace(is_first_rank=True, is_last_rank=False)
    namespace = dict(vars(model), get_pp_group=lambda: group)
    exec(
        compile(ast.Module(body=[forward], type_ignores=[]), model.__file__, "exec"),
        namespace,
    )
    run = namespace["forward"]
    expected = None

    def stage(first):
        buffer = torch.full((4, 8), -999, dtype=torch.int32)

        class Layer:
            use_sequence_parallel_moe = False

            def __call__(self, positions, hidden, residual, scaling):
                if first:
                    buffer[: len(positions)].copy_(expected)
                else:
                    torch.testing.assert_close(buffer[: len(positions)], expected)
                if residual is None:
                    residual = torch.zeros_like(hidden)
                return hidden, residual

        return SimpleNamespace(
            config=SimpleNamespace(),
            hidden_size=16,
            start_layer=0,
            end_layer=2,
            layers=[Layer(), Layer()],
            aux_hidden_state_layers=aux_layers,
            pp_index_topk=8,
            topk_indices_buffer=buffer,
            pcie_fuse_final_norm=False,
            norm=lambda hidden, residual: (hidden + residual, residual),
        )

    producer, consumer = stage(True), stage(False)
    for batch_size, offset in [(3, 70001), (1, 90003)]:
        expected = torch.arange(batch_size * 8, dtype=torch.int32).reshape(-1, 8)
        expected += offset
        expected[:, -1] = -1
        positions = torch.arange(batch_size)
        group.is_first_rank, group.is_last_rank = True, False
        sent = run(
            producer, None, positions, None, inputs_embeds=torch.ones(batch_size, 16)
        )
        assert isinstance(sent, IntermediateTensors)
        assert sent["indexer_topk"].dtype == torch.int32
        # Copy into distinct receive storage, as a pipeline transfer would.
        received = IntermediateTensors({k: v.clone() for k, v in sent.items()})
        consumer.topk_indices_buffer.fill_(-999)
        group.is_first_rank, group.is_last_rank = False, True
        run(consumer, None, positions, received)


def test_aux_schema_update_keeps_indexer_indices():
    owner = SimpleNamespace(
        model=SimpleNamespace(
            hidden_size=16, pp_index_topk=8, aux_hidden_state_layers=()
        )
    )
    model.DeepseekV2ForCausalLM.set_aux_hidden_state_layers(owner, (2, 5))
    tensors = owner.make_empty_intermediate_tensors(
        3, torch.bfloat16, torch.device("cpu")
    )
    assert set(tensors.tensors) == {
        "hidden_states",
        "residual",
        "aux_0",
        "aux_1",
        "indexer_topk",
    }
    assert tensors["indexer_topk"].dtype == torch.int32
