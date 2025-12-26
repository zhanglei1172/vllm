# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for CompressedTensors W8A8 Static Tensor MoE quantization."""

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsW8A8StaticTensorMoEMethod,
)


@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [1024])
@pytest.mark.parametrize("batch_size", [4])
def test_w8a8_static_tensor_moe_forward(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int,
):
    """Test forward pass with W8A8 static tensor quantization."""
    # Create quantization args
    weight_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )

    # Create MoE config
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    # Create quantization method
    quant_method = CompressedTensorsW8A8StaticTensorMoEMethod(
        weight_quant, input_quant, moe_config
    )

    # Create a MoE layer
    layer = FusedMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=torch.float16,
        quant_config=None,
    )

    # Initialize weights
    quant_method.create_weights(
        layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        params_dtype=torch.float16,
    )

    # Set quantization method
    layer.quant_method = quant_method

    # Create input tensor
    hidden_states = torch.randn(batch_size, hidden_size, dtype=torch.float16)
    router_logits = torch.randn(batch_size, num_experts, dtype=torch.float16)

    # Run forward pass
    with torch.no_grad():
        output = quant_method.apply(
            layer=layer,
            x=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=True,
        )

    # Verify output shape
    assert output.shape == (batch_size, hidden_size)

    # Verify output dtype
    assert output.dtype == torch.float16


@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [1024])
@pytest.mark.parametrize("batch_size", [4])
def test_w8a8_static_tensor_channel_wise_moe_forward(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int,
):
    """Test forward pass with W8A8 static tensor quantization and channel-wise weights."""
    # Create quantization args
    weight_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )

    # Create MoE config
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    # Create quantization method
    quant_method = CompressedTensorsW8A8StaticTensorMoEMethod(
        weight_quant, input_quant, moe_config
    )

    # Create a MoE layer
    layer = FusedMoE(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=torch.float16,
        quant_config=None,
    )

    # Initialize weights
    quant_method.create_weights(
        layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        params_dtype=torch.float16,
    )

    # Set quantization method
    layer.quant_method = quant_method

    # Create input tensor
    hidden_states = torch.randn(batch_size, hidden_size, dtype=torch.float16)
    router_logits = torch.randn(batch_size, num_experts, dtype=torch.float16)

    # Run forward pass
    with torch.no_grad():
        output = quant_method.apply(
            layer=layer,
            x=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=True,
        )

    # Verify output shape
    assert output.shape == (batch_size, hidden_size)

    # Verify output dtype
    assert output.dtype == torch.float16


def test_moe_method_selection():
    """Test that the correct MoE method is selected for static tensor W8A8."""
    # Create a mock quant config
    quant_config = {
        "model": {
            "test_layer.0.gate_proj": {
                "weights": {
                    "num_bits": 8,
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                },
            },
            "test_layer.0.up_proj": {
                "weights": {
                    "num_bits": 8,
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                },
            },
            "test_layer.0.down_proj": {
                "weights": {
                    "num_bits": 8,
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                },
            },
        }
    }

    # Create a CompressedTensorsConfig
    config = CompressedTensorsConfig(
        target_scheme_map=quant_config["model"],
        ignore=[],
        quant_format="pack_quantized",
        sparsity_scheme_map={},
        sparsity_ignore_list=[],
    )

    # Create a mock layer
    layer = torch.nn.Module()
    layer.moe_config = FusedMoEConfig(
        num_experts=8,
        top_k=2,
        hidden_size=512,
        intermediate_size=1024,
    )

    # Get the MoE method
    moe_method = CompressedTensorsW8A8StaticTensorMoEMethod.get_moe_method(
        config, layer, "test_layer"
    )

    # Verify the correct method is selected
    assert isinstance(moe_method, CompressedTensorsW8A8StaticTensorMoEMethod)