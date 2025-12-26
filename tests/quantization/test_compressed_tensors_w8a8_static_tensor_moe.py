# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for CompressedTensors W8A8 Static Tensor MoE quantization."""

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsW8A8StaticTensorMoEMethod,
)



@pytest.mark.parametrize("num_experts", [8, 32])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
def test_w8a8_static_tensor_channel_wise_moe_quant_method(
    num_experts: int, top_k: int, hidden_size: int, intermediate_size: int
):
    """Test the W8A8 static tensor MoE quantization method with channel-wise weights."""
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

    # Create MoE parallel config
    moe_parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=False,
        all2all_backend="nccl",
    )
    
    # Create MoE config
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        num_local_experts=num_experts,
        moe_parallel_config=moe_parallel_config,
        in_dtype=torch.float16,
    )

    # Create quantization method
    quant_method = CompressedTensorsW8A8StaticTensorMoEMethod(
        weight_quant, input_quant, moe_config
    )

    # Create a mock layer
    layer = torch.nn.Module()
    layer.moe_config = moe_config

    # Test weight creation
    quant_method.create_weights(
        layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        params_dtype=torch.float16,
    )

    # Verify weights are created
    assert hasattr(layer, "w13_weight")
    assert hasattr(layer, "w2_weight")
    assert hasattr(layer, "w13_weight_scale")
    assert hasattr(layer, "w2_weight_scale")
    assert hasattr(layer, "w13_input_scale")
    assert hasattr(layer, "w2_input_scale")

    # Verify weight shapes for channel-wise quantization
    assert layer.w13_weight.shape == (num_experts, 2 * intermediate_size, hidden_size)
    assert layer.w2_weight.shape == (num_experts, hidden_size, intermediate_size)
    assert layer.w13_weight_scale.shape == (num_experts, 2 * intermediate_size, 1)
    assert layer.w2_weight_scale.shape == (num_experts, hidden_size, 1)
    assert layer.w13_input_scale.shape == (num_experts,)
    assert layer.w2_input_scale.shape == (num_experts,)

    # Verify weight dtypes
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w2_weight.dtype == torch.int8
    assert layer.w13_weight_scale.dtype == torch.float32
    assert layer.w2_weight_scale.dtype == torch.float32
    assert layer.w13_input_scale.dtype == torch.float32
    assert layer.w2_input_scale.dtype == torch.float32


def test_is_static_tensor_w8a8():
    """Test the _is_static_tensor_w8a8 detection method."""
    # Test with valid static tensor w8a8
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

    assert CompressedTensorsConfig._is_static_tensor_w8a8(weight_quant, input_quant)

    # Test with channel-wise weights and tensor-wise inputs
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

    assert CompressedTensorsConfig._is_static_tensor_w8a8(weight_quant, input_quant)

    # Test with dynamic inputs (should be False)
    weight_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    )

    assert not CompressedTensorsConfig._is_static_tensor_w8a8(weight_quant, input_quant)

    # Test with different bit sizes (should be False)
    weight_quant = QuantizationArgs(
        num_bits=4,
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

    assert not CompressedTensorsConfig._is_static_tensor_w8a8(weight_quant, input_quant)


def test_invalid_quantization_strategies():
    """Test that invalid quantization strategies raise errors."""
    # Test with token-wise inputs (should raise ValueError)
    weight_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    )

    # Create MoE parallel config
    moe_parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=False,
        all2all_backend="nccl",
    )
    
    moe_config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=512,
        num_local_experts=8,
        moe_parallel_config=moe_parallel_config,
        in_dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="require either tensor-wise or channel-wise"):
        CompressedTensorsW8A8StaticTensorMoEMethod(weight_quant, input_quant, moe_config)

    # Test with dynamic inputs (should raise ValueError)
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
        dynamic=True,
    )

    with pytest.raises(ValueError, match="require static input scales"):
        CompressedTensorsW8A8StaticTensorMoEMethod(weight_quant, input_quant, moe_config)