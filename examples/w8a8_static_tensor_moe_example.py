#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example of using W8A8 Static Tensor MoE quantization.

This example demonstrates how to use the new W8A8 static tensor quantization
for MoE layers in vLLM.
"""

import os
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from vllm.distributed.parallel_state import init_distributed_environment
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


def main():
    """Main function demonstrating W8A8 static tensor MoE quantization."""
    print("W8A8 Static Tensor MoE Quantization Example")
    print("=" * 50)

    # Initialize distributed environment
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://localhost:12345",
        local_rank=0,
        backend="nccl",
    )

    # Create quantization args for static tensor W8A8
    weight_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.TENSOR,  # or CHANNEL for per-channel
        symmetric=True,
        dynamic=False,  # Static quantization
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        strategy=QuantizationStrategy.TENSOR,  # Static per-tensor
        symmetric=True,
        dynamic=False,  # Static quantization
    )

    # Verify that this is detected as static tensor W8A8
    is_static_w8a8 = CompressedTensorsConfig._is_static_tensor_w8a8(
        weight_quant, input_quant
    )
    print(f"Detected as static tensor W8A8: {is_static_w8a8}")

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
        num_experts=8,
        experts_per_token=2,
        hidden_dim=512,
        num_local_experts=8,
        moe_parallel_config=moe_parallel_config,
        in_dtype=torch.float16,
    )

    # Create quantization method
    quant_method = CompressedTensorsW8A8StaticTensorMoEMethod(
        weight_quant, input_quant, moe_config
    )
    print(f"Created quantization method: {quant_method.__class__.__name__}")

    # Create a MoE layer
    layer = FusedMoE(
        num_experts=8,
        top_k=2,
        hidden_size=512,
        intermediate_size=1024,
        params_dtype=torch.float16,
        quant_config=None,
    )

    # Initialize weights
    quant_method.create_weights(
        layer,
        num_experts=8,
        hidden_size=512,
        intermediate_size_per_partition=1024,
        params_dtype=torch.float16,
    )
    print("Created quantized weights for MoE layer")

    # Print weight information
    print("\nWeight Information:")
    print(f"  w13_weight shape: {layer.w13_weight.shape}")
    print(f"  w13_weight dtype: {layer.w13_weight.dtype}")
    print(f"  w2_weight shape: {layer.w2_weight.shape}")
    print(f"  w2_weight dtype: {layer.w2_weight.dtype}")
    print(f"  w13_weight_scale shape: {layer.w13_weight_scale.shape}")
    print(f"  w13_weight_scale dtype: {layer.w13_weight_scale.dtype}")
    print(f"  w2_weight_scale shape: {layer.w2_weight_scale.shape}")
    print(f"  w2_weight_scale dtype: {layer.w2_weight_scale.dtype}")
    print(f"  w13_input_scale shape: {layer.w13_input_scale.shape}")
    print(f"  w13_input_scale dtype: {layer.w13_input_scale.dtype}")
    print(f"  w2_input_scale shape: {layer.w2_input_scale.shape}")
    print(f"  w2_input_scale dtype: {layer.w2_input_scale.dtype}")

    # Set quantization method
    layer.quant_method = quant_method

    # Create input tensors
    batch_size = 4
    hidden_states = torch.randn(batch_size, 512, dtype=torch.float16)
    router_logits = torch.randn(batch_size, 8, dtype=torch.float16)

    print(f"\nInput shape: {hidden_states.shape}")
    print(f"Router logits shape: {router_logits.shape}")

    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = quant_method.apply(
            layer=layer,
            x=hidden_states,
            router_logits=router_logits,
            top_k=2,
            renormalize=True,
        )

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()