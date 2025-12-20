"""Test forward pass for Gemma3 model without loading checkpoints.

The model is initialized with random weights via the pre-sharded initialization,
which is sufficient to test that the forward pass runs correctly.

Designed for multi-controller TPU environments (e.g., 4x4 TPU v6e = 16 devices).
Uses FSDP (data parallelism) + TP (tensor/model parallelism) sharding.

Usage:
    python test_forward_pass.py --model_size 4b --batch_size 16 --seq_len 512
    python test_forward_pass.py --model_size 27b --batch_size 8 --seq_len 256
"""

import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.ponderTTT.models.gemma3 import (
    Gemma3Config,
    Gemma3ForCausalLM,
    create_gemma3_model,
)

jax.distributed.initialize()

def create_mesh(data_axis: str = "data", model_axis: str = "model") -> Mesh:
    """Create a 2D mesh for FSDP + TP on available devices.

    For 4x4 TPU v6e (16 devices), creates a mesh optimized for:
    - data axis: FSDP (batch sharding)
    - model axis: TP (tensor parallelism for large models)

    Returns:
        JAX Mesh configured for the available devices.
    """
    devices = jax.devices()
    num_devices = len(devices)

    # Determine mesh shape based on device count
    # For 16 devices (4x4 TPU): use 4x4 or 2x8 depending on model size
    # For smaller device counts, adjust accordingly
    if num_devices >= 16:
        # 4x4: 4 data parallel, 4 model parallel
        data_parallel = 4
        model_parallel = num_devices // data_parallel
    elif num_devices >= 8:
        data_parallel = 2
        model_parallel = num_devices // data_parallel
    elif num_devices >= 4:
        data_parallel = 2
        model_parallel = 2
    else:
        data_parallel = 1
        model_parallel = num_devices

    print(f"Creating mesh: {data_parallel}x{model_parallel} "
          f"(data_parallel x model_parallel) on {num_devices} devices")

    device_array = np.array(devices).reshape(data_parallel, model_parallel)
    mesh = Mesh(device_array, axis_names=(data_axis, model_axis))

    return mesh


def get_config(model_size: str) -> Gemma3Config:
    """Get model config by size string."""
    config_map = {
        "1b": Gemma3Config.gemma3_1b,
        "4b": Gemma3Config.gemma3_4b,
        "12b": Gemma3Config.gemma3_12b,
        "27b": Gemma3Config.gemma3_27b,
    }
    if model_size.lower() not in config_map:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(config_map.keys())}")
    return config_map[model_size.lower()]()


def test_forward_pass(
    model_size: str = "1b",
    batch_size: int = 8,
    seq_len: int = 128,
    use_mesh: bool = True,
):
    """Test forward pass with specified config and random weights.

    Args:
        model_size: Model size string (1b, 4b, 12b, 27b)
        batch_size: Batch size (should be divisible by data parallel dimension)
        seq_len: Sequence length
        use_mesh: Whether to use distributed mesh for sharding
    """
    print("=" * 70)
    print("Testing Gemma3 Forward Pass (Random Weights)")
    print("=" * 70)

    # Setup mesh for distributed execution
    mesh = None
    if use_mesh and len(jax.devices()) > 1:
        mesh = create_mesh()
    else:
        print("Running on single device (no mesh)")

    # Get config
    config = get_config(model_size)
    print(f"\nConfig: gemma3_{model_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  vocab_size: {config.vocab_size}")

    # Initialize model with random weights (sharded if mesh provided)
    print("\nInitializing model with random weights...")
    if mesh is not None:
        with mesh:
            model = create_gemma3_model(config=config, mesh=mesh, seed=42)
    else:
        model = create_gemma3_model(config=config, seed=42)
    print("Model initialized successfully!")

    # Create batched dummy input
    print(f"\nCreating batched input: batch_size={batch_size}, seq_len={seq_len}")
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # Shard input data across data axis if using mesh
    if mesh is not None:
        input_sharding = NamedSharding(mesh, P("data", None))
        input_ids = jax.device_put(input_ids, input_sharding)
        print(f"Input sharded across 'data' axis")

    print(f"Input shape: {input_ids.shape}")

    # JIT compile the forward pass
    @jax.jit
    def forward_fn(model, input_ids):
        return model(input_ids)

    # Run forward pass
    print("\nRunning forward pass (first call includes JIT compilation)...")
    if mesh is not None:
        with mesh:
            logits = forward_fn(model, input_ids)
    else:
        logits = forward_fn(model, input_ids)

    # Block until computation is done
    logits.block_until_ready()

    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")

    # Verify output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} != {expected_shape}"
    print("Shape verification passed!")

    # Check sharding of output
    if hasattr(logits, 'sharding'):
        print(f"Output sharding: {logits.sharding}")

    # Check for NaN/Inf
    has_nan = bool(jnp.any(jnp.isnan(logits)))
    has_inf = bool(jnp.any(jnp.isinf(logits)))
    print(f"\nContains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")

    if has_nan or has_inf:
        print("WARNING: Output contains NaN or Inf values!")

    # Basic stats (gather to host for printing)
    logits_f32 = logits.astype(jnp.float32)
    print(f"\nLogits stats:")
    print(f"  min: {float(jnp.min(logits_f32)):.4f}")
    print(f"  max: {float(jnp.max(logits_f32)):.4f}")
    print(f"  mean: {float(jnp.mean(logits_f32)):.4f}")
    print(f"  std: {float(jnp.std(logits_f32)):.4f}")

    # Run a second time to measure steady-state performance
    print("\nRunning forward pass again (steady-state, no compilation)...")
    import time
    start = time.perf_counter()
    if mesh is not None:
        with mesh:
            logits2 = forward_fn(model, input_ids)
    else:
        logits2 = forward_fn(model, input_ids)
    logits2.block_until_ready()
    elapsed = time.perf_counter() - start
    print(f"Steady-state forward pass time: {elapsed*1000:.2f} ms")

    print("\n" + "=" * 70)
    print("Forward pass test PASSED!")
    print("=" * 70)

    return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Gemma3 forward pass with random weights on TPU/GPU"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="1b",
        choices=["1b", "4b", "12b", "27b"],
        help="Model size to test (default: 1b)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8, should be divisible by data parallel dim)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length (default: 128)",
    )
    parser.add_argument(
        "--no_mesh",
        action="store_true",
        help="Disable mesh/sharding (run on single device)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Device count: {len(jax.devices())}")
    print(f"Process index: {jax.process_index()}")
    print(f"Process count: {jax.process_count()}")
    print()

    test_forward_pass(
        model_size=args.model_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        use_mesh=not args.no_mesh,
    )

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
