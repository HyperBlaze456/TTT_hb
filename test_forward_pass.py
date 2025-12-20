"""Test forward pass for Gemma3 model without loading checkpoints.

The model is initialized with random weights via the pre-sharded initialization,
which is sufficient to test that the forward pass runs correctly.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from src.ponderTTT.models.gemma3 import (
    Gemma3Config,
    Gemma3ForCausalLM,
    create_gemma3_model,
)


def test_forward_pass():
    """Test forward pass with 1B config (smallest) and random weights."""
    print("=" * 60)
    print("Testing Gemma3 Forward Pass (Random Weights)")
    print("=" * 60)

    # Use smallest config for faster testing
    config = Gemma3Config.gemma3_1b()
    print(f"\nConfig: gemma3_1b")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  vocab_size: {config.vocab_size}")

    # Initialize model with random weights
    print("\nInitializing model with random weights...")
    model = create_gemma3_model(config=config, seed=42)
    print("Model initialized successfully!")

    # Create dummy input
    batch_size = 2
    seq_len = 16
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    print(f"\nInput shape: {input_ids.shape}")

    # Run forward pass
    print("\nRunning forward pass...")
    logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")

    # Verify output shape
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} != {expected_shape}"
    print("\nShape verification passed!")

    # Check for NaN/Inf
    has_nan = jnp.any(jnp.isnan(logits))
    has_inf = jnp.any(jnp.isinf(logits))
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")

    # Basic stats
    print(f"\nLogits stats:")
    print(f"  min: {float(jnp.min(logits)):.4f}")
    print(f"  max: {float(jnp.max(logits)):.4f}")
    print(f"  mean: {float(jnp.mean(logits)):.4f}")
    print(f"  std: {float(jnp.std(logits)):.4f}")

    print("\n" + "=" * 60)
    print("Forward pass test PASSED!")
    print("=" * 60)

    return logits


def test_generate():
    """Test generation with random weights."""
    print("\n" + "=" * 60)
    print("Testing Gemma3 Generate (Random Weights)")
    print("=" * 60)

    config = Gemma3Config.gemma3_1b()
    model = create_gemma3_model(config=config, seed=42)

    # Create dummy input
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    print(f"\nInput shape: {input_ids.shape}")

    # Generate a few tokens
    print("Generating 5 tokens...")
    rngs = nnx.Rngs(0)
    generated = model.generate(
        input_ids,
        max_new_tokens=5,
        temperature=1.0,
        rngs=rngs,
    )
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated}")

    assert generated.shape == (1, 8), f"Expected shape (1, 8), got {generated.shape}"
    print("\nGenerate test PASSED!")

    return generated


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print()

    test_forward_pass()
    test_generate()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
