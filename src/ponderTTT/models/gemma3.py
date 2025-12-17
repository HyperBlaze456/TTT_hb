"""Gemma 3 Text-Only Model Implementation in JAX/Flax NNX.

This implements Gemma 3 with:
- Sliding window attention (LOCAL_SLIDING) interleaved with full attention (GLOBAL)
- Pattern: 5x LOCAL_SLIDING + 1x GLOBAL repeated
- Separate RoPE frequencies for local (10k) and global (1M) attention
- QK normalization, pre/post FFW normalization
- Sharded parameters for distributed computing using nnx.with_partitioning
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .attention_interface import sharded_flash_attention


class AttentionType(enum.Enum):
    """Attention type for each layer."""
    GLOBAL = 1
    LOCAL_SLIDING = 2


@dataclass
class Gemma3Config:
    """Configuration for Gemma 3 model."""
    # Model architecture
    vocab_size: int = 262144
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 34
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256

    # Attention
    sliding_window_size: int = 1024
    max_position_embeddings: int = 131072
    attn_logit_soft_cap: float | None = None
    final_logit_soft_cap: float | None = None
    query_pre_attn_scalar: int | None = None  # For 27B model

    # RoPE
    rope_local_base_freq: float = 10000.0
    rope_global_base_freq: float = 1000000.0
    rope_scaling_factor: float = 8.0

    # Normalization
    rms_norm_eps: float = 1e-6
    use_qk_norm: bool = True
    use_pre_ffw_norm: bool = True
    use_post_ffw_norm: bool = True

    # Other
    attention_dropout: float = 0.0
    hidden_activation: str = "gelu_pytorch_tanh"
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

    @property
    def attention_pattern(self) -> list[AttentionType]:
        """Returns attention pattern: 5x LOCAL_SLIDING + 1x GLOBAL."""
        return [AttentionType.LOCAL_SLIDING] * 5 + [AttentionType.GLOBAL]

    def get_attention_type(self, layer_idx: int) -> AttentionType:
        """Get attention type for a specific layer."""
        pattern = self.attention_pattern
        return pattern[layer_idx % len(pattern)]

    @classmethod
    def gemma3_1b(cls) -> "Gemma3Config":
        """Configuration for Gemma 3 1B model."""
        return cls(
            hidden_size=1152,
            intermediate_size=6912,
            num_hidden_layers=26,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=256,
            sliding_window_size=512,
            max_position_embeddings=32768,
        )

    @classmethod
    def gemma3_4b(cls) -> "Gemma3Config":
        """Configuration for Gemma 3 4B model."""
        return cls(
            hidden_size=2560,
            intermediate_size=10240,
            num_hidden_layers=34,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
            sliding_window_size=1024,
            max_position_embeddings=131072,
        )

    @classmethod
    def gemma3_12b(cls) -> "Gemma3Config":
        """Configuration for Gemma 3 12B model."""
        return cls(
            hidden_size=3840,
            intermediate_size=15360,
            num_hidden_layers=48,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=256,
            sliding_window_size=1024,
            max_position_embeddings=131072,
        )

    @classmethod
    def gemma3_27b(cls) -> "Gemma3Config":
        """Configuration for Gemma 3 27B model."""
        return cls(
            hidden_size=5376,
            intermediate_size=21504,
            num_hidden_layers=62,
            num_attention_heads=32,
            num_key_value_heads=16,
            head_dim=128,
            sliding_window_size=1024,
            max_position_embeddings=131072,
            query_pre_attn_scalar=168,
        )


def sharded_init(
    init_fn: Callable,
    sharding: tuple | None,
    mesh: Mesh | None = None,
) -> Callable:
    """Wrap an initializer with sharding specification.

    Args:
        init_fn: The initializer function
        sharding: Partition spec tuple like (None, 'model') or None for no sharding
        mesh: JAX mesh for distributed computing

    Returns:
        Sharded initializer if mesh is provided, otherwise original initializer
    """
    if mesh is None or sharding is None:
        return init_fn
    return nnx.with_partitioning(init_fn, NamedSharding(mesh, P(*sharding)))


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Precompute rotary embedding frequencies.

    Returns complex exponentials for RoPE: e^(i * theta * position).
    """
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    inv_freq = inv_freq / scaling_factor
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(positions, inv_freq)
    freqs_cis = jnp.exp(1j * freqs).astype(jnp.complex64)
    return freqs_cis


def apply_rotary_emb(
    x: jax.Array,
    freqs_cis: jax.Array,
    position_ids: jax.Array | None = None,
) -> jax.Array:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor [batch, num_heads, seq_len, head_dim]
        freqs_cis: Precomputed frequencies [max_seq_len, head_dim // 2]
        position_ids: Optional position indices [batch, seq_len]

    Returns:
        Tensor with rotary embeddings applied
    """
    batch, num_heads, seq_len, head_dim = x.shape

    if position_ids is not None:
        freqs = freqs_cis[position_ids]
        freqs = freqs[:, None, :, :]
    else:
        freqs = freqs_cis[:seq_len]
        freqs = freqs[None, None, :, :]

    x_complex = x.reshape(batch, num_heads, seq_len, head_dim // 2, 2)
    x_complex = x_complex[..., 0] + 1j * x_complex[..., 1]
    x_rotated = x_complex * freqs
    x_out = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1)
    x_out = x_out.reshape(batch, num_heads, seq_len, head_dim)

    return x_out.astype(x.dtype)


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization with optional unit offset.

    Gemma uses (1 + weight) * normalized instead of weight * normalized.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_features = num_features
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.dtype = dtype

        # Use sharded initializer for the scale parameter
        scale_init = sharded_init(
            nnx.initializers.zeros_init(),
            sharding=("model",) if mesh else None,
            mesh=mesh,
        )
        self.scale = nnx.Param(scale_init(rngs.params(), (num_features,), param_dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        x_float = x.astype(jnp.float32)
        variance = jnp.mean(x_float ** 2, axis=-1, keepdims=True)
        x_normalized = x_float * jax.lax.rsqrt(variance + self.eps)

        scale = self.scale.value.astype(jnp.float32)
        if self.add_unit_offset:
            output = x_normalized * (1.0 + scale)
        else:
            output = x_normalized * scale

        return output.astype(self.dtype)


class Gemma3MLP(nnx.Module):
    """Gemma 3 MLP with gated activation.

    Uses GELU with tanh approximation: gate * up -> down
    """

    def __init__(
        self,
        config: Gemma3Config,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        # Gate projection: [hidden, intermediate] - shard output (intermediate) dim
        self.gate_proj = nnx.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=sharded_init(
                nnx.initializers.xavier_uniform(),
                sharding=(None, "model"),
                mesh=mesh,
            ),
            rngs=rngs,
        )

        # Up projection: [hidden, intermediate] - shard output (intermediate) dim
        self.up_proj = nnx.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=sharded_init(
                nnx.initializers.xavier_uniform(),
                sharding=(None, "model"),
                mesh=mesh,
            ),
            rngs=rngs,
        )

        # Down projection: [intermediate, hidden] - shard input (intermediate) dim
        self.down_proj = nnx.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=sharded_init(
                nnx.initializers.xavier_uniform(),
                sharding=("model", None),
                mesh=mesh,
            ),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = jax.nn.gelu(self.gate_proj(x), approximate=True)
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Gemma3Attention(nnx.Module):
    """Gemma 3 Multi-Head Attention with sliding window support.

    Supports both global and local (sliding window) attention types.
    Uses grouped query attention (GQA) when num_kv_heads < num_heads.
    """

    def __init__(
        self,
        config: Gemma3Config,
        layer_idx: int,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Determine attention type for this layer
        self.attn_type = config.get_attention_type(layer_idx)
        self.sliding_window_size = (
            config.sliding_window_size
            if self.attn_type == AttentionType.LOCAL_SLIDING
            else None
        )

        # Scaling factor for attention
        if config.query_pre_attn_scalar is not None:
            self.scale = config.query_pre_attn_scalar ** -0.5
        else:
            self.scale = self.head_dim ** -0.5

        # Q projection: [hidden, num_heads * head_dim] - shard output (heads) dim
        self.q_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=sharded_init(
                nnx.initializers.xavier_uniform(),
                sharding=(None, "model"),
                mesh=mesh,
            ),
            rngs=rngs,
        )

        # K projection: [hidden, num_kv_heads * head_dim] - shard output (heads) dim
        self.k_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=sharded_init(
                nnx.initializers.xavier_uniform(),
                sharding=(None, "model"),
                mesh=mesh,
            ),
            rngs=rngs,
        )

        # V projection: [hidden, num_kv_heads * head_dim] - shard output (heads) dim
        self.v_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=sharded_init(
                nnx.initializers.xavier_uniform(),
                sharding=(None, "model"),
                mesh=mesh,
            ),
            rngs=rngs,
        )

        # Output projection: [num_heads * head_dim, hidden] - shard input (heads) dim
        self.o_proj = nnx.Linear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=sharded_init(
                nnx.initializers.xavier_uniform(),
                sharding=("model", None),
                mesh=mesh,
            ),
            rngs=rngs,
        )

        # QK normalization (Gemma 3 specific)
        if config.use_qk_norm:
            self.q_norm = RMSNorm(
                num_features=self.head_dim,
                eps=config.rms_norm_eps,
                add_unit_offset=True,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
                mesh=mesh,
                rngs=rngs,
            )
            self.k_norm = RMSNorm(
                num_features=self.head_dim,
                eps=config.rms_norm_eps,
                add_unit_offset=True,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
                mesh=mesh,
                rngs=rngs,
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(
        self,
        x: jax.Array,
        freqs_cis: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        flash_attention_fn=None,
    ) -> jax.Array:
        """Forward pass for attention."""
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization before RoPE
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Transpose to [batch, heads, seq_len, head_dim] for RoPE
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, freqs_cis, position_ids)
        k = apply_rotary_emb(k, freqs_cis, position_ids)

        # Expand K and V for grouped query attention
        if self.num_kv_groups > 1:
            k = jnp.repeat(k, self.num_kv_groups, axis=1)
            v = jnp.repeat(v, self.num_kv_groups, axis=1)

        # Compute attention
        if flash_attention_fn is not None:
            output = flash_attention_fn(q, k, v, segment_ids)
        else:
            output = self._standard_attention(q, k, v, attention_mask)

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch, seq_len, -1)
        output = self.o_proj(output)

        return output

    def _standard_attention(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Standard scaled dot-product attention with optional sliding window."""
        batch, num_heads, q_len, head_dim = q.shape
        kv_len = k.shape[2]

        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale

        if self.config.attn_logit_soft_cap is not None:
            cap = self.config.attn_logit_soft_cap
            attn_weights = cap * jnp.tanh(attn_weights / cap)

        causal_mask = jnp.tril(jnp.ones((q_len, kv_len), dtype=jnp.bool_))

        if self.sliding_window_size is not None:
            positions_q = jnp.arange(q_len)[:, None]
            positions_k = jnp.arange(kv_len)[None, :]
            window_mask = positions_q - positions_k < self.sliding_window_size
            causal_mask = causal_mask & window_mask

        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask

        attn_weights = jnp.where(
            causal_mask[None, None, :, :],
            attn_weights,
            jnp.finfo(attn_weights.dtype).min,
        )

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        return output


class Gemma3Block(nnx.Module):
    """Gemma 3 Transformer Block."""

    def __init__(
        self,
        config: Gemma3Config,
        layer_idx: int,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.layer_idx = layer_idx

        # Pre-attention norm
        self.input_layernorm = RMSNorm(
            num_features=config.hidden_size,
            eps=config.rms_norm_eps,
            add_unit_offset=True,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            mesh=mesh,
            rngs=rngs,
        )

        # Self-attention
        self.self_attn = Gemma3Attention(
            config,
            layer_idx,
            mesh=mesh,
            rngs=rngs,
        )

        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(
            num_features=config.hidden_size,
            eps=config.rms_norm_eps,
            add_unit_offset=True,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            mesh=mesh,
            rngs=rngs,
        )

        # Pre-FFW norm (Gemma 3 specific)
        if config.use_pre_ffw_norm:
            self.pre_feedforward_layernorm = RMSNorm(
                num_features=config.hidden_size,
                eps=config.rms_norm_eps,
                add_unit_offset=True,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
                mesh=mesh,
                rngs=rngs,
            )
        else:
            self.pre_feedforward_layernorm = None

        # MLP
        self.mlp = Gemma3MLP(config, mesh=mesh, rngs=rngs)

        # Post-FFW norm (Gemma 3 specific)
        if config.use_post_ffw_norm:
            self.post_feedforward_layernorm = RMSNorm(
                num_features=config.hidden_size,
                eps=config.rms_norm_eps,
                add_unit_offset=True,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
                mesh=mesh,
                rngs=rngs,
            )
        else:
            self.post_feedforward_layernorm = None

    def __call__(
        self,
        x: jax.Array,
        freqs_cis: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        flash_attention_fn=None,
    ) -> jax.Array:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            freqs_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
            segment_ids=segment_ids,
            flash_attention_fn=flash_attention_fn,
        )
        x = self.post_attention_layernorm(x)
        x = residual + x

        # MLP with residual
        residual = x
        if self.pre_feedforward_layernorm is not None:
            x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        if self.post_feedforward_layernorm is not None:
            x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x


class Gemma3Model(nnx.Module):
    """Gemma 3 Model (without LM head)."""

    def __init__(
        self,
        config: Gemma3Config,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.mesh = mesh

        # Token embeddings - shard along vocab dimension
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            embedding_init=sharded_init(
                nnx.initializers.normal(stddev=1.0),
                sharding=("model", None),
                mesh=mesh,
            ),
            rngs=rngs,
        )

        # Embedding normalizer (Gemma multiplies embeddings by sqrt(hidden_size))
        self.embed_normalizer = math.sqrt(config.hidden_size)

        # Precompute RoPE frequencies for both local and global attention
        self.local_freqs_cis = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            base=config.rope_local_base_freq,
            scaling_factor=config.rope_scaling_factor,
        )
        self.global_freqs_cis = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            base=config.rope_global_base_freq,
            scaling_factor=config.rope_scaling_factor,
        )

        # Transformer layers
        self.layers = []
        for layer_idx in range(config.num_hidden_layers):
            layer = Gemma3Block(
                config,
                layer_idx,
                mesh=mesh,
                rngs=rngs,
            )
            self.layers.append(layer)

        # Final normalization
        self.norm = RMSNorm(
            num_features=config.hidden_size,
            eps=config.rms_norm_eps,
            add_unit_offset=True,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            mesh=mesh,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        flash_attention_fn=None,
    ) -> jax.Array:
        """Forward pass."""
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.embed_normalizer
        hidden_states = hidden_states.astype(self.config.dtype)

        # Create position IDs if not provided
        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = jnp.arange(seq_len)[None, :]

        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            attn_type = self.config.get_attention_type(layer_idx)
            if attn_type == AttentionType.LOCAL_SLIDING:
                freqs_cis = self.local_freqs_cis
            else:
                freqs_cis = self.global_freqs_cis

            hidden_states = layer(
                hidden_states,
                freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                segment_ids=segment_ids,
                flash_attention_fn=flash_attention_fn,
            )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states


class Gemma3ForCausalLM(nnx.Module):
    """Gemma 3 for Causal Language Modeling."""

    def __init__(
        self,
        config: Gemma3Config,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.mesh = mesh
        self.model = Gemma3Model(config, mesh=mesh, rngs=rngs)
        # LM head uses tied embedding weights

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        flash_attention_fn=None,
    ) -> jax.Array:
        """Forward pass for language modeling."""
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            segment_ids=segment_ids,
            flash_attention_fn=flash_attention_fn,
        )

        # Project to vocabulary (tied weights)
        embed_weights = self.model.embed_tokens.embedding.value
        logits = jnp.dot(hidden_states.astype(jnp.float32), embed_weights.T)

        # Apply final logit soft capping if configured
        if self.config.final_logit_soft_cap is not None:
            cap = self.config.final_logit_soft_cap
            logits = cap * jnp.tanh(logits / cap)

        return logits.astype(self.config.dtype)

    def generate(
        self,
        input_ids: jax.Array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Simple autoregressive generation."""
        batch_size = input_ids.shape[0]
        generated = input_ids

        for _ in range(max_new_tokens):
            logits = self(generated)[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
                logits = jnp.full_like(logits, float("-inf"))
                logits = logits.at[jnp.arange(batch_size)[:, None], top_k_indices].set(top_k_logits)

            if top_p is not None:
                sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
                sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = jnp.concatenate([
                    jnp.zeros((batch_size, 1), dtype=jnp.bool_),
                    sorted_indices_to_remove[:, :-1]
                ], axis=-1)
                indices_to_remove = jnp.take_along_axis(
                    sorted_indices_to_remove,
                    jnp.argsort(sorted_indices, axis=-1),
                    axis=-1
                )
                logits = jnp.where(indices_to_remove, float("-inf"), logits)

            probs = jax.nn.softmax(logits, axis=-1)
            next_token = jax.random.categorical(rngs.params(), jnp.log(probs + 1e-10))
            next_token = next_token[:, None]
            generated = jnp.concatenate([generated, next_token], axis=1)

        return generated


def create_gemma3_model(
    config: Gemma3Config | str = "4b",
    mesh: Mesh | None = None,
    seed: int = 0,
) -> Gemma3ForCausalLM:
    """Factory function to create a Gemma 3 model.

    Args:
        config: Model configuration or size string ("1b", "4b", "12b", "27b")
        mesh: Optional JAX mesh for distributed computing
        seed: Random seed for initialization

    Returns:
        Initialized Gemma 3 model
    """
    if isinstance(config, str):
        config_map = {
            "1b": Gemma3Config.gemma3_1b,
            "4b": Gemma3Config.gemma3_4b,
            "12b": Gemma3Config.gemma3_12b,
            "27b": Gemma3Config.gemma3_27b,
        }
        if config.lower() not in config_map:
            raise ValueError(f"Unknown model size: {config}. Choose from {list(config_map.keys())}")
        config = config_map[config.lower()]()

    rngs = nnx.Rngs(seed)
    model = Gemma3ForCausalLM(config, mesh=mesh, rngs=rngs)

    return model


def create_sharded_flash_attention(
    mesh: Mesh,
    config: Gemma3Config,
    causal: bool = True,
) -> callable:
    """Create a sharded flash attention function for use with Gemma 3."""
    sm_scale = 1.0 / math.sqrt(config.head_dim)
    if config.query_pre_attn_scalar is not None:
        sm_scale = 1.0 / math.sqrt(config.query_pre_attn_scalar)

    return sharded_flash_attention(
        mesh=mesh,
        causal=causal,
        sm_scale=sm_scale,
    )