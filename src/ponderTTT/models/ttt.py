from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from .gemma3 import Gemma3Config, Gemma3Model, RMSNorm, sharded_init


@dataclass(frozen=True)
class TTTConfig:
    adapter_dim: int = 256
    use_norm: bool = True


class TTTAdapter(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        adapter_dim: int,
        *,
        rms_norm_eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        mesh: Mesh | None = None,
        use_norm: bool = True,
        rngs: nnx.Rngs,
    ):
        if adapter_dim <= 0:
            raise ValueError(f"adapter_dim must be > 0, got {adapter_dim}")

        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        self.dtype = dtype

        if use_norm:
            self.norm = RMSNorm(
                num_features=hidden_size,
                eps=rms_norm_eps,
                add_unit_offset=True,
                dtype=dtype,
                param_dtype=param_dtype,
                mesh=mesh,
                rngs=rngs,
            )
        else:
            self.norm = None

        self.down_proj = nnx.Linear(
            in_features=hidden_size,
            out_features=adapter_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=sharded_init(
                nnx.initializers.xavier_uniform(),
                sharding=(None, "model"),
                mesh=mesh,
            ),
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            in_features=adapter_dim,
            out_features=hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=sharded_init(
                nnx.initializers.zeros_init(),
                sharding=("model", None),
                mesh=mesh,
            ),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x
        if self.norm is not None:
            y = self.norm(y)
        y = self.down_proj(y)
        y = jax.nn.gelu(y, approximate=True)
        y = self.up_proj(y)
        return (x + y).astype(self.dtype)


class TTTGemma3Model(nnx.Module):
    """Gemma3 backbone + a trainable post-backbone TTT adapter.

    The TTT adapter is applied after the full Gemma3Model forward pass (hidden states).
    """

    def __init__(
        self,
        gemma_config: Gemma3Config,
        ttt_config: TTTConfig | None = None,
        mesh: Mesh | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = gemma_config
        self.ttt_config = ttt_config or TTTConfig()
        self.mesh = mesh

        self.backbone = Gemma3Model(gemma_config, mesh=mesh, rngs=rngs)
        self.ttt = TTTAdapter(
            hidden_size=gemma_config.hidden_size,
            adapter_dim=self.ttt_config.adapter_dim,
            rms_norm_eps=gemma_config.rms_norm_eps,
            dtype=gemma_config.dtype,
            param_dtype=gemma_config.param_dtype,
            mesh=mesh,
            use_norm=self.ttt_config.use_norm,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        segment_ids: jax.Array | None = None,
        flash_attention_fn=None,
        *,
        return_hidden_states: bool = False,
        return_backbone_hidden_states: bool = False,
    ):
        backbone_hidden_states = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            segment_ids=segment_ids,
            flash_attention_fn=flash_attention_fn,
        )
        hidden_states = self.ttt(backbone_hidden_states)

        embed_weights = self.backbone.embed_tokens.embedding.value
        logits = jnp.dot(hidden_states.astype(jnp.float32), embed_weights.T).astype(self.config.dtype)

        if self.config.final_logit_soft_cap is not None:
            cap = self.config.final_logit_soft_cap
            logits = cap * jnp.tanh(logits / cap)

        if not return_hidden_states and not return_backbone_hidden_states:
            return logits
        if return_hidden_states and not return_backbone_hidden_states:
            return logits, hidden_states
        if not return_hidden_states and return_backbone_hidden_states:
            return logits, backbone_hidden_states
        return logits, hidden_states, backbone_hidden_states
