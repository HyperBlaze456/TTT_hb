from __future__ import annotations

"""Utilities for loading pretrained Gemma3 weights.

This is intended for loading local checkpoint files (e.g., downloaded from Kaggle)
into the NNX implementation in `Gemma3Model` / `TTTGemma3Model`.

Example:
  ```
  from flax import nnx
  import jax.numpy as jnp
  from ponderTTT.models import Gemma3Config, TTTConfig, TTTGemma3Model
  from ponderTTT.models import load_gemma3_weights_from_checkpoint

  cfg = Gemma3Config.gemma3_4b()
  model = TTTGemma3Model(cfg, TTTConfig(adapter_dim=256), rngs=nnx.Rngs(0))
  load_gemma3_weights_from_checkpoint(model, "/path/to/flax_model.msgpack")

  logits = model(jnp.zeros((1, 16), dtype=jnp.int32))
  ```
"""

from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from .checkpoint import load_checkpoint_tensors
from .gemma3 import Gemma3ForCausalLM, Gemma3Model


def _find_unique_suffix(tensors: dict[str, Any], suffix: str) -> str | None:
    matches = [k for k in tensors.keys() if k.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


def _lookup_tensor(
    tensors: dict[str, Any],
    candidates: Iterable[str],
    *,
    required: bool,
    name: str,
) -> Any | None:
    for key in candidates:
        if key in tensors:
            return tensors[key]

    for key in candidates:
        suffix = key.split(".", 1)[-1]
        match = _find_unique_suffix(tensors, suffix)
        if match is not None:
            return tensors[match]

    if required:
        raise KeyError(f"Missing tensor for {name}. Tried: {sorted(set(candidates))}")
    return None


def _detect_backbone_prefix(tensors: dict[str, Any]) -> str:
    for prefix in ("model.", "", "transformer.", "decoder.", "gemma."):
        if (
            f"{prefix}embed_tokens.embedding" in tensors
            or f"{prefix}embed_tokens.weight" in tensors
        ):
            return prefix

    match = _find_unique_suffix(tensors, "embed_tokens.embedding")
    if match is not None:
        return match[: -len("embed_tokens.embedding")]
    match = _find_unique_suffix(tensors, "embed_tokens.weight")
    if match is not None:
        return match[: -len("embed_tokens.weight")]

    raise KeyError(
        "Could not find backbone tensors; expected a key ending with "
        "`embed_tokens.embedding` or `embed_tokens.weight`."
    )


def _assign(param: nnx.Param, value: Any, *, name: str) -> None:
    target = param.value
    array = jnp.asarray(value)

    if array.shape != target.shape:
        if array.ndim == 2 and array.T.shape == target.shape:
            array = array.T
        else:
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint {array.shape} vs model {target.shape}"
            )

    array = array.astype(target.dtype)

    try:
        sharding = target.sharding
    except AttributeError:
        sharding = None
    if sharding is not None:
        array = jax.device_put(array, sharding)

    param.value = array


def load_gemma3_backbone_weights(
    backbone: Gemma3Model,
    tensors: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Loads pretrained weights into a `Gemma3Model` (backbone only)."""
    prefix = _detect_backbone_prefix(tensors)

    _assign(
        backbone.embed_tokens.embedding,
        _lookup_tensor(
            tensors,
            (f"{prefix}embed_tokens.embedding", f"{prefix}embed_tokens.weight"),
            required=True,
            name="embed_tokens.embedding",
        ),
        name="embed_tokens.embedding",
    )

    final_norm = _lookup_tensor(
        tensors,
        (f"{prefix}norm.scale", f"{prefix}norm.weight"),
        required=strict,
        name="norm.scale",
    )
    if final_norm is not None:
        _assign(backbone.norm.scale, final_norm, name="norm.scale")

    for layer_idx, layer in enumerate(backbone.layers):
        layer_prefixes = (
            f"{prefix}layers.{layer_idx}.",
            f"{prefix}layers_{layer_idx}.",
        )

        def lk(suffixes: Iterable[str], *, required: bool, name: str):
            keys = [p + s for p in layer_prefixes for s in suffixes]
            return _lookup_tensor(tensors, keys, required=required, name=name)

        # Norms
        input_ln = lk(
            ("input_layernorm.scale", "input_layernorm.weight"),
            required=strict,
            name=f"layers[{layer_idx}].input_layernorm.scale",
        )
        if input_ln is not None:
            _assign(layer.input_layernorm.scale, input_ln, name=f"layers[{layer_idx}].input_layernorm.scale")

        post_attn_ln = lk(
            ("post_attention_layernorm.scale", "post_attention_layernorm.weight"),
            required=strict,
            name=f"layers[{layer_idx}].post_attention_layernorm.scale",
        )
        if post_attn_ln is not None:
            _assign(
                layer.post_attention_layernorm.scale,
                post_attn_ln,
                name=f"layers[{layer_idx}].post_attention_layernorm.scale",
            )

        if layer.pre_feedforward_layernorm is not None:
            pre_ff_ln = lk(
                ("pre_feedforward_layernorm.scale", "pre_feedforward_layernorm.weight"),
                required=strict,
                name=f"layers[{layer_idx}].pre_feedforward_layernorm.scale",
            )
            if pre_ff_ln is not None:
                _assign(
                    layer.pre_feedforward_layernorm.scale,
                    pre_ff_ln,
                    name=f"layers[{layer_idx}].pre_feedforward_layernorm.scale",
                )

        if layer.post_feedforward_layernorm is not None:
            post_ff_ln = lk(
                ("post_feedforward_layernorm.scale", "post_feedforward_layernorm.weight"),
                required=strict,
                name=f"layers[{layer_idx}].post_feedforward_layernorm.scale",
            )
            if post_ff_ln is not None:
                _assign(
                    layer.post_feedforward_layernorm.scale,
                    post_ff_ln,
                    name=f"layers[{layer_idx}].post_feedforward_layernorm.scale",
                )

        # Attention projections
        q_proj = lk(
            ("self_attn.q_proj.kernel", "self_attn.q_proj.weight"),
            required=strict,
            name=f"layers[{layer_idx}].self_attn.q_proj.kernel",
        )
        if q_proj is not None:
            _assign(layer.self_attn.q_proj.kernel, q_proj, name=f"layers[{layer_idx}].self_attn.q_proj.kernel")

        k_proj = lk(
            ("self_attn.k_proj.kernel", "self_attn.k_proj.weight"),
            required=strict,
            name=f"layers[{layer_idx}].self_attn.k_proj.kernel",
        )
        if k_proj is not None:
            _assign(layer.self_attn.k_proj.kernel, k_proj, name=f"layers[{layer_idx}].self_attn.k_proj.kernel")

        v_proj = lk(
            ("self_attn.v_proj.kernel", "self_attn.v_proj.weight"),
            required=strict,
            name=f"layers[{layer_idx}].self_attn.v_proj.kernel",
        )
        if v_proj is not None:
            _assign(layer.self_attn.v_proj.kernel, v_proj, name=f"layers[{layer_idx}].self_attn.v_proj.kernel")

        o_proj = lk(
            ("self_attn.o_proj.kernel", "self_attn.o_proj.weight"),
            required=strict,
            name=f"layers[{layer_idx}].self_attn.o_proj.kernel",
        )
        if o_proj is not None:
            _assign(layer.self_attn.o_proj.kernel, o_proj, name=f"layers[{layer_idx}].self_attn.o_proj.kernel")

        # QK norms (if present in this config)
        if layer.self_attn.q_norm is not None:
            qn = lk(
                ("self_attn.q_norm.scale", "self_attn.q_norm.weight"),
                required=strict,
                name=f"layers[{layer_idx}].self_attn.q_norm.scale",
            )
            if qn is not None:
                _assign(
                    layer.self_attn.q_norm.scale,
                    qn,
                    name=f"layers[{layer_idx}].self_attn.q_norm.scale",
                )
        if layer.self_attn.k_norm is not None:
            kn = lk(
                ("self_attn.k_norm.scale", "self_attn.k_norm.weight"),
                required=strict,
                name=f"layers[{layer_idx}].self_attn.k_norm.scale",
            )
            if kn is not None:
                _assign(
                    layer.self_attn.k_norm.scale,
                    kn,
                    name=f"layers[{layer_idx}].self_attn.k_norm.scale",
                )

        # MLP projections
        gate = lk(
            ("mlp.gate_proj.kernel", "mlp.gate_proj.weight"),
            required=strict,
            name=f"layers[{layer_idx}].mlp.gate_proj.kernel",
        )
        if gate is not None:
            _assign(layer.mlp.gate_proj.kernel, gate, name=f"layers[{layer_idx}].mlp.gate_proj.kernel")

        up = lk(
            ("mlp.up_proj.kernel", "mlp.up_proj.weight"),
            required=strict,
            name=f"layers[{layer_idx}].mlp.up_proj.kernel",
        )
        if up is not None:
            _assign(layer.mlp.up_proj.kernel, up, name=f"layers[{layer_idx}].mlp.up_proj.kernel")

        down = lk(
            ("mlp.down_proj.kernel", "mlp.down_proj.weight"),
            required=strict,
            name=f"layers[{layer_idx}].mlp.down_proj.kernel",
        )
        if down is not None:
            _assign(layer.mlp.down_proj.kernel, down, name=f"layers[{layer_idx}].mlp.down_proj.kernel")


def load_gemma3_weights_from_checkpoint(
    model: Gemma3Model | Gemma3ForCausalLM | Any,
    checkpoint_path: str,
    *,
    strict: bool = True,
) -> None:
    """Loads a pretrained Gemma3 checkpoint into an NNX model.

    Supports:
      - `Gemma3Model`
      - `Gemma3ForCausalLM` (loads into `.model`)
      - `TTTGemma3Model` (loads into `.backbone`)
    """
    tensors = load_checkpoint_tensors(checkpoint_path)

    if isinstance(model, Gemma3ForCausalLM):
        backbone = model.model
    elif isinstance(model, Gemma3Model):
        backbone = model
    elif hasattr(model, "backbone") and isinstance(model.backbone, Gemma3Model):
        backbone = model.backbone
    else:
        raise TypeError(
            "Unsupported model type for Gemma3 loading. Expected Gemma3Model, "
            "Gemma3ForCausalLM, or an object with `.backbone: Gemma3Model`."
        )

    load_gemma3_backbone_weights(backbone, tensors, strict=strict)
