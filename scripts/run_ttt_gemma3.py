#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    src = _repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_mesh(mesh_str: str | None, *, devices: list[object]):
    if not mesh_str:
        return None
    import jax
    from jax.sharding import Mesh

    jax.distributed.initialize()
    # Format: "data=1,model=-1" (where -1 means "use the remaining devices")
    kvs: list[tuple[str, int]] = []
    for part in mesh_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --mesh fragment {part!r}; expected name=int")
        name, value = part.split("=", 1)
        name = name.strip()
        value = int(value.strip())
        if not name:
            raise ValueError(f"Invalid --mesh axis name in {part!r}")
        kvs.append((name, value))

    if not kvs:
        return None

    axis_names = tuple(k for k, _ in kvs)
    shape = [v for _, v in kvs]
    total = len(devices)

    unknown = [i for i, v in enumerate(shape) if v == -1]
    if len(unknown) > 1:
        raise ValueError("Only one mesh axis can be -1")
    if len(unknown) == 1:
        known_product = 1
        for v in shape:
            if v != -1:
                known_product *= v
        if total % known_product != 0:
            raise ValueError(
                f"Device count {total} not divisible by known mesh product {known_product}"
            )
        shape[unknown[0]] = total // known_product

    product = 1
    for v in shape:
        product *= v
    if product != total:
        raise ValueError(f"Mesh shape {shape} does not match device count {total}")

    mesh_devices = np.array(devices).reshape(shape)
    return Mesh(mesh_devices, axis_names)


def _parse_prompt_ids(prompt_ids: str | None, *, batch: int, seq_len: int, seed: int):
    import jax.numpy as jnp

    if prompt_ids:
        ids = [int(x.strip()) for x in prompt_ids.split(",") if x.strip()]
        if not ids:
            raise ValueError("--prompt-ids was provided but parsed to an empty list")
        if batch != 1:
            raise ValueError("--prompt-ids currently supports --batch=1 only")
        x = jnp.asarray(ids, dtype=jnp.int32)[None, :]
        return x

    rng = np.random.default_rng(seed)
    x = rng.integers(low=0, high=256, size=(batch, seq_len), dtype=np.int32)
    return jnp.asarray(x, dtype=jnp.int32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TTTGemma3Model with a pretrained Gemma3 backbone checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .msgpack/.msg/.npz")

    parser.add_argument("--kaggle-model", type=str, default=None, help="Kaggle model handle (e.g. google/gemma-3/flax/gemma3-4b-it)")
    parser.add_argument("--kaggle-dataset", type=str, default=None, help="Kaggle dataset handle (e.g. owner/dataset)")
    parser.add_argument("--kaggle-file", type=str, default=None, help="Specific file to download from model/dataset")
    parser.add_argument("--force-download", action="store_true", help="Force Kaggle re-download")

    parser.add_argument("--size", type=str, default="4b", choices=["1b", "4b", "12b", "27b"])
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--no-adapter-norm", action="store_true")

    parser.add_argument("--mesh", type=str, default=None, help='e.g. "data=1,model=-1"')
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--jit", action="store_true")

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--prompt-ids", type=str, default=None, help="Comma-separated token ids (batch=1)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.kaggle_model is not None and args.kaggle_dataset is not None:
        raise ValueError("Pass only one of --kaggle-model or --kaggle-dataset")

    import jax
    from flax import nnx

    _ensure_src_on_path()
    from ponderTTT.models import Gemma3Config, TTTConfig, TTTGemma3Model, create_sharded_flash_attention
    from ponderTTT.models import download_gemma3_from_kaggle, load_gemma3_weights_from_checkpoint

    checkpoint_path: str | None = args.checkpoint

    if checkpoint_path is None:
        if args.kaggle_model is None and args.kaggle_dataset is None:
            raise ValueError("Pass --checkpoint or one of --kaggle-model/--kaggle-dataset")

        if args.kaggle_model is not None:
            checkpoint_path = download_gemma3_from_kaggle(
                args.kaggle_model,
                kind="model",
                path=args.kaggle_file,
                force_download=args.force_download,
            )
        else:
            checkpoint_path = download_gemma3_from_kaggle(
                args.kaggle_dataset,
                kind="dataset",
                path=args.kaggle_file,
                force_download=args.force_download,
            )

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg_map = {
        "1b": Gemma3Config.gemma3_1b,
        "4b": Gemma3Config.gemma3_4b,
        "12b": Gemma3Config.gemma3_12b,
        "27b": Gemma3Config.gemma3_27b,
    }
    gemma_cfg = cfg_map[args.size]()

    mesh = _parse_mesh(args.mesh, devices=jax.devices())
    data_axis = None
    if mesh is not None:
        if "data" in mesh.axis_names:
            data_axis = "data"
        elif "batch" in mesh.axis_names:
            data_axis = "batch"

    rngs = nnx.Rngs(args.seed)
    model = TTTGemma3Model(
        gemma_cfg,
        TTTConfig(adapter_dim=args.adapter_dim, use_norm=not args.no_adapter_norm),
        mesh=mesh,
        rngs=rngs,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    load_gemma3_weights_from_checkpoint(model, str(checkpoint_path), strict=True)

    flash_attention_fn = None
    if args.use_flash_attn:
        if mesh is None:
            raise ValueError("--use-flash-attn requires --mesh")
        if data_axis is None or "model" not in mesh.axis_names:
            raise ValueError(
                '--use-flash-attn requires a mesh with axes ("data" or "batch") and "model"'
            )
        flash_attention_fn = create_sharded_flash_attention(
            mesh,
            gemma_cfg,
            causal=True,
            data_axis=data_axis,
            model_axis="model",
        )

    input_ids = _parse_prompt_ids(args.prompt_ids, batch=args.batch, seq_len=args.seq_len, seed=args.seed)
    if input_ids.shape[1] > gemma_cfg.max_position_embeddings:
        raise ValueError(
            f"Sequence length {input_ids.shape[1]} exceeds max_position_embeddings={gemma_cfg.max_position_embeddings}"
        )

    if mesh is not None:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        if data_axis is not None and args.batch % mesh.shape[data_axis] == 0:
            input_ids = jax.device_put(input_ids, NamedSharding(mesh, P(data_axis, None)))
        else:
            # Replicate if we can't shard the batch dimension cleanly.
            input_ids = jax.device_put(input_ids, NamedSharding(mesh, P()))

    def forward(m, x):
        return m(x, flash_attention_fn=flash_attention_fn)

    if args.jit:
        forward = nnx.jit(forward)

    logits = forward(model, input_ids)

    logits = jax.device_get(logits)
    print(f"logits.shape={logits.shape} dtype={logits.dtype}")
    print(f"next_token_argmax={np.asarray(logits[0, -1]).argmax()}")


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
