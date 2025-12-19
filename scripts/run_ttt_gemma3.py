#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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


def _run_checked(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _kaggle_download(
    *,
    kind: str,
    handle: str,
    out_dir: Path,
    file: str | None,
    force: bool,
) -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "kaggle CLI not found on PATH. Install it and configure credentials, or pass --checkpoint."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[list[str]] = []
    if kind == "datasets":
        cmd = ["kaggle", "datasets", "download", "-d", handle, "-p", str(out_dir)]
        if force:
            cmd.append("--force")
        if file:
            cmd.extend(["-f", file])
        attempts.append(cmd)
    elif kind == "models":
        # Kaggle CLI model-download syntax has evolved; try a couple variants.
        cmd1 = ["kaggle", "models", "download", handle, "-p", str(out_dir)]
        if force:
            cmd1.append("--force")
        if file:
            cmd1.extend(["-f", file])
        attempts.append(cmd1)

        cmd2 = ["kaggle", "models", "instances", "versions", "download", "-m", handle, "-p", str(out_dir)]
        if force:
            cmd2.append("--force")
        if file:
            cmd2.extend(["-f", file])
        attempts.append(cmd2)
    else:
        raise ValueError(f"Unsupported Kaggle kind: {kind!r}")

    last: subprocess.CompletedProcess[str] | None = None
    for cmd in attempts:
        last = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if last.returncode == 0:
            return

    assert last is not None
    raise RuntimeError(
        "Kaggle download failed. Last attempt output:\n"
        f"{last.stdout}\n"
        "Tip: if you already downloaded weights, pass --checkpoint to the local .msgpack/.npz file."
    )


def _maybe_extract_archives(out_dir: Path) -> None:
    # Kaggle often downloads .zip bundles.
    for zip_path in out_dir.glob("*.zip"):
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)


def _resolve_checkpoint(weights_dir: Path, patterns: list[str]) -> Path:
    matches: list[Path] = []
    for pat in patterns:
        matches.extend([p for p in weights_dir.glob(pat) if p.is_file()])
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found under {weights_dir} with patterns: {patterns}"
        )
    if len(matches) > 1:
        joined = "\n".join(str(p) for p in matches[:20])
        raise RuntimeError(
            "Multiple candidate checkpoints found; pass --checkpoint to disambiguate.\n"
            f"First matches:\n{joined}"
        )
    return matches[0]


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

    parser.add_argument("--kaggle-model", type=str, default=None, help="Kaggle model handle")
    parser.add_argument("--kaggle-dataset", type=str, default=None, help="Kaggle dataset handle")
    parser.add_argument("--kaggle-file", type=str, default=None, help="Specific file to download")
    parser.add_argument("--weights-dir", type=str, default="weights/gemma3", help="Where to download/extract")
    parser.add_argument("--force-download", action="store_true", help="Force Kaggle re-download")
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default="**/*flax_model.msgpack,**/*.msgpack,**/*.msg,**/*.npz",
        help="Comma-separated glob patterns used to locate a checkpoint under --weights-dir",
    )

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
    from ponderTTT.models import load_gemma3_weights_from_checkpoint

    weights_dir = _repo_root() / args.weights_dir
    checkpoint_path: Path | None = Path(args.checkpoint).expanduser() if args.checkpoint else None

    if checkpoint_path is None:
        if args.kaggle_model is None and args.kaggle_dataset is None:
            raise ValueError("Pass --checkpoint or one of --kaggle-model/--kaggle-dataset")

        kind, handle = ("models", args.kaggle_model) if args.kaggle_model else ("datasets", args.kaggle_dataset)
        _kaggle_download(
            kind=kind,
            handle=handle,  # type: ignore[arg-type]
            out_dir=weights_dir,
            file=args.kaggle_file,
            force=args.force_download,
        )
        _maybe_extract_archives(weights_dir)
        patterns = [p.strip() for p in args.checkpoint_glob.split(",") if p.strip()]
        checkpoint_path = _resolve_checkpoint(weights_dir, patterns)

    if not checkpoint_path.exists():
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
