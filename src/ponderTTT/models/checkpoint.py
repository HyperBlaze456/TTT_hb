from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_key(key: str) -> str:
    key = key.replace("/", ".")
    while key.startswith("params."):
        key = key[len("params.") :]
    return key


def _flatten_tree(tree: Any, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            flat.update(_flatten_tree(v, (*prefix, str(k))))
        return flat
    if isinstance(tree, Sequence) and not isinstance(tree, (str, bytes, bytearray)):
        for i, v in enumerate(tree):
            flat.update(_flatten_tree(v, (*prefix, str(i))))
        return flat
    flat[".".join(prefix)] = tree
    return flat


def load_checkpoint_tensors(path: str | Path) -> dict[str, Any]:
    """Loads a Flax/JAX checkpoint file into a flat dict of tensors.

    Supported:
      - `.npz` (NumPy archive)
      - `.msgpack` / `.msg` (Flax msgpack serialization)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            tensors = {k: data[k] for k in data.files}
        return {_normalize_key(k): v for k, v in tensors.items()}

    if suffix in {".msgpack", ".msg"}:
        from flax.serialization import msgpack_restore

        tree = msgpack_restore(path.read_bytes())
        flat = _flatten_tree(tree)
        return {_normalize_key(k): v for k, v in flat.items()}

    raise ValueError(f"Unsupported checkpoint format: {path} (expected .npz or .msgpack)")

