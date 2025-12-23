#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
src = Path(__file__).resolve().parents[1] / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from jax.sharding import Mesh
from flax import nnx

from ponderTTT.models import Gemma3Config, TTTConfig, TTTGemma3Model

# Initialize distributed JAX
jax.distributed.initialize()

# Create 4x4 mesh
devices = np.array(jax.devices()).reshape(4, 4)
mesh = Mesh(devices, axis_names=("data", "model"))
print(f"Created mesh: {mesh}")

# Initialize model
gemma_cfg = Gemma3Config.gemma3_4b()
ttt_cfg = TTTConfig(adapter_dim=256, use_norm=True)
rngs = nnx.Rngs(0)

model = TTTGemma3Model(
    gemma_cfg,
    ttt_cfg,
    mesh=mesh,
    rngs=rngs,
)
print("Model initialized")

