#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.sharding import Mesh
from flax import nnx
import optax

from ponderTTT.models import Gemma3Config, TTTConfig, TTTGemma3Model


def force_shard_state(obj, mesh, *, state_filter=None):
    state = nnx.state(obj)
    if state_filter is not None:
        state = nnx.filter_state(state, state_filter)
    shardings = nnx.get_named_sharding(state, mesh)
    state = jax.lax.with_sharding_constraint(state, shardings) # try using nnx.with_sharding_constraint later on
    nnx.update(obj, state)
    return obj

def train_step(model, optimizer, x, y):
    def loss_fn(model):
        logits = model(x)  # [batch, seq_len, vocab_size]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.mean(loss)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def main():
    jax.distributed.initialize()
    print("Multihost ready")
    gemma_cfg = Gemma3Config.gemma3_4b()
    ttt_cfg = TTTConfig(adapter_dim=256, use_norm=True)
    rngs = nnx.Rngs(0)
    print("configs are up")
    nnx.use_eager_sharding(True)
    num_hosts = jax.process_count()
    devices_per_host = len(jax.local_devices())
    devices = np.array(jax.devices()).reshape(num_hosts, devices_per_host)
    mesh = Mesh(devices, axis_names=("data", "model"))
    print(f"Created HSDP mesh: {mesh} (hosts={num_hosts}, devices_per_host={devices_per_host})")
    batch_sharding = NamedSharding(mesh, P("data", None))

    with jax.set_mesh(mesh):
        model = TTTGemma3Model(
            gemma_cfg,
            ttt_cfg,
            mesh=mesh,
            rngs=rngs,
        )
        print("Model was initialized")
        optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)
        print("Optimizer was initialized")
        model = force_shard_state(model, mesh)
        optimizer = force_shard_state(optimizer, mesh, state_filter=nnx.optimizer.OptState)
        print("Starting training...")

        global_batch = 16  # Reduced for 27B model

        per_process = global_batch // jax.process_count()
        seq_len = 128
        # input_ids: [batch, seq_len] of token indices in [0, vocab_size)
        px = np.random.randint(0, gemma_cfg.vocab_size, size=(per_process, seq_len)).astype(np.int32)
        # target_ids: [batch, seq_len] of token indices (typically shifted input)
        py = np.random.randint(0, gemma_cfg.vocab_size, size=(per_process, seq_len)).astype(np.int32)

        x = jax.make_array_from_process_local_data(batch_sharding, px)
        y = jax.make_array_from_process_local_data(batch_sharding, py)

        for step in range(1000):
            loss = train_step(model, optimizer, x, y)
            if step % 100 == 0 and jax.process_index() == 0:
                print(f"step: {step}, loss: {loss}")

if __name__ == "__main__":
    main()
