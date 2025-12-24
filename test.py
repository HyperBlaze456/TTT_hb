import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

def init_distributed():
    jax.distributed.initialize()

def make_mesh():
    # Example: pure data-parallel mesh (1D)
    mesh = jax.make_mesh((jax.device_count(),), ('data',))
    return mesh

class Model(nnx.Module):
    def __init__(self, d, *, rngs: nnx.Rngs):
        init = nnx.initializers.lecun_normal()
        self.l1 = nnx.Linear(
            d, d,
            kernel_init=nnx.with_partitioning(init, (None, 'data')),  # FSDP-ish
            rngs=rngs,
        )
        self.l2 = nnx.Linear(
            d, d,
            kernel_init=nnx.with_partitioning(init, ('data', None)),  # FSDP-ish
            rngs=rngs,
        )

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))

def force_shard_state(obj, mesh, *, state_filter=None):
    state = nnx.state(obj)
    if state_filter is not None:
        state = nnx.filter_state(state, state_filter)
    shardings = nnx.get_named_sharding(state, mesh)
    state = jax.lax.with_sharding_constraint(state, shardings)
    nnx.update(obj, state)
    return obj

@nnx.jit(donate_argnames=("model", "optimizer"))
def train_step(model, optimizer, x, y):
    def loss_fn(m):
        pred = m(x)
        return jnp.mean((pred - y) ** 2)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def main():
    init_distributed()

    nnx.use_eager_sharding(True)
    mesh = make_mesh()
    batch_sharding = NamedSharding(mesh, P('data', None))

    with jax.set_mesh(mesh):
        model = Model(1024, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)

        # Make sure model + optimizer state are sharded as intended
        model = force_shard_state(model, mesh)
        optimizer = force_shard_state(optimizer, mesh, state_filter=nnx.optimizer.OptState)

        # per-process batch
        global_batch = 1024
        per_process = global_batch // jax.process_count()
        px = np.random.randn(per_process, 1024).astype(np.float32)
        py = np.random.randn(per_process, 1024).astype(np.float32)

        x = jax.make_array_from_process_local_data(batch_sharding, px)
        y = jax.make_array_from_process_local_data(batch_sharding, py)

        for step in range(1000):
            loss = train_step(model, optimizer, x, y)
            if step % 100 == 0 and jax.process_index() == 0:
                print("step", step, "loss", float(loss))

if __name__ == "__main__":
    main()
