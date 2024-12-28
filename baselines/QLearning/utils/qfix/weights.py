import flax.linen as nn
import jax
import jax.numpy as jnp
from .mlp import MLP


def gt_constraint(x: jax.Array, threshold: float) -> jax.Array:
    """Applies greater-than constraint on input."""
    return jnp.abs(x - threshold) + threshold + 1e-10


class W_Module(nn.Module):
    hidden_dim: int
    output_dim: int

    w_delta: float
    w_gt: float

    init_scale: float = 1.0

    def setup(self):
        self.mlp = MLP(
            features=[self.hidden_dim, self.output_dim],
            init_scale=self.init_scale,
        )

    def __call__(self, states: jax.Array, joint_action_n_hot: jax.Array):
        # states.shape == (T, B, DS)
        # joint_action_n_hot.shape == (T, B, N*A)

        inputs = jnp.concatenate([states, joint_action_n_hot], axis=-1)
        # inputs.shape == (T, B, DS + N*A)
        w = self.mlp(inputs)
        # w.shape == (T, B, D)

        w = gt_constraint(w + self.w_delta, self.w_gt)
        # w.shape == (T, B)

        return w


class B_Module(nn.Module):
    hidden_dim: int
    init_scale: float = 1.0

    def setup(self):
        self.mlp = MLP(
            features=[self.hidden_dim, 1],
            init_scale=self.init_scale,
        )

    @nn.compact
    def __call__(self, states: jax.Array):
        # states.shape == (T, B, DS)

        b = self.mlp(states)
        # b.shape == (T, B, 1)

        return b
