import flax.linen as nn
import jax
import jax.numpy as jnp

from .protocol import QFixProtocol


class FFAdapter(nn.Module):
    """Adapts QFix and AdditiveQFix to timeless batches."""

    module: QFixProtocol

    def __call__(
        self,
        individual_qvalues: jax.Array,
        individual_vvalues: jax.Array,
        states: jax.Array,
        joint_action_n_hot: jax.Array,
    ) -> jax.Array:
        return self.qvalues(
            individual_qvalues,
            individual_vvalues,
            states,
            joint_action_n_hot,
        )

    def qvalues(
        self,
        individual_qvalues: jax.Array,
        individual_vvalues: jax.Array,
        states: jax.Array,
        joint_action_n_hot: jax.Array,
    ) -> jax.Array:
        # individual_qvalues.shape == (N, B)
        # individual_vvalues.shape == (N, B)
        # states.shape == (B, DS)
        # joint_action_n_hot.shape == (B, N*A), N-hot encoding

        individual_qvalues = jnp.expand_dims(individual_qvalues, axis=1)
        # individual_qvalues.shape == (N, T, B)

        individual_vvalues = jnp.expand_dims(individual_vvalues, axis=1)
        # individual_vvalues.shape == (N, T, B)

        states = jnp.expand_dims(states, axis=0)
        # states.shape == (T, B, DS)

        joint_action_n_hot = jnp.expand_dims(joint_action_n_hot, axis=0)
        # joint_action_n_hot.shape == (T, B, N*A), N-hot encoding

        mixed_qvalues = self.module.qvalues(
            individual_qvalues,
            individual_vvalues,
            states,
            joint_action_n_hot,
        )
        # mixed_qvalues.shape == (T, B)
        assert isinstance(mixed_qvalues, jax.Array)

        mixed_qvalues = mixed_qvalues.squeeze(0)
        # mixed_qvalues.shape == (B,)

        return mixed_qvalues

    def vvalues(
        self,
        individual_vvalues: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        # individual_qvalues.shape == (N, B)
        # individual_vvalues.shape == (N, B)
        # states.shape == (B, DS)
        # joint_action_n_hot.shape == (B, N*A), N-hot encoding

        individual_vvalues = jnp.expand_dims(individual_vvalues, axis=1)
        # individual_vvalues.shape == (N, T, B)

        states = jnp.expand_dims(states, axis=0)
        # states.shape == (T, B, DS)

        mixed_vvalues = self.module.vvalues(
            individual_vvalues,
            states,
        )
        # mixed_qvalues.shape == (T, B)
        assert isinstance(mixed_vvalues, jax.Array)

        mixed_vvalues = mixed_vvalues.squeeze(0)
        # mixed_qvalues.shape == (B,)

        return mixed_vvalues
