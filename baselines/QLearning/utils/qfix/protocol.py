from typing import Protocol

import jax


class QFixProtocol(Protocol):
    def __call__(
        self,
        individual_qvalues: jax.Array,
        individual_vvalues: jax.Array,
        states: jax.Array,
        joint_action_n_hot: jax.Array,
    ) -> jax.Array:
        # individual_qvalues.shape == (N, T, B)
        # individual_vvalues.shape == (N, T, B)
        # states.shape == (T, B, DS)
        # joint_action_n_hot.shape == (T, B, N*A), N-hot encoding
        ...

    def qvalues(
        self,
        individual_qvalues: jax.Array,
        individual_vvalues: jax.Array,
        states: jax.Array,
        joint_action_n_hot: jax.Array,
    ) -> jax.Array:
        # individual_qvalues.shape == (N, T, B)
        # individual_vvalues.shape == (N, T, B)
        # states.shape == (T, B, DS)
        # joint_action_n_hot.shape == (T, B, N*A), N-hot encoding
        ...

    def vvalues(
        self,
        individual_vvalues: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        # individual_vvalues.shape == (N, T, B)
        # states.shape == (T, B, DS)
        ...
