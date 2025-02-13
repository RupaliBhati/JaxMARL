import flax.linen as nn
import jax
import jax.numpy as jnp

from .weights import B_Module, W_Module


class QFixLin(nn.Module):
    """
    QFIX-lin fixing network for projecting IGM-incomplete fixees into IGM-complete values.
    """

    hidden_size: int
    num_agents: int

    w_delta: float
    w_gt: float

    debug_recover_fixee: bool

    def setup(self):
        if not self.debug_recover_fixee:
            self.w_module = W_Module(
                self.hidden_size,
                self.num_agents,
                self.w_delta,
                self.w_gt,
            )
            self.b_module = B_Module(self.hidden_size)

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
        # individual_qvalues.shape == (N, T, B)
        # individual_vvalues.shape == (N, T, B)
        # states.shape == (T, B, DS)
        # joint_action_n_hot.shape == (T, B, N*A), N-hot encoding

        individual_advantages = individual_qvalues - individual_vvalues
        # fixee_advantages.shape == (N, T, B)

        if self.debug_recover_fixee:
            N, T, B = individual_advantages.shape
            w = jnp.ones((T, B, N))
            # w.shape == (T, B, N)
            b = individual_vvalues.sum(axis=0)
            # b.shape == (T, B)
        else:
            w = self.w_module(states, joint_action_n_hot)
            # w.shape == (T, B, N)
            b = self.b_module(states).squeeze(-1)
            # b.shape == (T, B)

        joint_qvalues = jnp.einsum("TBN,NTB->TB", w, individual_advantages) + b
        # joint_qvalues.shape == (T, B)

        return joint_qvalues

    def vvalues(
        self,
        individual_vvalues: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        # individual_vvalues.shape == (N, T, B)
        # states.shape == (T, B, DS)

        b = self.b_module(states).squeeze(-1)
        # b.shape == (T, B)

        joint_vvalues = b
        # joint_vvalues.shape == (T, B)

        return joint_vvalues


class AdditiveQFixLin(nn.Module):
    """
    Q+FIX-lin fixing network for projecting IGM-incomplete fixees into IGM-complete values.
    """

    hidden_size: int
    num_agents: int

    w_delta: float
    w_gt: float
    detach_advantages: bool

    debug_recover_fixee: bool

    def setup(self):
        if not self.debug_recover_fixee:
            self.w_module = W_Module(
                self.hidden_size,
                self.num_agents,
                self.w_delta,
                self.w_gt,
            )
            self.b_module = B_Module(self.hidden_size)

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
        # individual_qvalues.shape == (N, T, B)
        # individual_vvalues.shape == (N, T, B)
        # states.shape == (T, B, DS)
        # joint_action_n_hot.shape == (T, B, N*A), N-hot encoding

        fixee_qvalues = individual_qvalues.sum(axis=0)
        # fixee_qvalues.shape == (T, B)
        individual_advantages = individual_qvalues - individual_vvalues
        # individual_advantages.shape == (T, B)

        if self.detach_advantages:
            individual_advantages = jax.lax.stop_gradient(individual_advantages)

        if self.debug_recover_fixee:
            w = 0
            b = 0
        else:
            w = self.w_module(states, joint_action_n_hot)
            # w.shape == (T, B, N)
            b = self.b_module(states).squeeze(-1)
            # b.shape == (T, B)

        joint_qvalues = (
            fixee_qvalues + jnp.einsum("TBN,NTB->TB", w, individual_advantages) + b
        )
        # joint_qvalues.shape == (T, B)

        return joint_qvalues

    def vvalues(
        self,
        individual_vvalues: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        # individual_vvalues.shape == (N, T, B)
        # states.shape == (T, B, DS)

        fixee_vvalues = individual_vvalues.sum(axis=0)
        # fixee_vvalues.shape == (T, B)

        if self.debug_recover_fixee:
            b = 0
        else:
            b = self.b_module(states).squeeze(-1)
            # b.shape == (T, B)

        joint_vvalues = fixee_vvalues + b
        # joint_qvalues.shape == (T, B)

        return joint_vvalues
