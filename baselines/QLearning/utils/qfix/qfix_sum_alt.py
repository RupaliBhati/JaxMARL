import flax.linen as nn
import jax
import jax.numpy as jnp
from .weights import W_Module, B_Module


class QFixSumAlt(nn.Module):
    """
    QFIX-sum-alt fixing network for projecting IGM-incomplete fixees into IGM-complete values.
    """

    hidden_size: int
    num_agents: int

    w_delta: float
    w_gt: float

    def setup(self):
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
    ):
        # individual_qvalues.shape == (N, T, B)
        # individual_vvalues.shape == (N, T, B)
        # states.shape == (T, B, DS)
        # joint_action_n_hot.shape == (T, B, N*A), N-hot encoding

        individual_advantages = individual_qvalues - individual_vvalues
        # fixee_advantages.shape == (N, T, B)

        w = self.w_module(states, joint_action_n_hot)
        # w.shape == (T, B, N)
        b = self.b_module(states).squeeze(-1)
        # b.shape == (T, B)

        joint_qvalues = jnp.einsum("TBN,NTB->TB", w, individual_advantages) + b
        # joint_qvalues.shape == (T, B)

        return joint_qvalues


class AdditiveQFixSumAlt(nn.Module):
    """
    Q+FIX-sum-alt fixing network for projecting IGM-incomplete fixees into IGM-complete values.
    """

    hidden_size: int
    num_agents: int

    w_delta: float
    w_gt: float
    detach_advantages: bool

    def setup(self):
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
    ):
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

        w = self.w_module(states, joint_action_n_hot)
        # w.shape == (T, B, N)
        b = self.b_module(states).squeeze(-1)
        # b.shape == (T, B)

        joint_qvalues = (
            fixee_qvalues + jnp.einsum("TBN,NTB->TB", w, individual_advantages) + b
        )
        # joint_qvalues.shape == (T, B)

        return joint_qvalues
