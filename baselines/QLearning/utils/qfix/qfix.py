import flax.linen as nn
import jax
from .weights import W_Module, B_Module


class QFix(nn.Module):
    """
    QFIX fixing network for projecting IGM-incomplete fixees into IGM-complete values.
    """

    hidden_size: int
    fixee: nn.Module

    w_delta: float
    w_gt: float

    def setup(self):
        self.w_module = W_Module(self.hidden_size, 1, self.w_delta, self.w_gt)
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

        fixee_qvalues = self.fixee(individual_qvalues, states)
        # fixee_qvalues.shape == (T, B)
        fixee_vvalues = self.fixee(individual_vvalues, states)
        # fixee_vvalues.shape == (T, B)
        fixee_advantages = fixee_qvalues - fixee_vvalues
        # fixee_advantages.shape == (T, B)

        w = self.w_module(states, joint_action_n_hot).squeeze(-1)
        # w.shape == (T, B)
        b = self.b_module(states).squeeze(-1)
        # b.shape == (T, B)

        joint_qvalues = w * fixee_advantages + b
        # joint_qvalues.shape == (T, B)

        return joint_qvalues


class AdditiveQFix(nn.Module):
    """
    Q+FIX fixing network for projecting IGM-incomplete fixees into IGM-complete values.
    """

    hidden_size: int
    fixee: nn.Module

    w_delta: float
    w_gt: float
    detach_advantages: bool

    def setup(self):
        self.w_module = W_Module(self.hidden_size, 1, self.w_delta, self.w_gt)
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

        fixee_qvalues = self.fixee(individual_qvalues, states)
        # fixee_qvalues.shape == (T, B)
        fixee_vvalues = self.fixee(individual_vvalues, states)
        # fixee_vvalues.shape == (T, B)
        fixee_advantages = fixee_qvalues - fixee_vvalues
        # fixee_advantages.shape == (T, B)

        if self.detach_advantages:
            fixee_advantages = jax.lax.stop_gradient(fixee_advantages)

        w = self.w_module(states, joint_action_n_hot).squeeze(-1)
        # w.shape == (T, B)
        b = self.b_module(states).squeeze(-1)
        # b.shape == (T, B)

        joint_qvalues = fixee_qvalues + w * fixee_advantages + b
        # joint_qvalues.shape == (T, B)

        return joint_qvalues
