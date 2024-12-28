import flax.linen as nn
import jax


class VDN(nn.Module):
    @nn.compact
    def __call__(self, individual_qvalues: jax.Array, states: jax.Array):
        # qvalues.shape == (N, T, B)
        return individual_qvalues.sum(axis=0)
