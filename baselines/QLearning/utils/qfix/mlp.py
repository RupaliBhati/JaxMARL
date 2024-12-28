import flax.linen as nn
import jax
from flax.linen.initializers import constant, orthogonal


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron with reLU nonlinearities."""

    features: list[int]
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x: jax.Array):
        for i, features in enumerate(self.features):
            if i != 0:
                x = nn.relu(x)

            x = nn.Dense(
                features,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)

        return x
