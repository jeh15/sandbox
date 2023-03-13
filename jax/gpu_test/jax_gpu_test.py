import jax
import jax.numpy as jnp


def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))


def main(argv=None):
    x_small = jnp.arange(3.)
    derivative_fn = jax.grad(sum_logistic)
    print(derivative_fn(x_small))


if __name__ == "__main__":
    main()
