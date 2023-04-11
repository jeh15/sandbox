from functools import partial

import numpy as np
import numpy.typing
import jax
import jax.numpy as jnp


@jax.jit
def multiply(qi: jax.typing.ArrayLike, qj: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Multiply two quaternions together where
        qi is on the left and qj is on the right.
    """

    qi = jnp.asarray(qi)
    _qi = jnp.expand_dims(qi[..., 0], axis=-1)
    qj = jnp.asarray(qj)
    _qj = jnp.expand_dims(qj[..., 0], axis=-1)

    a = (
        qi[..., 0] * qj[..., 0] - jnp.sum(
            qi[..., 1:] * qj[..., 1:], axis=-1
        )
    )

    b = (
        _qi * qj[..., 1:]
        + _qj * qi[..., 1:]
        + jnp.cross(qi[..., 1:], qj[..., 1:])
    )

    q = jnp.concatenate(
        (jnp.expand_dims(a, axis=-1), b),
        axis=-1,
    )

    return q


@jax.jit
def exp(q: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Compute the exponential of quaternion q.
    """

    q = jnp.asarray(q)

    # Create output array:
    q_exp = jnp.zeros_like(q)
    norms = jnp.linalg.norm(
        q[..., 1:],
        axis=-1,
    )
    _norms = jnp.expand_dims(norms, axis=-1)
    e = jnp.exp(q[..., 0])
    _e = jnp.expand_dims(e, axis=-1)

    """
        Break calculation into subcomponents:
            exp(q) = exp(a) * (cos(||q||) + (q / ||q||) * sin(||q||))
            a = exp(a) * cos(||q||)
            b = exp(a) * (q / ||q||) * sin(||q||)
    """
    a = e * jnp.cos(norms)
    b = (
        _e
        * jnp.where(_norms == 0, 0, q[..., 1:] / _norms)
        * jnp.sin(_norms)
    )
    q_exp = q_exp.at[..., 0].set(a)
    q_exp = q_exp.at[..., 1:].set(b)

    return q_exp


@partial(jax.jit, static_argnames=['dt'])
def integrate(q: jax.typing.ArrayLike, v: jax.typing.ArrayLike, dt: float) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/calculus/__init__.py
    q = jnp.asarray(q)
    v = jnp.asarray(v)
    dt = jnp.asarray(dt)
    return multiply(exp(_jax_promote_vector(0.5 * v * dt)), q)


@jax.jit
def conjugate(q: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Compute the conjugate of quaternion q.
    """
    q = jnp.asarray(q)
    q = (
        q.at[..., 1:]
        .set(-1 * q[..., 1:])
    )
    return q


@jax.jit
def rotate(q: jax.typing.ArrayLike, v: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Rotate vectors v by quaternions q.
    """
    q = jnp.asarray(q)
    v = jnp.asarray(v)
    _v = _jax_promote_vector(v)
    return multiply(q, multiply(_v, conjugate(q)))[..., 1:]


@jax.jit
def jax_normalize(q: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Normalize quaternions q such that the
        first element is the identity element.
    """
    q = jnp.asarray(q)
    norms = jnp.expand_dims(
        jnp.linalg.norm(q, axis=-1),
        axis=-1,
    )
    return q / norms


def numpy_normalize(q: numpy.typing.ArrayLike) -> np.ndarray:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Normalize quaternions q such that the
        first element is the identity element.
    """
    q = np.asarray(q)
    norms = np.linalg.norm(q, axis=-1)[..., np.newaxis]
    return q / norms


@jax.jit
def _jax_promote_vector(q: numpy.typing.ArrayLike) -> np.ndarray:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Converts 3 element vectors to 4 element vectors
        for compatability with quaternion math operations
    """
    return jnp.concatenate((jnp.zeros(q.shape[:-1] + (1,)), q), axis=-1)


def _numpy_promote_vector(q: numpy.typing.ArrayLike) -> np.ndarray:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Converts 3 element vectors to 4 element vectors
        for compatability with quaternion math operations
    """
    return np.concatenate((np.zeros(q.shape[:-1] + (1,)), q), axis=-1)
