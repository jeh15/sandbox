import jax
import jax.numpy as jnp
import brax
from brax import scan
from brax.base import Motion, System, Transform
from brax.generalized.base import State


def calculate_coriolis_matrix(
        sys: System,
        state: State,
) -> jnp.ndarray:
  """Calculates the system's forces given input motions.

  This function computes inverse dynamics using the Newton-Euler algorithm:

  https://scaron.info/robot-locomotion/recursive-newton-euler-algorithm.html

  Args:
    sys: a brax system
    state: generalized state

  Returns:
    tau: generalized forces resulting from joint positions and velocities
  """
  # forward scan over tree: accumulate link center of mass acceleration
  def cdd_fn(cdd_parent, cdofd, qd, dof_idx):
    if cdd_parent is None:
      num_roots = len([p for p in sys.link_parents if p == -1])
      cdd_parent = Motion.create(vel=-jnp.tile(sys.gravity, (num_roots, 1)))

    # cdd = cdd[parent] + map-sum(cdofd * qd)
    cdd = cdd_parent.index_sum(dof_idx, jax.vmap(lambda x, y: x * y)(cdofd, qd))

    return cdd

  cdd = scan.tree(
      sys, cdd_fn, 'ddd', state.cdofd, state.qd, sys.dof_link(depth=True)
  )

  # cfrc_flat = cinr * cdd + cd x (cinr * cd)
  def frc(cinr, cdd, cd):
    return cinr.mul(cdd) + cd.cross(cinr.mul(cd))

  cfrc_flat = jax.vmap(frc)(state.cinr, cdd, state.cd)

  # backward scan up tree: accumulate link center of mass forces
  def cfrc_fn(cfrc_child, cfrc):
    if cfrc_child is not None:
      cfrc += cfrc_child
    return cfrc

  cfrc = scan.tree(sys, cfrc_fn, 'l', cfrc_flat, reverse=True)

  # tau = cdof * cfrc[dof_link]
  tau = jax.vmap(lambda x, y: x.dot(y))(state.cdof, cfrc.take(sys.dof_link()))

  return tau
