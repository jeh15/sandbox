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

    ----
    q - joint angle
    qd - joint velocity
    qdd - joint acceleration
    ----
    cdof - dofs in com frame (spatial velocity) -- [omega, v]
    cdofd - velocity of dofs in com frame (spatial acceleration) -- [omega_d, v_d]
    """

    # forward scan over tree: accumulate link center of mass acceleration
    def cdd_fn(cdd_parent, cdofd, qd, dof_idx):
        if cdd_parent is None:
            num_roots = len([p for p in sys.link_parents if p == -1])
            cdd_parent = Motion.create(vel=-jnp.tile(sys.gravity, (num_roots, 1)))

        # cdd = cdd[parent] + map-sum(cdofd * qd)
        # a_i = a_(i-1) + sd_i * qd_i
        cdd = cdd_parent.index_sum(dof_idx, jax.vmap(lambda x, y: x * y)(cdofd, qd))

        return cdd

    zero_motion_vector = brax.Motion.create(
        ang=jnp.zeros_like(state.cdofd.ang),
        vel=jnp.zeros_like(state.cdofd.vel),
    )
    zero_joint_velocity = jnp.zeros_like(state.qd)
    # cdd = scan.tree(
    #     sys, cdd_fn, 'ddd', state.cdofd, state.qd, sys.dof_link(depth=True)
    # )
    cdd = scan.tree(
        sys,
        cdd_fn,
        "ddd",
        zero_motion_vector,
        zero_joint_velocity,
        sys.dof_link(depth=True),
    )

    # cfrc_flat = cinr * cdd + cd x (cinr * cd)
    def frc(cinr, cdd, cd):
        return cinr.mul(cdd) + cd.cross(cinr.mul(cd))

    # To make cd zero do same thing as zero_motion_vector
    # cfrc_flat = jax.vmap(frc)(state.cinr, cdd, state.cd)
    cfrc_flat = jax.vmap(frc)(state.cinr, cdd, zero_motion_vector)

    # backward scan up tree: accumulate link center of mass forces
    def cfrc_fn(cfrc_child, cfrc):
        if cfrc_child is not None:
            cfrc += cfrc_child
        return cfrc

    cfrc = scan.tree(sys, cfrc_fn, "l", cfrc_flat, reverse=True)

    # tau = cdof * cfrc[dof_link]
    tau = jax.vmap(lambda x, y: x.dot(y))(state.cdof, cfrc.take(sys.dof_link()))

    return tau

def _cross_product_matrix(
    A: jax.Array,
) -> jnp.ndarray:
    """Calculates the cross product matrix of a vector.

    Args:
      A: a vector

    Returns:
      A_x: the cross product matrix of A
    """
    A_x = jnp.array(
        [
            [0, -A[2], A[1]],
            [A[2], 0, -A[0]],
            [-A[1], A[0], 0],
        ]
    )
    return A_x


def cross_product_matrix(
    A: jax.Array,
) -> jnp.ndarray:
    return jnp.cross(A, jnp.identity(A.shape[0]) * -1)


def calculate_spatial_inertia(
    I_c: jax.Array,
    c_cross: jax.Array,
    m: jax.typing.ArrayLike,
) -> jnp.ndarray:
    up_left = I_c + m * c_cross @ c_cross.T
    up_right = m * c_cross
    down_left = m * c_cross.T
    down_right = m * jnp.identity(3)
    return jnp.block([[up_left, up_right], [down_left, down_right]])


def calculate_gravity_forces(
    sys: System,
    state: State,
) -> jnp.ndarray:
    # Forward Scan:
    def gravity_force(
        motion_subspace: Motion,
        spatial_ineria: jax.Array,
        gravity_vector: jax.Array,
    ) -> jnp.ndarray:
        # g = S.T @ I_o @ spatial_gravity_vector
        spatial_acceleration = jnp.concatenate(
            [jnp.zeros_like(gravity_vector), -gravity_vector],
        )
        return motion_subspace.T @ spatial_ineria @ spatial_acceleration
    
    c_cross = jax.vmap(cross_product_matrix, in_axes=0, out_axes=0)(
        state.cinr.transform.pos,
    )

    spatial_inertias = jax.vmap(
        calculate_spatial_inertia, in_axes=(0, 0, 0), out_axes=0,
    )(
        state.cinr.i,
        c_cross,
        state.cinr.mass,
    )
    
    composite_spatial_inertias = jnp.flip(
        jnp.cumsum(
            jnp.flip(spatial_inertias, axis=0),
            axis=0,
        ),
        axis=0,
    )
    
    motion_subspace_fn = lambda i: jnp.concatenate(
        [state.cdof.ang[i], state.cdof.vel[i]],
    )
    motion_subspaces = jax.vmap(motion_subspace_fn, in_axes=0, out_axes=0)(
        jnp.arange(state.cdof.ang.shape[0]),
    )
    
    generalized_gravity_force = jax.vmap(
        gravity_force, in_axes=(0, 0, None), out_axes=0,
    )(
        motion_subspaces,
        composite_spatial_inertias,
        sys.gravity,
    )

    return generalized_gravity_force

