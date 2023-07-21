import jax
import jax.numpy as jnp
import brax
from brax import scan
from brax.base import Motion, System
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

    zero_motion_cdofd = brax.Motion.create(
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
        zero_motion_cdofd,
        zero_joint_velocity,
        sys.dof_link(depth=True),
    )

    # cfrc_flat = cinr * cdd + cd x (cinr * cd)
    def frc(cinr, cdd, cd):
        return cinr.mul(cdd) + cd.cross(cinr.mul(cd))

    # To make cd zero do same thing as zero_motion_vector
    # cfrc_flat = jax.vmap(frc)(state.cinr, cdd, state.cd)
    zero_motion_cd = brax.Motion.create(
        ang=jnp.zeros_like(state.cd.ang),
        vel=jnp.zeros_like(state.cd.vel),
    )
    cfrc_flat = jax.vmap(frc)(state.cinr, cdd, zero_motion_cd)

    # backward scan up tree: accumulate link center of mass forces
    def cfrc_fn(cfrc_child, cfrc):
        if cfrc_child is not None:
            cfrc += cfrc_child
        return cfrc

    cfrc = scan.tree(sys, cfrc_fn, "l", cfrc_flat, reverse=True)

    # tau = cdof * cfrc[dof_link]
    tau = jax.vmap(lambda x, y: x.dot(y))(state.cdof, cfrc.take(sys.dof_link()))

    return tau


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
        # spatial_gravity_vector = promote_to_spatial_vector(-gravity_vector)
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
    
    motion_subspaces = state.cdof.matrix()
    
    generalized_gravity_force = jax.vmap(
        gravity_force, in_axes=(0, 0, None), out_axes=0,
    )(
        motion_subspaces,
        composite_spatial_inertias,
        sys.gravity,
    )

    return generalized_gravity_force


def motion_cross_product(
    v: jax.Array,
    m: jax.Array,
) -> jnp.ndarray:
    vel = jnp.cross(v[:3], m[3:]) + jnp.cross(v[3:], m[:3])
    ang = jnp.cross(v[:3], m[:3])
    return jnp.concatenate([ang, vel])


def force_cross_product(
    v: jax.Array,
    f: jax.Array,
) -> jnp.ndarray:
    vel = jnp.cross(v[:3], f[3:])
    ang = jnp.cross(v[:3], f[:3]) + jnp.cross(v[3:], f[3:])
    return jnp.concatenate([ang, vel])


def calculate_coriolis_forces(
    sys: System,
    state: State,
) -> jnp.ndarray:
    # Zetta Function:
    # cumsum (v_i x S_i * dq_i)
    def _calculate_zetta(
        spatial_velocity: jax.Array,
        motion_subspace: jax.Array,
        joint_velocity: jax.typing.ArrayLike,
    ) -> jnp.ndarray:
    # Calculate: v x S * dq
        return motion_cross_product(spatial_velocity, motion_subspace * joint_velocity)
   
    # Coriolis Force Function:
    # S_i.T reverse_cumsum (v_k x* I_k v_k + I_k zetta_k)
    def _calculate_coriolis_force(
        composite_spatial_velocity: jax.Array,
        motion_subspace: jax.Array,
        composite_spatial_inertia: jax.Array,
        zetta: jax.Array,
    ) -> jnp.ndarray:
        spatial_force = composite_spatial_inertia @ composite_spatial_velocity
        cross_product = force_cross_product(
            composite_spatial_velocity,
            spatial_force,
        )
        zetta_spatial_force = composite_spatial_inertia @ zetta
        coriolis_force = motion_subspace.T @ (cross_product + zetta_spatial_force)
        return coriolis_force

    link_spatial_velocities = state.cd.matrix()
    motion_subspaces = state.cdof.matrix()

    # Calculate Zetta:
    zetta = jnp.cumsum(
        jax.vmap(_calculate_zetta, in_axes=(0, 0, 0), out_axes=0)(
            link_spatial_velocities,
            motion_subspaces,
            state.qd,
        ),
        axis=0,
    )
    
    # Calculate Coriolis Forces:
    # C(q, qd)qd = S_i.T reverse_cumsum (v_k x* I_k v_k + I_k zetta_k)

    # Calculate composite zetta: zetta_k
    composite_zetta = jnp.flip(
        jnp.cumsum(
            jnp.flip(zetta, axis=0),
            axis=0,
        ),
        axis=0,
    )
    
    # Calculate composite spatial inertias: I_k
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
    
    # Calculate composite spatial velocities: v_k
    composite_spatial_velocities = jnp.flip(
        jnp.cumsum(
            jnp.flip(link_spatial_velocities, axis=0),
            axis=0,
        ),
        axis=0,
    )

    coriolis_forces = jax.vmap(
        _calculate_coriolis_force, in_axes=(0, 0, 0, 0), out_axes=0,
    )(
        composite_spatial_velocities,
        motion_subspaces,
        composite_spatial_inertias,
        composite_zetta,
    )

    return coriolis_forces

