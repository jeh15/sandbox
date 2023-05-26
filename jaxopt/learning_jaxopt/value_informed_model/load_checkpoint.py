from absl import app
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from brax.envs import wrapper
from brax.envs.env import Env
import orbax.checkpoint

import matplotlib.pyplot as plt

import model
import model_utilities
import puck
import custom_wrapper
import visualize_puck as visualizer
import qp


def create_environment(
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Env:
    """Creates an environment from the registry.
    Args:
        episode_length: length of episode
        action_repeat: how many repeated actions to take per environment step
        auto_reset: whether to auto reset the environment after an episode is done
        batch_size: the number of environments to batch together
        **kwargs: keyword argments that get passed to the Env class constructor
    Returns:
        env: an environment
    """
    env = puck.Puck(**kwargs)

    if episode_length is not None:
        env = wrapper.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrapper.VmapWrapper(env, batch_size)
    if auto_reset:
        env = custom_wrapper.AutoResetWrapper(env)

    return env


def init_params(module, input_size, key):
    params = module.init(
        key,
        jnp.ones(input_size),
    )['params']
    return params


def create_train_state(module, params, learning_rate):
    """Creates an initial `TrainState`."""
    tx = optax.adam(
        learning_rate=learning_rate,
    )
    return train_state.TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
    )


def main(argv=None):
    # RNG Key:
    key_seed = 42

    # Setup Gym Environment:
    num_envs = 1
    max_episode_length = 100
    env = create_environment(
        episode_length=max_episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Create Base Model to load parameters to:
    initial_key = jax.random.PRNGKey(key_seed)
    network = model.ActorCriticNetworkVmap(
        action_space=env.num_actions,
        time_horizon=0.1,
        nodes=3,
    )
    initial_params = init_params(
        module=network,
        input_size=(num_envs, env.observation_size),
        key=initial_key,
    )
    learning_rate = 0.0
    base_state = create_train_state(
        module=network,
        params=initial_params,
        learning_rate=learning_rate,
    )
    del initial_params

    target = {'model': base_state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state_restored = orbax_checkpointer.restore('./checkpoints', item=target)['model']

    state_history = []
    key, env_key = jax.random.split(initial_key)
    states = reset_fn(env_key)
    state_history.append(states)
    position_trajectory = []
    velocity_trajectory = []
    acceleration_trajectory = []
    for _ in range(max_episode_length):
        key, env_key = jax.random.split(env_key)
        mean, std, values, qp_output = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                state_restored.params,
                state_restored.apply_fn,
                states.obs,
            )
        )
        # Make sure the QP Layer is solving:
        trajectory, objective_value, status = qp_output
        assert (status.status).any()
        actions, _, _ = jax.lax.stop_gradient(
            model_utilities.select_action(
                mean,
                std,
                env_key,
            )
        )
        next_states = jax.lax.stop_gradient(
            step_fn(
                states,
                actions,
                env_key,
            )
        )

        # Save Trajectory and Control Input:
        position_trajectory.append(np.squeeze(states.obs)[0])
        velocity_trajectory.append(np.squeeze(states.obs)[-1])
        acceleration_trajectory.append(np.squeeze(actions))

        states = next_states
        state_history.append(states)

    # # Simulation Playback:
    # visualize_batches = num_envs
    # visualizer.generate_batch_video(
    #     env=env,
    #     states=state_history,
    #     batch_size=visualize_batches,
    #     name=f'./videos/puck_simulation'
    # )

    # Trajectory Plotting:
    position_trajectory = np.asarray(position_trajectory)
    velocity_trajectory = np.asarray(velocity_trajectory)
    acceleration_trajectory = np.asarray(acceleration_trajectory)

    # Check Objective Value:
    objective_value = qp.objective_function(
        q=np.concatenate([position_trajectory, velocity_trajectory, acceleration_trajectory], axis=0),
        target_position=1.0,
    )

    print(f'Objective Value: {np.sum(objective_value)}')

    # Create plot handles for visualization:
    fig, axs = plt.subplots(3, 1)
    lb, ub = -5, 5
    time_vector = np.linspace(0, max_episode_length*env.dt, max_episode_length)

    # Position Plot:
    axs[0].plot(time_vector, position_trajectory, color='royalblue')
    axs[0].set_ylabel('Position')
    axs[0].set_ylim([lb, ub])
    axs[0].set_xlim([0, max_episode_length*env.dt])
    axs[0].hlines(1, time_vector[0], time_vector[-1], colors='black', linewidth=0.75, linestyles='--')

    # Velocity Plot:
    axs[1].plot(time_vector, velocity_trajectory, color='limegreen')
    axs[1].set_ylabel('Velocity')
    axs[1].set_ylim([lb, ub])
    axs[1].set_xlim([0, max_episode_length*env.dt])

    # Acceleration Plot:
    axs[2].plot(time_vector, acceleration_trajectory, color='darkorange')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Acceleration')
    axs[2].set_ylim([lb, ub])
    axs[2].set_xlim([0, max_episode_length*env.dt])

    # Save Plot:
    fig.canvas.draw()
    plt.savefig('./figures/puck_trajectory_25.png', dpi=300)


if __name__ == '__main__':
    app.run(main)
