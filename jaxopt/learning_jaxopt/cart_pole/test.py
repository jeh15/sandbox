from absl import app
from typing import Optional
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from flax.training import train_state
from brax.envs.wrappers import training as wrapper
from brax.envs.base import Env

import model
import model_utilities
import cartpole
import custom_wrapper
import visualize_cartpole as visualizer


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
    env = cartpole.CartPole(**kwargs)

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

    best_reward = np.NINF
    best_iteration = 0

    # Create Environment:
    episode_length = 200
    num_envs = 32
    env = create_environment(
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Initize Networks:
    initial_key = jax.random.PRNGKey(key_seed)

    # Vmap Network:
    time_horizon = 0.1
    nodes = 2
    num_states = 5
    gravity = 9.81
    network = model.ActorCriticNetworkVmap(
        action_space=env.num_actions,
        time_horizon=time_horizon,
        nodes=nodes,
        num_states=num_states,
        mass_cart=env.sys.link.inertia.mass[0],
        mass_pole=env.sys.link.inertia.mass[1],
        length=env.sys.geoms[-1].length[0] / 2,
        gravity=gravity,
    )

    initial_params = init_params(
        module=network,
        input_size=(num_envs, env.observation_size + num_states * nodes),
        key=initial_key,
    )

    # Create a train state:
    learning_rate = 0.001
    model_state = create_train_state(
        module=network,
        params=initial_params,
        learning_rate=learning_rate,
    )
    del initial_params

    # Test Environment:
    training_length = 1000
    order_idx = np.array([0, 2, 1, 3])  # Reorder the state vector to be compatible with QP Layer
    key, env_key = jax.random.split(initial_key)
    for iteration in range(training_length):
        states = reset_fn(env_key)
        state_history = []
        state_history.append(states)
        model_input_episode = []
        states_episode = []
        values_episode = []
        log_probability_episode = []
        actions_episode = []
        rewards_episode = []
        masks_episode = []
        objective_value_history = []
        initial_condition = states.obs[..., order_idx]
        linearization_trajectory = np.repeat(
            a=np.concatenate(
                [initial_condition, np.zeros((num_envs, 1))],
                axis=-1,
            ),
            repeats=nodes,
            axis=0,
        ).reshape((num_envs, -1))
        model_input = jnp.concatenate(
            [initial_condition, linearization_trajectory],
            axis=-1,
        )
        for environment_step in range(episode_length):
            # Brax Environment Step:
            key, env_key = jax.random.split(env_key)
            mean, std, values, qp_output = model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                model_input,
            )
            # Unpack QP Output:
            state_trajectory, objective_value, status = qp_output
            assert status.all()
            actions, log_probability, entropy = model_utilities.select_action(
                mean,
                std,
                env_key,
            )
            next_states = step_fn(
                states,
                actions,
                env_key,
            )
            states_episode.append(states.obs)
            values_episode.append(jnp.squeeze(values))
            log_probability_episode.append(jnp.squeeze(log_probability))
            actions_episode.append(jnp.squeeze(actions))
            rewards_episode.append(jnp.squeeze(states.reward))
            masks_episode.append(jnp.where(states.done == 0, 1.0, 0.0))
            objective_value_history.append(objective_value)
            model_input_episode.append(model_input)
            states = next_states
            state_history.append(states)
            # Update Model Input:
            initial_condition = states.obs[..., order_idx]
            linearization_trajectory = jnp.concatenate(
                [
                    jnp.concatenate([initial_condition, actions], axis=-1),
                    state_trajectory[..., num_states:],
                ],
                axis=-1,
            )
            model_input = jnp.concatenate(
                [initial_condition, linearization_trajectory],
                axis=-1,
            )
            # Break Point

        # Convert to Jax Arrays:
        states_episode = jnp.swapaxes(
            jnp.asarray(states_episode), axis1=1, axis2=0,
        )
        values_episode = jnp.swapaxes(
            jnp.asarray(values_episode), axis1=1, axis2=0,
        )
        log_probability_episode = jnp.swapaxes(
            jnp.asarray(log_probability_episode), axis1=1, axis2=0,
        )
        actions_episode = jnp.swapaxes(
            jnp.asarray(actions_episode), axis1=1, axis2=0,
        )
        rewards_episode = jnp.swapaxes(
            jnp.asarray(rewards_episode), axis1=1, axis2=0,
        )
        masks_episode = jnp.swapaxes(
            jnp.asarray(masks_episode), axis1=1, axis2=0,
        )
        model_input_episode = jnp.swapaxes(
            jnp.asarray(model_input_episode), axis1=1, axis2=0,
        )

        # No Gradient Calculation:
        _, _, values, qp_output = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                model_input,
            ),
        )
        # Make sure the QP Layer is solving:
        trajectory, objective_value, status = qp_output
        objective_value_history.append(objective_value)
        assert status.all()

        objective_value_history = jnp.swapaxes(
            jnp.squeeze(jnp.asarray(objective_value_history)), axis1=1, axis2=0,
        )

        # Calculate Advantage:
        values_episode = jnp.concatenate(
            [values_episode, values],
            axis=1,
        )

        advantage_episode, returns_episode = jax.lax.stop_gradient(
            model_utilities.calculate_advantage(
                rewards_episode,
                values_episode,
                masks_episode,
                episode_length,
            )
        )

        # Update Function:
        model_state, loss = model_utilities.train_step(
            model_state,
            model_input_episode,
            actions_episode,
            advantage_episode,
            returns_episode,
            log_probability_episode,
        )

        average_reward = np.mean(
            np.sum(
                (rewards_episode),
                axis=1,
            ),
        )

        average_value = np.mean(
            np.mean(
                (values_episode),
                axis=1,
            ),
            axis=0
        )

        average_cost = np.mean(
            np.mean(
                (objective_value_history),
                axis=1,
            ),
            axis=0
        )

        if average_reward >= best_reward:
            best_reward = average_reward
            best_iteration = iteration

        if iteration % 25 == 0:
            visualize_batches = 16
            visualizer.generate_batch_video(
                env=env,
                states=state_history,
                batch_size=visualize_batches,
                name=f'cartpole_training_{iteration}'
            )

        print(f'Epoch: {iteration} \t Average Reward: {average_reward} \t Loss: {loss} \t Average Value: {average_value} \t Average Objective Value: {average_cost}')
        # print(f'Average Value: {average_value} \t Average Objective Value: {average_cost}')

    print(f'The best reward of {best_reward} was achieved at iteration {best_iteration}')

    state_history = []
    states = reset_fn(env_key)
    state_history.append(states)
    initial_condition = states.obs[..., order_idx]
    linearization_trajectory = np.repeat(
        a=np.concatenate(
            [initial_condition, np.zeros((num_envs, 1))],
            axis=-1,
        ),
        repeats=nodes,
        axis=0,
    ).reshape((num_envs, -1))
    model_input = jnp.concatenate(
        [initial_condition, linearization_trajectory],
        axis=-1,
    )
    for _ in range(episode_length):
        key, env_key = jax.random.split(env_key)
        mean, std, values, qp_output = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                model_input,
            )
        )
        # Make sure the QP Layer is solving:
        state_trajectory, objective_value, status = qp_output
        actions, _, _ = jax.lax.stop_gradient(
            model_utilities.select_action(
                mean,
                std,
                env_key,
            )
        )
        states = jax.lax.stop_gradient(
            step_fn(
                states,
                actions,
                env_key,
            )
        )
        state_history.append(states)
        initial_condition = states.obs[..., order_idx]
        linearization_trajectory = jnp.concatenate(
            [
                jnp.concatenate([initial_condition, actions], axis=-1),
                state_trajectory[..., num_states:],
            ],
            axis=-1,
        )
        model_input = jnp.concatenate(
            [initial_condition, linearization_trajectory],
            axis=-1,
        )

    visualize_batches = 16
    visualizer.generate_batch_video(
        env=env,
        states=state_history,
        batch_size=visualize_batches,
        name=f'./videos/puck_simulation_{iteration}'
    )


if __name__ == '__main__':
    app.run(main)
