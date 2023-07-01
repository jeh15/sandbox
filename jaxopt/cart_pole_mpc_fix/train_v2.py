import os
from absl import app
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import brax
from brax.envs.wrappers import training as wrapper
from brax.envs.base import Env

import large_model as model
import model_utilities_v2 as model_utilities
import cartpole
import custom_wrapper
import visualize_cartpole as visualizer

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64


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
        jax.random.split(key, input_size[0]),
    )['params']
    return params


@optax.inject_hyperparams
def optimizer(learning_rate):
    return optax.chain(
        optax.amsgrad(
            learning_rate=learning_rate,
        ),
    )


def create_train_state(module, params, optimizer):
    return train_state.TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optimizer,
    )


def main(argv=None):
    # RNG Key:
    key_seed = 42

    best_reward = np.NINF
    best_iteration = 0

    # Create Environment:
    episode_length = 200
    num_envs = 128
    env = create_environment(
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
        backend='generalized'
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Initize Networks:
    initial_key = jax.random.PRNGKey(key_seed)

    # Vmap Network:
    asset_path = r'assets/cartpole_friction.xml'
    filepath = os.path.join(os.path.dirname(__file__), asset_path)
    pipeline_model = brax.io.mjcf.load(filepath)

    network = model.ActorCriticNetworkVmap(
        action_space=env.num_actions,
        env=pipeline_model,
    )

    initial_params = init_params(
        module=network,
        input_size=(num_envs, 4),
        key=initial_key,
    )

    # Create a train state:
    learning_rate = 1e-3
    end_learning_rate = 1e-6
    schedule = optax.linear_schedule(
        init_value=learning_rate,
        end_value=end_learning_rate,
        transition_steps=1000,
        transition_begin=1000,
    )
    tx = optimizer(learning_rate=schedule)
    model_state = create_train_state(
        module=network,
        params=initial_params,
        optimizer=tx,
    )
    del initial_params

    # Test Environment:
    training_length = 500
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
        keys_episode = []
        for environment_step in range(episode_length):
            # Brax Environment Step:
            key, env_key = jax.random.split(env_key)
            model_key = jax.random.split(env_key, num_envs)
            model_input = states.obs
            mean, std, values = model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                model_input,
                model_key,
            )
            # Unpack QP Output:
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
            model_input_episode.append(model_input)
            keys_episode.append(model_key)
            states = next_states
            state_history.append(states)

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
        keys_episode = jnp.swapaxes(
            jnp.asarray(keys_episode), axis1=1, axis2=0,
        )

        # No Gradient Calculation:
        model_input = states.obs
        model_key = jax.random.split(env_key, num_envs)
        _, _, values = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                model_input,
                model_key,
            ),
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

        # Generate new RNG keys:
        train_step_keys = jax.random.split(
            env_key,
            (num_envs * episode_length),
        )
        train_step_keys = jnp.reshape(
            train_step_keys,
            (num_envs, episode_length, train_step_keys.shape[-1]),
        )

        # Update Function:
        model_state, loss = model_utilities.train_step(
            model_state,
            model_input_episode,
            actions_episode,
            advantage_episode,
            returns_episode,
            log_probability_episode,
            keys_episode,
            num_envs,
            episode_length,
            10,
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

        current_learning_rate = model_state.opt_state.hyperparams['learning_rate']
        print(f'Epoch: {iteration} \t Average Reward: {average_reward} \t Loss: {loss} \t Average Value: {average_value} \t Learning Rate: {current_learning_rate}')

    print(f'The best reward of {best_reward} was achieved at iteration {best_iteration}')

    state_history = []
    states = reset_fn(env_key)
    state_history.append(states)
    for _ in range(episode_length):
        key, env_key = jax.random.split(env_key)
        model_input = states.obs
        mean, std, values = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_state.params,
                model_state.apply_fn,
                model_input,
            )
        )
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

    visualize_batches = 16
    visualizer.generate_batch_video(
        env=env,
        states=state_history,
        batch_size=visualize_batches,
        name=f'./videos/puck_simulation_{iteration}'
    )


if __name__ == '__main__':
    app.run(main)
