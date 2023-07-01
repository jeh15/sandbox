from typing import Callable, Tuple
import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
import distrax

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def forward_pass(model_params, apply_fn, x, key):
    # Print Statement:
    print('Compiling Forward Pass...')
    mean, std, values = apply_fn({'params': model_params}, x, key)
    return mean, std, values


@jax.jit
def select_action(mean, std, key):
    probability_distribution = distrax.Normal(loc=mean, scale=std)
    actions = probability_distribution.sample(seed=key)
    log_probability = probability_distribution.log_prob(actions)
    entropy = probability_distribution.entropy()
    return actions, log_probability, entropy


@jax.jit
def evaluate_action(mean, std, action):
    probability_distribution = distrax.Normal(loc=mean, scale=std)
    log_probability = probability_distribution.log_prob(action)
    entropy = probability_distribution.entropy()
    return log_probability, entropy


# Vmap Version:
@functools.partial(jax.jit, static_argnames=['episode_length'])
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None), out_axes=(0, 0))
def calculate_advantage(rewards, values, mask, episode_length):
    gamma = 0.99
    lam = 0.95
    gae = 0.0
    advantage = []
    for i in reversed(range(episode_length)):
        error = rewards[i] + gamma * values[i+1] * mask[i] - values[i]
        gae = error + gamma * lam * mask[i] * gae
        advantage.append(gae)
    advantage = jnp.array(advantage, dtype=dtype)[::-1]
    returns = advantage + values[:-1]
    return advantage, returns


@functools.partial(jax.jit, static_argnames=['batch_size', 'episode_length', 'ppo_steps'])
def train_step(
    model_state,
    model_input,
    actions,
    advantages,
    returns,
    previous_log_probability,
    keys,
    batch_size,
    episode_length,
    ppo_steps,
):
    # Print Statement:
    print('Compiling Train Step...')

    # PPO Optimixation Loop:
    def ppo_loop(carry, xs):
        model_state, keys = carry
        loss, gradients = gradient_function(
            model_state.params,
            model_state.apply_fn,
            model_input,
            actions,
            advantages,
            returns,
            previous_log_probability,
            keys,
        )
        model_state = model_state.apply_gradients(grads=gradients)
        # Generate new RNG keys:
        keys = jax.random.split(
            keys[0, 0],
            (batch_size * episode_length),
        )
        keys = jnp.reshape(
            keys,
            (batch_size, episode_length, keys.shape[-1]),
        )
        # Pack carry and data:
        carry = model_state, keys
        data = loss
        return carry, data

    gradient_function = jax.value_and_grad(loss_function)

    carry, data = jax.lax.scan(
        f=ppo_loop,
        init=(model_state, keys),
        xs=None,
        length=ppo_steps,
    )

    # Unpack carry and data:
    model_state, _ = carry
    loss = data
    loss = jnp.mean(loss)

    return model_state, loss


# @functools.partial(jax.jit, static_argnames=['batch_size', 'episode_length', 'ppo_steps'])
# def train_step(
#     model_state,
#     model_input,
#     actions,
#     advantages,
#     returns,
#     previous_log_probability,
#     keys,
#     batch_size,
#     episode_length,
#     ppo_steps,
# ):
#     # Print Statement:
#     print('Compiling Train Step...')
#     gradient_function = jax.value_and_grad(loss_function)
#     # PPO Optimixation Loop:
#     for ppo_step in range(ppo_steps):
#         loss, gradients = gradient_function(
#             model_state.params,
#             model_state.apply_fn,
#             model_input,
#             actions,
#             advantages,
#             returns,
#             previous_log_probability,
#             keys,
#         )
#         model_state = model_state.apply_gradients(grads=gradients)
#         # Generate new RNG keys:
#         keys = jax.random.split(
#             keys[0, 0],
#             (batch_size * episode_length),
#         )
#         keys = jnp.reshape(
#             keys,
#             (batch_size, episode_length, keys.shape[-1]),
#         )

#     return model_state, loss


# Vmapped Version:
@functools.partial(jax.jit, static_argnames=['apply_fn'])
def loss_function(
    model_params,
    apply_fn,
    model_input,
    actions,
    advantages,
    returns,
    previous_log_probability,
    keys,
):
    # Print Statement:
    print('Compiling Loss Function...')

    # Algorithm Coefficients:
    value_coeff = 0.5
    entropy_coeff = 0.01
    clip_coeff = 0.2

    values, log_probability, entropy = replay(
        model_params,
        apply_fn,
        model_input,
        actions,
        keys,
    )

    # Calculate Ratio: (Should this be No Grad?)
    log_ratios = log_probability - previous_log_probability
    ratios = jnp.exp(log_ratios)

    # Policy Loss:
    unclipped_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(
        1.0 - clip_coeff,
        ratios,
        1.0 + clip_coeff,
    )
    ppo_loss = -jnp.mean(
        jnp.minimum(unclipped_loss, clipped_loss),
    )

    # Value Loss:
    value_loss = value_coeff * jnp.mean(
        jnp.square(values - returns),
    )

    # Entropy Loss:
    entropy_loss = -entropy_coeff * jnp.mean(entropy)

    return ppo_loss + value_loss + entropy_loss


@functools.partial(jax.jit, static_argnames=['apply_fn'])
@functools.partial(jax.vmap, in_axes=(None, None, 1, 1, 1), out_axes=(1, 1, 1))
def replay(
    model_params,
    apply_fn,
    model_input,
    actions,
    keys,
):
    print('Compiling Replay Function...')
    mean, std, values = forward_pass(
        model_params, apply_fn, model_input, keys,
    )
    log_probability, entropy = jax.vmap(evaluate_action)(mean, std, actions)
    return jnp.squeeze(values), jnp.squeeze(log_probability), jnp.squeeze(entropy)
