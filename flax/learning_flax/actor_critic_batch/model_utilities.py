import functools

import numpy as np
import jax
import jax.numpy as jnp
import distrax


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def forward_pass(model_params, apply_fn, x):
    logits, values = apply_fn({'params': model_params}, x)
    return logits, values


@jax.jit
def select_action(logits, key):
    probability_distribution = distrax.Categorical(logits=logits)
    actions = probability_distribution.sample(seed=key)
    log_probability = probability_distribution.log_prob(actions)
    entropy = probability_distribution.entropy()
    return actions, log_probability, entropy


@functools.partial(jax.jit, static_argnames=['episode_length'])
def calculate_advantage(rewards, values, mask, episode_length):
    gamma = 0.999
    lam = 0.95
    gae = jnp.zeros((rewards.shape[0], 1), dtype=jnp.float32)
    advantage = []
    advantage.append(jnp.array(rewards[:, -1, :]-values[:, -1, :], dtype=jnp.float32))
    for i in reversed(range(episode_length - 1)):
        error = rewards[:, i, :] + gamma * values[:, i+1, :] * mask[:, i, :] - values[:, i, :]
        gae = error + gamma * lam * mask[:, i, :] * gae
        advantage.append(gae)
    advantage = jnp.array(advantage, dtype=jnp.float32)
    advantage = jnp.reshape(advantage, (-1, episode_length, 1))
    advantage = jnp.flip(advantage, axis=1)
    return advantage


def loss_function(
    model_params,
    apply_fn,
    advantage,
    states,
    key,
):
    entropy_coeff = 0.01
    value_coeff = 0.5
    logits, _ = forward_pass(model_params, apply_fn, states)
    _, log_probability, entropy = select_action(logits, key)
    value_loss = value_coeff * jnp.mean(
        jnp.square(advantage)
    )
    actor_loss = (
        -jnp.mean(
            jax.lax.stop_gradient(advantage) * log_probability
        ) - entropy_coeff * jnp.mean(entropy)
    )
    return actor_loss + value_loss


@jax.jit
def train_step(
    model_state,
    advantages,
    states,
    keys,
):
    total_loss = 0
    for advantage, state, key in zip(advantages, states, keys):
        gradient_function = jax.value_and_grad(loss_function)
        loss, gradients = gradient_function(
            model_state.params,
            model_state.apply_fn,
            advantage,
            state,
            key,
        )
        total_loss += loss
        model_state = model_state.apply_gradients(grads=gradients)
    return model_state, total_loss
