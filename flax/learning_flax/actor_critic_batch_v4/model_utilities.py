import functools

import numpy as np
import jax
import jax.numpy as jnp
import distrax


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def forward_pass(model_params, apply_fn, x):
    logits, values = apply_fn({'params': model_params}, x)
    return logits, values


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def actor_forward_pass(model_params, apply_fn, x):
    logits = apply_fn({'params': model_params}, x)
    return logits


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def critic_forward_pass(model_params, apply_fn, x):
    values = apply_fn({'params': model_params}, x)
    return values


@jax.jit
def select_action(logits, key):
    probability_distribution = distrax.Categorical(logits=logits)
    actions = probability_distribution.sample(seed=key)
    log_probability = probability_distribution.log_prob(actions)
    entropy = probability_distribution.entropy()
    return actions, log_probability, entropy


# # @jax.jit
# def retrieve_action(logits, actions):
#     log_probability = []
#     entropy = []
#     range_length = logits.shape[0]
#     # test 1:
#     for i in range(range_length):
#         probability_distribution = distrax.Categorical(logits=logits[i])
#         _log_probability = probability_distribution.log_prob(actions[i])
#         _entropy = probability_distribution.entropy()
#         log_probability.append(_log_probability)
#         entropy.append(_entropy)
#     log_probability_1 = jnp.asarray(log_probability)
#     entropy_1 = jnp.asarray(entropy)

#     probability_distribution_2 = distrax.Categorical(logits=logits)
#     log_probability_2 = probability_distribution_2.log_prob(actions)
#     log_probability_2 = jnp.diag(log_probability_2)
#     entropy_2 = probability_distribution_2.entropy()

#     probability_distribution_3 = distrax.Categorical(logits=logits)
#     log_probability_3 = []
#     for i in range(range_length):
#         _log_probability = probability_distribution_3.log_prob(actions[i])[i]
#         log_probability.append(_log_probability)
#     entropy = probability_distribution_2.entropy()
#     log_probability = jnp.asarray(log_probability)

#     return log_probability, entropy


# @jax.jit
# def retrieve_action(logits, actions):
#     log_probability = []
#     entropy = []
#     range_length = logits.shape[0]
#     probability_distribution = distrax.Categorical(logits=logits)
#     for i in range(range_length):
#         _log_probability = probability_distribution.log_prob(actions[i])[i]
#         log_probability.append(_log_probability)
#     entropy = probability_distribution.entropy()
#     log_probability = jnp.asarray(log_probability)

#     return log_probability, entropy


@jax.jit
def retrieve_action(logits, actions):
    probability_distribution = distrax.Categorical(logits=logits)
    log_probability = probability_distribution.log_prob(actions)
    log_probability = jnp.diag(log_probability)
    entropy = probability_distribution.entropy()
    return log_probability, entropy


# @jax.jit
# def retrieve_action(logits, actions):
#     probability_distribution = distrax.Categorical(logits=logits)
#     log_probability = probability_distribution.log_prob(actions)
#     entropy = probability_distribution.entropy()
#     log_probability = jnp.diag(log_probability)
#     return log_probability, entropy


@functools.partial(jax.jit, static_argnames=['episode_length'])
def calculate_advantage(rewards, values, mask, episode_length):
    gamma = 0.99
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
    returns = advantage + values
    return advantage, returns


def loss_function(
    model_params,
    apply_fn,
    advantage,
    returns,
    states,
    actions,
    previous_log_probability,
):
    value_coeff = 0.5
    entropy_coeff = 0.01
    clip_param = 0.1
    logits, values = forward_pass(model_params, apply_fn, states)
    log_probability, entropy = retrieve_action(logits, actions)
    ratios = jnp.exp(log_probability - previous_log_probability)

    value_loss = value_coeff * jnp.mean(
        jnp.square(returns - values)
    )
    entropy_loss = -entropy_coeff * jnp.mean(entropy)

    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    pg_loss = ratios * advantage
    clipped_loss = advantage * jax.lax.clamp(1.0-clip_param, ratios, 1.0+clip_param)
    ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss))

    return ppo_loss + value_loss + entropy_loss


def policy_loss_function(
    model_params,
    apply_fn,
    advantage,
    states,
    key,
):
    entropy_coeff = 0.01
    logits = actor_forward_pass(model_params, apply_fn, states)
    _, log_probability, entropy = select_action(logits, key)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    actor_loss = (
        -jnp.mean(
            jax.lax.stop_gradient(advantage) * log_probability
        ) - entropy_coeff * jnp.mean(entropy)
    )
    return actor_loss


def value_loss_function(
    model_params,
    apply_fn,
    returns,
    states,
):
    values = critic_forward_pass(model_params, apply_fn, states)
    value_loss = jnp.mean(
        jnp.square(returns - values)
    )
    return value_loss


# @functools.partial(jax.jit, static_argnames=['batch_size', 'num_steps'])
# def train_step(
#     policy_model_state,
#     value_model_state,
#     advantages,
#     returns,
#     states,
#     key,
#     batch_size,
#     num_steps,
# ):
#     advantages = jnp.reshape(advantages, (batch_size * num_steps, -1))
#     returns = jnp.reshape(returns, (batch_size * num_steps, -1))
#     states = jnp.reshape(states, (batch_size * num_steps, -1))
#     value_gradient_function = jax.value_and_grad(value_loss_function)
#     value_loss, value_gradients = value_gradient_function(
#         value_model_state.params,
#         value_model_state.apply_fn,
#         returns,
#         states,
#     )
#     value_model_state = value_model_state.apply_gradients(grads=value_gradients)
#     policy_gradient_function = jax.value_and_grad(policy_loss_function)
#     policy_loss, policy_gradients = policy_gradient_function(
#         policy_model_state.params,
#         policy_model_state.apply_fn,
#         advantages,
#         states,
#         key,
#     )
#     policy_model_state = policy_model_state.apply_gradients(grads=policy_gradients)
#     return policy_model_state, value_model_state, policy_loss, value_loss


@functools.partial(jax.jit, static_argnames=['batch_size', 'num_steps'])
def train_step(
    model_state,
    advantages,
    returns,
    states,
    actions,
    previous_log_probability,
    batch_size,
    num_steps,
):
    advantages = jnp.reshape(advantages, (batch_size * num_steps, -1)).flatten()
    returns = jnp.reshape(returns, (batch_size * num_steps, -1)).flatten()
    states = jnp.reshape(states, (batch_size * num_steps, -1))
    actions = jnp.reshape(actions, (batch_size * num_steps, -1))
    previous_log_probability = jnp.reshape(previous_log_probability, (batch_size * num_steps, -1)).flatten()
    # PPO Loss Function:
    gradient_function = jax.value_and_grad(loss_function)
    loss, gradients = gradient_function(
        model_state.params,
        model_state.apply_fn,
        advantages,
        returns,
        states,
        actions,
        previous_log_probability,
    )
    model_state.apply_gradients(grads=gradients)
    return model_state, loss
