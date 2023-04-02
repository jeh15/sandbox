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


@jax.jit
def evaluate_action(logits, action):
    probability_distribution = distrax.Categorical(logits=logits)
    log_probability = probability_distribution.log_prob(action)
    entropy = probability_distribution.entropy()
    return log_probability, entropy


# @functools.partial(jax.jit, static_argnames=['episode_length'])
# def calculate_advantage(rewards, values, mask, episode_length):
#     print(f"JIT: Advantage")
#     gamma = 0.99
#     lam = 0.95
#     gae = 0.0
#     advantage = jnp.zeros_like(rewards)
#     for i in reversed(range(episode_length - 1)):
#         error = rewards[i] + gamma * values[i+1] * mask[i] - values[i]
#         gae = error + gamma * lam * mask[i] * gae
#         advantage = advantage.at[i].set(gae)
#     advantage = jnp.array(advantage, dtype=jnp.float32)
#     returns = advantage + values[:-1]
#     return advantage, returns


@functools.partial(jax.jit, static_argnames=['episode_length'])
def calculate_advantage(rewards, values, mask, episode_length):
    gamma = 0.99
    lam = 0.95
    gae = 0.0
    advantage = []
    for i in reversed(range(episode_length)):
        error = rewards[i] + gamma * values[i+1] * mask[i] - values[i]
        gae = error + gamma * lam * mask[i] * gae
        advantage.append(gae)
    advantage = jnp.array(advantage, dtype=jnp.float32)[::-1]
    returns = advantage + values[:-1]
    return advantage, returns


# # OLD:
# def loss_function(
#     model_params,
#     apply_fn,
#     advantage,
#     states,
#     key,
# ):
#     entropy_coeff = 0.01
#     value_coeff = 0.5
#     logits, _ = forward_pass(model_params, apply_fn, states)
#     _, log_probability, entropy = select_action(logits, key)
#     value_loss = value_coeff * jnp.mean(
#         jnp.square(advantage)
#     )
#     policy_loss = (
#         -jnp.mean(
#             jax.lax.stop_gradient(advantage) * log_probability
#         ) - entropy_coeff * jnp.mean(entropy)
#     )
#     return policy_loss + value_loss


# def log_probability_fn(
#     logit,
#     action,
# ):
#     return jax.nn.softmax(logit)[action]


@functools.partial(jax.jit, static_argnames=['apply_fn', 'episode_length'])
def loss_function(
    model_params,
    apply_fn,
    states,
    actions,
    advantages,
    episode_length,
):
    # Algorithm Coefficients:
    value_coeff = 0.5
    entropy_coeff = 0.01
    loss = 0
    for i in range(episode_length):
        logits, values = forward_pass(model_params, apply_fn, states[i])
        log_probability, entropy = evaluate_action(logits, actions[i])
        value_loss = value_coeff * jnp.mean(
            jnp.square(advantages[i]),
        )
        policy_loss = (
            -jnp.mean(
                jax.lax.stop_gradient(advantages[i]) * log_probability
            ) - entropy_coeff * jnp.mean(entropy)
        )
        loss += policy_loss + value_loss
    return loss


# # OLD:
# @functools.partial(jax.jit, static_argnames=['batch_size', 'num_steps'])
# def train_step(
#     model_state,
#     advantages,
#     states,
#     key,
# ):
#     gradient_function = jax.value_and_grad(loss_function)
#     loss, gradients = gradient_function(
#         model_state.params,
#         model_state.apply_fn,
#         advantages,
#         states,
#         key,
#     )
#     model_state = model_state.apply_gradients(grads=gradients)
#     return model_state, loss


@functools.partial(jax.jit, static_argnames=['episode_length'])
def train_step(
    model_state,
    states,
    actions,
    advantages,
    episode_length,
):
    gradient_function = jax.value_and_grad(loss_function)
    loss, gradients = gradient_function(
        model_state.params,
        model_state.apply_fn,
        states,
        actions,
        advantages,
        episode_length,
    )
    model_state = model_state.apply_gradients(grads=gradients)
    return model_state, loss
