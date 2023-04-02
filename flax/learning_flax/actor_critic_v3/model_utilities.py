import functools

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
    # Forward Pass Network:
    logits, values = forward_pass(model_params, apply_fn, states)
    # Replay actions:
    log_probability, entropy = jax.vmap(evaluate_action)(logits, actions)
    # Calculate Loss:
    value_loss = value_coeff * jnp.mean(
        jnp.square(advantages),
    )
    policy_loss = (
        -jnp.mean(
            jax.lax.stop_gradient(advantages) * log_probability
        ) - entropy_coeff * jnp.mean(entropy)
    )
    return policy_loss + value_loss


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
