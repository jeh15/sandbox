import numpy as np
import jax
import jax.numpy as jnp
import distrax


def forward_pass(model_params, apply_fn, x):
    logits, values = apply_fn({'params': model_params}, x)
    return logits, values


def select_action(logits, key):
    probability_distribution = distrax.Categorical(logits=logits)
    actions = probability_distribution.sample(seed=key)
    log_probability = probability_distribution.log_prob(actions)
    entropy = probability_distribution.entropy()
    return actions, log_probability, entropy


def calculate_advantage(rewards, values):
    gamma = 0.999
    lam = 0.95
    episode_length = len(rewards)
    gae = 0.0
    advantage = []
    advantage.append(jnp.array([0.0], dtype=jnp.float32))
    for i in reversed(range(episode_length - 1)):
        error = rewards[i] + gamma * values[i+1] - values[i]
        gae = error + gamma * lam * gae
        advantage.append(gae)
    advantage = jnp.array(advantage, dtype=jnp.float32)[::-1]
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
        jnp.power(advantage, 2)
    )
    actor_loss = (
        -jnp.mean(
            advantage * log_probability
        ) - entropy_coeff * jnp.mean(entropy)
    )
    return actor_loss + value_loss


def train_step(
    model_state,
    advantage,
    states,
    key,
):
    gradient_function = jax.value_and_grad(loss_function)
    loss, gradients = gradient_function(
        model_state.params,
        model_state.apply_fn,
        advantage,
        states,
        key,
    )
    model_state = model_state.apply_gradients(grads=gradients)
    return model_state, loss
