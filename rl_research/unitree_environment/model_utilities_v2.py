from typing import Callable, Tuple, Any
import functools

import jax
import jax.numpy as jnp
import flax
import distrax

# jax.config.update("jax_enable_x64", True)
dtype = jnp.float32


@functools.partial(jax.jit, static_argnames=["apply_fn"])
def forward_pass(
    model_params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    x: jax.typing.ArrayLike,
    key: jax.random.PRNGKeyArray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    print("Forward Pass Compiling")
    mean, std, values = apply_fn({"params": model_params}, x, key)
    return mean, std, values


@functools.partial(jax.jit, static_argnames=["multivariate"])
def select_action(
    mean: jax.Array,
    std: jax.Array,
    key: jax.random.PRNGKeyArray,
    multivariate: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if multivariate:
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std,
        )
        actions = probability_distribution.sample(seed=key)
        log_probability = probability_distribution.log_prob(actions)
        entropy = probability_distribution.entropy()
    else:
        probability_distribution = distrax.Normal(
            loc=mean,
            scale=std,
        )
        actions = probability_distribution.sample(seed=key)
        log_probability = jnp.sum(probability_distribution.log_prob(actions), axis=-1)
        entropy = jnp.sum(probability_distribution.entropy(), axis=-1)
    return actions, log_probability, entropy


@functools.partial(jax.jit, static_argnames=["multivariate"])
def evaluate_action(
    mean: jax.Array,
    std: jax.Array,
    action: jax.Array,
    multivariate: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if multivariate:
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std,
        )
        log_probability = probability_distribution.log_prob(action)
        entropy = probability_distribution.entropy()
    else:
        probability_distribution = distrax.Normal(
            loc=mean,
            scale=std,
        )
        log_probability = jnp.sum(probability_distribution.log_prob(action), axis=-1)
        entropy = jnp.sum(probability_distribution.entropy(), axis=-1)
    return log_probability, entropy


@functools.partial(jax.jit, static_argnames=["episode_length"])
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None), out_axes=(0, 0))
def calculate_advantage(
    rewards: jax.typing.ArrayLike,
    values: jax.typing.ArrayLike,
    mask: jax.typing.ArrayLike,
    episode_length: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    gamma = 0.99
    lam = 0.95
    gae = 0.0
    advantage = []
    for i in reversed(range(episode_length)):
        error = rewards[i] + gamma * values[i + 1] * mask[i] - values[i]
        gae = error + gamma * lam * mask[i] * gae
        advantage.append(gae)
    advantage = jnp.array(advantage, dtype=dtype)[::-1]
    returns = advantage + values[:-1]
    return advantage, returns


# @functools.partial(jax.jit, static_argnames=["apply_fn", "episode_length"])
def loss_function(
    model_params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    model_input: jax.typing.ArrayLike,
    actions: jax.typing.ArrayLike,
    advantages: jax.typing.ArrayLike,
    returns: jax.typing.ArrayLike,
    previous_log_probability: jax.typing.ArrayLike,
    keys: jax.random.PRNGKeyArray,
    episode_length: int,
) -> jnp.ndarray:
    # Algorithm Coefficients:
    value_coeff = 0.5
    entropy_coeff = 0.01
    clip_coeff = 0.2

    # # Vmapped Replay:
    # values, log_probability, entropy = replay(
    #     model_params,
    #     apply_fn,
    #     model_input,
    #     actions,
    #     keys,
    # )

    # Serialized Replay:
    model_input = jnp.swapaxes(
        jnp.asarray(model_input), axis1=1, axis2=0,
    )
    actions = jnp.swapaxes(
        jnp.asarray(actions), axis1=1, axis2=0,
    )
    keys = jnp.swapaxes(
        jnp.asarray(keys), axis1=1, axis2=0,
    )

    # values, log_probability, entropy = replay_serial(
    #     model_params,
    #     apply_fn,
    #     model_input,
    #     actions,
    #     keys,
    #     episode_length,
    # )

    values, log_probability, entropy = replay_loop(
        model_params,
        apply_fn,
        model_input,
        actions,
        keys,
        episode_length,
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


# Vmapped Replay Function:
@functools.partial(jax.jit, static_argnames=["apply_fn"])
@functools.partial(jax.vmap, in_axes=(None, None, 1, 1, 1), out_axes=(1, 1, 1))
def replay(
    model_params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    model_input: jax.typing.ArrayLike,
    actions: jax.typing.ArrayLike,
    keys: jax.random.PRNGKeyArray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean, std, values = forward_pass(
        model_params,
        apply_fn,
        model_input,
        keys,
    )
    log_probability, entropy = evaluate_action(mean, std, actions)
    return jnp.squeeze(values), jnp.squeeze(log_probability), jnp.squeeze(entropy)


# Serialized Replay Function:
# @functools.partial(jax.jit, static_argnames=["apply_fn", "episode_length"])
def replay_serial(
    model_params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    model_input: jax.Array,
    actions: jax.Array,
    keys: jax.random.PRNGKeyArray,
    episode_length: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def forward_pass_rollout(
            carry: None,
            xs: Tuple[jax.Array, jax.Array, jax.random.PRNGKeyArray],
    ) -> Tuple[None, Tuple[jax.Array, jax.Array, jax.random.PRNGKeyArray]]:
        model_input, actions, key = xs
        mean, std, values = forward_pass(
            model_params,
            apply_fn,
            model_input,
            key,
        )
        log_probability, entropy = evaluate_action(
            mean,
            std,
            actions,
        )
        carry = None
        data = (jnp.squeeze(values), log_probability, entropy)
        return carry, data

    # Scan over replay:
    _, data = jax.lax.scan(
        forward_pass_rollout,
        None,
        (model_input, actions, keys),
        episode_length,
    )
    values, log_probability, entropy = data
    values = jnp.swapaxes(
        jnp.asarray(values), axis1=1, axis2=0,
    )
    log_probability = jnp.swapaxes(
        jnp.asarray(log_probability), axis1=1, axis2=0,
    )
    entropy = jnp.swapaxes(
        jnp.asarray(entropy), axis1=1, axis2=0,
    )
    return values, log_probability, entropy


def replay_loop(
    model_params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    model_input: jax.Array,
    actions: jax.Array,
    keys: jax.random.PRNGKeyArray,
    episode_length: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    values = []
    log_probabilities = []
    entropies = []
    for i in range(episode_length):
        x = model_input[i]
        key = keys[i]
        action = actions[i]

        mean, std, value = forward_pass(
            model_params,
            apply_fn,
            x,
            key,
        )

        log_probability, entropy = evaluate_action(
            mean,
            std,
            action,
        )

        values.append(value)
        log_probabilities.append(log_probability)
        entropies.append(entropy)

    values = jnp.asarray(values)
    log_probability = jnp.asarray(log_probabilities)
    entropy = jnp.asarray(entropies)

    values = jnp.swapaxes(
        jnp.asarray(values), axis1=1, axis2=0,
    )
    log_probability = jnp.swapaxes(
        jnp.asarray(log_probability), axis1=1, axis2=0,
    )
    entropy = jnp.swapaxes(
        jnp.asarray(entropy), axis1=1, axis2=0,
    )
    return values, log_probability, entropy


# @functools.partial(
#     jax.jit, static_argnames=["batch_size", "episode_length", "ppo_steps"]
# )
def train_step(
    model_state: flax.training.train_state.TrainState,
    model_input: jax.typing.ArrayLike,
    actions: jax.typing.ArrayLike,
    advantages: jax.typing.ArrayLike,
    returns: jax.typing.ArrayLike,
    previous_log_probability: jax.typing.ArrayLike,
    keys: jax.random.PRNGKeyArray,
    batch_size: int,
    episode_length: int,
    ppo_steps: int,
) -> Tuple[flax.training.train_state.TrainState, jnp.ndarray]:
    # PPO Optimixation Loop:
    loss_history = []
    gradient_function = jax.value_and_grad(loss_function)
    for _ in range(ppo_steps):
        loss, gradients = gradient_function(
            model_state.params,
            model_state.apply_fn,
            model_input,
            actions,
            advantages,
            returns,
            previous_log_probability,
            keys,
            episode_length,
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
        loss_history.append(loss)

    loss = jnp.asarray(loss_history)
    loss = jnp.mean(loss)

    return model_state, loss