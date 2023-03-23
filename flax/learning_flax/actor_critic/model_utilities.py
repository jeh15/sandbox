import numpy as np
import jax
import jax.numpy as jnp
import distrax


def forward_pass(actor, critic, actor_params, critic_params, x):
    logits = actor.apply({'params': actor_params}, x)
    value = critic.apply({'params': critic_params}, x)
    return logits, value


def select_action(key, logits):
    probability_distribution = distrax.Categorical(logits=logits)
    actions = probability_distribution.sample(seed=key)
    log_probability = probability_distribution.log_prob(actions)
    entropy = probability_distribution.entropy()
    return actions, log_probability, entropy


def calculate_advantage(rewards, value):
    gamma = 0.999
    lam = 0.95  # hyperparameter for GAE

    episode_length = len(rewards)
    gae = 0.0
    advantage = []
    advantage.append(jnp.array(0.0, dtype=jnp.float32))
    for i in reversed(range(episode_length - 1)):
        error = rewards[i] + gamma * value[i+1] - value[i]
        gae = error + gamma * lam * gae
        advantage.append(gae)
    advantage = jnp.array(advantage, dtype=jnp.float32).flatten()
    advantage = jnp.flip(advantage)
    return advantage


def critic_loss_function(advantage):
    loss = jnp.mean(
        jnp.power(advantage, 2)
    )
    return loss


# Why do we not take the advtange into account?
def actor_loss_function(advantage, log_probability, entropy):
    entropy_coeff = 0.01  # coefficient for the entropy bonus (to encourage exploration)
    loss = (
        -jnp.mean(
            jax.lax.stop_gradient(advantage) * log_probability
        ) - entropy_coeff * jnp.mean(entropy)
    )
    return loss


def critic_loss(
        critic_params,
        actor_params,
        actor_network,
        critic_network,
        states,
        rewards,
        key,
):
    episode_length = states.shape[0]
    value_episode = []
    for i in range(episode_length):
        key, subkey = jax.random.split(key)
        logits, values = forward_pass(
            actor=actor_network,
            critic=critic_network,
            actor_params=actor_params,
            critic_params=critic_params,
            x=states[i],
        )
        actions, log_probability, entropy = select_action(key=subkey, logits=logits)
        value_episode.append(values)

    # Convert to Jax Array:
    value_episode = jnp.asarray(value_episode, dtype=jnp.float32).flatten()

    advantage_episode = calculate_advantage(
        rewards=rewards,
        value=value_episode,
    )
    loss = critic_loss_function(
        advantage=advantage_episode,
    )
    return loss


def actor_loss(
        actor_params,
        critic_params,
        actor_network,
        critic_network,
        states,
        rewards,
        key,
):
    episode_length = states.shape[0]
    value_episode = []
    log_probability_episode = []
    entropy_episode = []
    for i in range(episode_length):
        key, subkey = jax.random.split(key)
        logits, values = forward_pass(
            actor=actor_network,
            critic=critic_network,
            actor_params=actor_params,
            critic_params=critic_params,
            x=states[i],
        )
        actions, log_probability, entropy = select_action(key=subkey, logits=logits)
        value_episode.append(values)
        log_probability_episode.append(log_probability)
        entropy_episode.append(entropy)

    # Convert to Jax Array:
    value_episode = jnp.asarray(value_episode, dtype=jnp.float32).flatten()
    log_probability_episode = jnp.asarray(log_probability_episode, dtype=jnp.float32).flatten()
    entropy_episode = jnp.asarray(entropy_episode, dtype=jnp.float32).flatten()

    advantage_episode = calculate_advantage(
        rewards=rewards,
        value=value_episode,
    )
    loss = actor_loss_function(
        advantage=advantage_episode,
        log_probability=log_probability_episode,
        entropy=entropy_episode,
    )
    return loss


# TODO: create single loss function but specific method changes what to take the gradient of.
def update_critic(
        actor_state,
        critic_state,
        actor_network,
        critic_network,
        states,
        rewards,
        key,
):
    grad_fn = jax.value_and_grad(critic_loss)
    loss, grads = grad_fn(
        critic_state.params,
        actor_params=actor_state.params,
        actor_network=actor_network,
        critic_network=critic_network,
        states=states,
        rewards=rewards,
        key=key,
    )
    critic_state = critic_state.apply_gradients(grads=grads)
    return critic_state, loss


def update_actor(
        actor_state,
        critic_state,
        actor_network,
        critic_network,
        states,
        rewards,
        key,
):
    grad_fn = jax.value_and_grad(actor_loss)
    loss, grads = grad_fn(
        actor_state.params,
        critic_params=critic_state.params,
        actor_network=actor_network,
        critic_network=critic_network,
        states=states,
        rewards=rewards,
        key=key,
    )
    actor_state = actor_state.apply_gradients(grads=grads)
    return actor_state, loss
