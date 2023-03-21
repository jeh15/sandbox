import jax
import jax.numpy as jnp
from jax import random
import distrax


class ModelUtilities():

    def forward_pass(self, actor, critic, actor_params, critic_params, x):
        logits = actor.apply({'params': actor_params}, x)
        value = critic.apply({'params': critic_params}, x)
        return logits, value

    def select_action(self, logits, value):
        probability_distribution = distrax.Categorical(logits=logits)
        actions = probability_distribution.sample()
        log_probability = probability_distribution.log_prob(actions)
        entropy = probability_distribution.entropy()
        return (actions, log_probability, value, entropy)

    def calculate_advantage(self, rewards, value, masks):
        gamma = 0.999
        lam = 0.95  # hyperparameter for GAE

        episode_length = len(rewards)
        gae = 0.0
        advantage = []
        for i in reversed(range(episode_length - 1)):
            error = rewards[i] + gamma * masks[i] * value[i+1] - value[i]
            gae = error + gamma * lam * masks[i] * gae
            advantage.append(gae)

        advantage = jnp.asarray(
            advantage.reverse(),
        )
        return advantage

    def critic_loss(self, advantage):
        loss = jnp.mean(
            jnp.power(advantage, 2)
        )
        return loss

    def actor_loss(self, advantage, log_probability, entropy):
        entropy_coeff = 0.01  # coefficient for the entropy bonus (to encourage exploration)
        loss = (
            -jnp.mean(
                jax.lax.stop_gradient(advantage) * log_probability
            ) - entropy_coeff * jnp.mean(entropy)
        )
        return loss

    def update_actor(self, model_state, loss_fn, advantage, log_probability, entropy):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(model_state.params, advantage, log_probability, entropy)
        model_state = model_state.apply_gradients(grads=grads)
        return model_state, loss

    def update_critic(self, model_state, loss_fn, advantage):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(model_state.params, advantage)
        model_state = model_state.apply_gradients(grads=grads)
        return model_state, loss
