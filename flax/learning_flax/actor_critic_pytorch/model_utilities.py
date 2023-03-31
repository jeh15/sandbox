import torch


def calculate_advantage(rewards, values, mask, episode_length):
    gamma = 0.99
    lam = 0.95
    gae = 0.0
    advantage = torch.zeros_like(rewards)
    for i in reversed(range(episode_length - 1)):
        error = rewards[i] + gamma * values[i+1] * mask[i] - values[i]
        gae = error + gamma * lam * mask[i] * gae
        advantage[i] = gae
    returns = advantage + values[:-1]
    return advantage, returns


def calculate_loss(advantage, returns, log_probability, entropy):
    # Algorithm Coefficients:
    value_coeff = 0.5
    entropy_coeff = 0.01
    # Mean Squared Error:
    value_loss = value_coeff * advantage.pow(2).mean()

    policy_loss = (
        -(advantage.detach() * log_probability).mean() - entropy_coeff * entropy.mean()
    )
    return policy_loss + value_loss


def update_parameters(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return optimizer
