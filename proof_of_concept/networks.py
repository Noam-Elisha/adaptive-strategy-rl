"""Actor-Critic network with strategy-vector conditioning.

The observation input is [env_state | strategy_vector], so the network
implicitly learns different policies for different strategy weightings.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init_layer(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """Orthogonal weight init (standard for PPO)."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for discrete action spaces.

    Input dimension = env_obs_dim + n_strategies.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            _init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor = _init_layer(nn.Linear(hidden_dim, act_dim), gain=0.01)
        self.critic = _init_layer(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob, value)."""
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self(obs)
        return value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Return (log_prob, entropy, value) for given obs-action pairs."""
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)
