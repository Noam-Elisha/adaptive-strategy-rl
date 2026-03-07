"""Hyperparameters and configuration for the strategy-conditioned PPO proof of concept."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    # -- Environment --
    env_id: str = "LunarLander-v3"
    n_envs: int = 16

    # -- Strategy conditioning --
    n_strategies: int = 3
    strategy_names: List[str] = field(
        default_factory=lambda: ["speed", "fuel_efficiency", "precision"]
    )

    # -- Network --
    hidden_dim: int = 128

    # -- PPO --
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # -- Rollout --
    n_steps: int = 128          # steps per env per rollout
    n_epochs: int = 4           # PPO epochs per rollout
    minibatch_size: int = 256

    # -- Training budget --
    total_timesteps: int = 1_500_000

    # -- Logging / evaluation --
    eval_interval: int = 50_000     # timesteps between evaluations
    eval_episodes: int = 20         # episodes per strategy per eval
    log_interval: int = 2_048       # timesteps between console logs
    save_interval: int = 250_000    # timesteps between checkpoints
    results_dir: str = "results/poc"
