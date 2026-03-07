"""PPO implementation with rollout buffer and GAE.

Designed for vectorised environments — all tensor operations are batched so
they benefit from GPU acceleration when available.
"""

import torch
import torch.nn as nn
import numpy as np

from .networks import ActorCritic


class RolloutBuffer:
    """Fixed-size buffer for one rollout across N parallel environments."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device

        self.obs = torch.zeros((n_steps, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
        self.rewards = torch.zeros((n_steps, n_envs), device=device)
        self.dones = torch.zeros((n_steps, n_envs), device=device)
        self.log_probs = torch.zeros((n_steps, n_envs), device=device)
        self.values = torch.zeros((n_steps, n_envs), device=device)
        self.advantages = torch.zeros((n_steps, n_envs), device=device)
        self.returns = torch.zeros((n_steps, n_envs), device=device)

        self.ptr = 0

    def store(self, obs, actions, rewards, dones, log_probs, values):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.ptr += 1

    def compute_gae(self, next_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Generalised Advantage Estimation (Schulman et al., 2016)."""
        last_gae = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            next_val = next_value if t == self.n_steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def reset(self):
        self.ptr = 0


class PPOAgent:
    """Proximal Policy Optimisation agent."""

    def __init__(self, obs_dim: int, act_dim: int, config, device: torch.device):
        self.config = config
        self.device = device

        self.model = ActorCritic(obs_dim, act_dim, config.hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.buffer = RolloutBuffer(config.n_steps, config.n_envs, obs_dim, device)

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, obs_np: np.ndarray):
        """Convert numpy obs → action, log_prob, value (all numpy)."""
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        action, log_prob, value = self.model.get_action_and_value(obs_t)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def get_value(self, obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        return self.model.get_value(obs_t).cpu().numpy()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self):
        """Run multiple epochs of minibatch PPO on the current buffer.

        Returns a dict of mean losses for logging.
        """
        cfg = self.config
        buf = self.buffer
        batch_size = cfg.n_steps * cfg.n_envs

        # Flatten rollout into a single batch
        b_obs = buf.obs.reshape(batch_size, -1)
        b_actions = buf.actions.reshape(batch_size)
        b_log_probs = buf.log_probs.reshape(batch_size)
        b_advantages = buf.advantages.reshape(batch_size)
        b_returns = buf.returns.reshape(batch_size)

        # Normalise advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(cfg.n_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = indices[start:end]

                new_log_probs, entropy, new_values = self.model.evaluate_actions(
                    b_obs[mb_idx], b_actions[mb_idx]
                )

                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - b_log_probs[mb_idx])
                adv = b_advantages[mb_idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio,
                                    1.0 - cfg.clip_epsilon,
                                    1.0 + cfg.clip_epsilon) * adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                v_loss = nn.functional.mse_loss(new_values, b_returns[mb_idx])

                # Combined loss
                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_pg_loss / n_updates,
            "value_loss": total_v_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
