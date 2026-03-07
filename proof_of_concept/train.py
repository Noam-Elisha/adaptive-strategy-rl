"""Main training script for the strategy-conditioned PPO agent.

Usage:
    python -m proof_of_concept.train [--total_timesteps N] [--n_envs N]
"""

import argparse
import os
import sys
import time

import gymnasium
import numpy as np
import torch

from .config import PPOConfig
from .env_wrapper import StrategyWrapper
from .ppo import PPOAgent
from .utils import CSVLogger, Timer


def make_envs(config: PPOConfig, fixed_strategy=None):
    """Create a SyncVectorEnv with StrategyWrapper on each sub-env."""
    def _make(idx):
        def _thunk():
            env = gymnasium.make(config.env_id)
            env = StrategyWrapper(env, n_strategies=config.n_strategies,
                                  fixed_strategy=fixed_strategy)
            return env
        return _thunk
    return gymnasium.vector.SyncVectorEnv([_make(i) for i in range(config.n_envs)])


def evaluate(agent: PPOAgent, config: PPOConfig, device: torch.device):
    """Run evaluation episodes for each pure strategy.

    Returns a dict  strategy_name -> {mean_reward, mean_length, mean_fuel,
                                      mean_landing_x, success_rate}
    """
    results = {}
    for s_idx, s_name in enumerate(config.strategy_names):
        strategy_vec = np.zeros(config.n_strategies, dtype=np.float32)
        strategy_vec[s_idx] = 1.0

        # Single env for eval
        env = gymnasium.make(config.env_id)
        env = StrategyWrapper(env, n_strategies=config.n_strategies,
                              fixed_strategy=strategy_vec)

        ep_rewards, ep_lengths, ep_fuels, ep_xs, ep_landed = [], [], [], [], []
        for _ in range(config.eval_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = agent.model(obs_t)
                action = logits.argmax(dim=-1).item()  # greedy
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            ep_rewards.append(total_reward)
            ep_lengths.append(info.get("episode_steps", 0))
            ep_fuels.append(info.get("episode_fuel", 0.0))
            ep_xs.append(abs(info.get("landing_x", 0.0)))
            ep_landed.append(float(info.get("landed", False)))
        env.close()

        results[s_name] = {
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_length": float(np.mean(ep_lengths)),
            "mean_fuel": float(np.mean(ep_fuels)),
            "mean_landing_x": float(np.mean(ep_xs)),
            "success_rate": float(np.mean(ep_landed)),
        }
    return results


def train(config: PPOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Directories
    log_dir = os.path.join(config.results_dir, "logs")
    ckpt_dir = os.path.join(config.results_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Environments
    envs = make_envs(config)
    obs_dim = envs.single_observation_space.shape[0]  # env_obs + strategy
    act_dim = envs.single_action_space.n
    print(f"Obs dim: {obs_dim}  Act dim: {act_dim}  Envs: {config.n_envs}")

    # Agent
    agent = PPOAgent(obs_dim, act_dim, config, device)
    total_params = sum(p.numel() for p in agent.model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loggers
    train_logger = CSVLogger(
        os.path.join(log_dir, "training.csv"),
        ["timestep", "wall_time", "mean_reward", "mean_length",
         "policy_loss", "value_loss", "entropy", "fps"],
    )
    eval_logger = CSVLogger(
        os.path.join(log_dir, "evaluation.csv"),
        ["timestep", "strategy", "mean_reward", "mean_length",
         "mean_fuel", "mean_landing_x", "success_rate"],
    )

    # ---- Training loop ----
    obs, _ = envs.reset()
    timer = Timer()
    global_step = 0
    ep_rewards_buffer = []  # recent episode rewards for logging
    ep_lengths_buffer = []

    print(f"\nTraining for {config.total_timesteps:,} timesteps...\n")

    while global_step < config.total_timesteps:
        agent.buffer.reset()

        # -- Collect rollout --
        for step in range(config.n_steps):
            actions, log_probs, values = agent.get_action(obs)

            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            dones = np.logical_or(terminated, truncated)

            # Handle truncation: bootstrap value into reward
            if np.any(truncated):
                for i in range(config.n_envs):
                    if truncated[i] and "final_observation" in infos:
                        final_obs = infos["final_observation"][i]
                        if final_obs is not None:
                            boot_val = agent.get_value(final_obs[np.newaxis])[0]
                            rewards[i] += config.gamma * boot_val

            # Track completed episodes
            if "_final_info" in infos:
                for i in range(config.n_envs):
                    if infos["_final_info"][i]:
                        fi = infos["final_info"][i]
                        if "episode_steps" in fi:
                            ep_lengths_buffer.append(fi["episode_steps"])
                        # Use the accumulated reward (approximation from done flag)

            agent.buffer.store(
                torch.as_tensor(obs, dtype=torch.float32, device=device),
                torch.as_tensor(actions, dtype=torch.long, device=device),
                torch.as_tensor(rewards, dtype=torch.float32, device=device),
                torch.as_tensor(dones, dtype=torch.float32, device=device),
                torch.as_tensor(log_probs, dtype=torch.float32, device=device),
                torch.as_tensor(values, dtype=torch.float32, device=device),
            )
            obs = next_obs
            global_step += config.n_envs

        # -- Compute GAE --
        next_value = torch.as_tensor(agent.get_value(obs),
                                     dtype=torch.float32, device=device)
        agent.buffer.compute_gae(next_value, config.gamma, config.gae_lambda)

        # -- PPO update --
        losses = agent.update()

        # -- Logging --
        rollout_rewards = agent.buffer.rewards.sum(dim=0).mean().item()
        ep_rewards_buffer.append(rollout_rewards)

        elapsed = timer.elapsed()
        fps = global_step / max(elapsed, 1e-6)

        if global_step % config.log_interval < config.n_envs * config.n_steps:
            mean_r = np.mean(ep_rewards_buffer[-20:]) if ep_rewards_buffer else 0.0
            mean_l = np.mean(ep_lengths_buffer[-50:]) if ep_lengths_buffer else 0.0
            print(f"Step {global_step:>9,} | "
                  f"Reward {mean_r:>8.1f} | "
                  f"Length {mean_l:>6.0f} | "
                  f"PG {losses['policy_loss']:.4f} | "
                  f"VL {losses['value_loss']:.4f} | "
                  f"Ent {losses['entropy']:.4f} | "
                  f"FPS {fps:.0f}")
            train_logger.log({
                "timestep": global_step,
                "wall_time": f"{elapsed:.1f}",
                "mean_reward": f"{mean_r:.2f}",
                "mean_length": f"{mean_l:.1f}",
                "policy_loss": f"{losses['policy_loss']:.6f}",
                "value_loss": f"{losses['value_loss']:.6f}",
                "entropy": f"{losses['entropy']:.6f}",
                "fps": f"{fps:.0f}",
            })

        # -- Evaluation --
        if global_step % config.eval_interval < config.n_envs * config.n_steps:
            eval_results = evaluate(agent, config, device)
            for s_name, metrics in eval_results.items():
                print(f"  EVAL [{s_name:>15s}] "
                      f"R={metrics['mean_reward']:>7.1f}  "
                      f"L={metrics['mean_length']:>5.0f}  "
                      f"Fuel={metrics['mean_fuel']:>5.1f}  "
                      f"|x|={metrics['mean_landing_x']:.3f}  "
                      f"Land={metrics['success_rate']:.0%}")
                eval_logger.log({
                    "timestep": global_step,
                    "strategy": s_name,
                    "mean_reward": f"{metrics['mean_reward']:.2f}",
                    "mean_length": f"{metrics['mean_length']:.1f}",
                    "mean_fuel": f"{metrics['mean_fuel']:.2f}",
                    "mean_landing_x": f"{metrics['mean_landing_x']:.4f}",
                    "success_rate": f"{metrics['success_rate']:.4f}",
                })

        # -- Checkpoint --
        if global_step % config.save_interval < config.n_envs * config.n_steps:
            path = os.path.join(ckpt_dir, f"model_{global_step}.pt")
            torch.save(agent.model.state_dict(), path)

    # ---- Final save ----
    final_path = os.path.join(ckpt_dir, "model_final.pt")
    torch.save(agent.model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Total wall time: {timer.elapsed():.1f}s")

    envs.close()
    train_logger.close()
    eval_logger.close()

    return agent


def main():
    parser = argparse.ArgumentParser(description="Train strategy-conditioned PPO")
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=None)
    args = parser.parse_args()

    config = PPOConfig()
    if args.total_timesteps:
        config.total_timesteps = args.total_timesteps
    if args.n_envs:
        config.n_envs = args.n_envs

    train(config)


if __name__ == "__main__":
    main()
