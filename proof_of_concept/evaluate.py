"""Evaluation script — loads a trained model, runs episodes per strategy,
prints a comparison table, and generates matplotlib plots.

Usage:
    python -m proof_of_concept.evaluate [--checkpoint PATH] [--episodes N]
"""

import argparse
import csv
import os

import gymnasium
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import PPOConfig
from .env_wrapper import StrategyWrapper
from .networks import ActorCritic


STRATEGY_COLORS = {
    "speed": "#e74c3c",
    "fuel_efficiency": "#2ecc71",
    "precision": "#3498db",
}


def run_evaluation(model, config, device, n_episodes=50):
    """Run n_episodes per pure strategy. Returns per-strategy episode data."""
    data = {}
    for s_idx, s_name in enumerate(config.strategy_names):
        strategy_vec = np.zeros(config.n_strategies, dtype=np.float32)
        strategy_vec[s_idx] = 1.0

        env = gymnasium.make(config.env_id)
        env = StrategyWrapper(env, n_strategies=config.n_strategies,
                              fixed_strategy=strategy_vec)

        episodes = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(obs_t)
                action = logits.argmax(dim=-1).item()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            episodes.append({
                "reward": total_reward,
                "length": info.get("episode_steps", 0),
                "fuel": info.get("episode_fuel", 0.0),
                "landing_x": abs(info.get("landing_x", 0.0)),
                "landed": info.get("landed", False),
            })
        env.close()
        data[s_name] = episodes
    return data


def print_table(data):
    """Pretty-print a comparison table."""
    header = f"{'Strategy':>17s} | {'Reward':>8s} | {'Length':>6s} | {'Fuel':>6s} | {'|x|':>6s} | {'Land%':>5s}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, episodes in data.items():
        r = np.mean([e["reward"] for e in episodes])
        l = np.mean([e["length"] for e in episodes])
        f = np.mean([e["fuel"] for e in episodes])
        x = np.mean([e["landing_x"] for e in episodes])
        s = np.mean([e["landed"] for e in episodes])
        print(f"{name:>17s} | {r:>8.1f} | {l:>6.0f} | {f:>6.1f} | {x:>6.3f} | {s:>5.0%}")
    print(sep)


def save_eval_csv(data, path):
    """Save per-episode data to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "strategy", "episode", "reward", "length", "fuel", "landing_x", "landed"
        ])
        writer.writeheader()
        for name, episodes in data.items():
            for i, ep in enumerate(episodes):
                writer.writerow({
                    "strategy": name,
                    "episode": i,
                    "reward": f"{ep['reward']:.2f}",
                    "length": ep["length"],
                    "fuel": f"{ep['fuel']:.2f}",
                    "landing_x": f"{ep['landing_x']:.4f}",
                    "landed": int(ep["landed"]),
                })


def plot_strategy_comparison(data, out_dir):
    """Generate bar-chart comparison across strategies."""
    os.makedirs(out_dir, exist_ok=True)
    names = list(data.keys())
    colors = [STRATEGY_COLORS.get(n, "#95a5a6") for n in names]

    metrics = {
        "Episode Length (lower = faster)": ("length", False),
        "Fuel Used (lower = more efficient)": ("fuel", False),
        "Landing |x| (lower = more precise)": ("landing_x", False),
        "Success Rate": ("landed", True),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("Strategy Comparison — Trained Agent", fontsize=14, y=1.02)

    for ax, (title, (key, higher_better)) in zip(axes, metrics.items()):
        means = [np.mean([e[key] for e in data[n]]) for n in names]
        stds = [np.std([e[key] for e in data[n]]) for n in names]
        bars = ax.bar(names, means, yerr=stds, color=colors, capsize=5,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("")
        # Highlight best bar
        best_idx = int(np.argmax(means) if higher_better else np.argmin(means))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(out_dir, "strategy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_training_curves(log_path, out_dir):
    """Plot training reward & loss curves from the training CSV log."""
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(log_path):
        print(f"No training log found at {log_path}, skipping curve plot.")
        return

    steps, rewards, pg, vl, ent = [], [], [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["timestep"]))
            rewards.append(float(row["mean_reward"]))
            pg.append(float(row["policy_loss"]))
            vl.append(float(row["value_loss"]))
            ent.append(float(row["entropy"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training Curves", fontsize=14, y=1.02)

    axes[0].plot(steps, rewards, color="#2c3e50")
    axes[0].set_xlabel("Timestep")
    axes[0].set_title("Mean Rollout Reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, pg, label="Policy", color="#e74c3c")
    axes[1].plot(steps, vl, label="Value", color="#3498db")
    axes[1].set_xlabel("Timestep")
    axes[1].set_title("Losses")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, ent, color="#2ecc71")
    axes[2].set_xlabel("Timestep")
    axes[2].set_title("Entropy")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_eval_over_training(eval_log_path, config, out_dir):
    """Plot per-strategy evaluation metrics over training timesteps."""
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(eval_log_path):
        print(f"No eval log found at {eval_log_path}, skipping.")
        return

    # Parse CSV
    rows = []
    with open(eval_log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    strategies = config.strategy_names
    metrics_to_plot = [
        ("mean_reward", "Mean Reward"),
        ("mean_length", "Episode Length"),
        ("mean_fuel", "Fuel Used"),
        ("success_rate", "Success Rate"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle("Evaluation Metrics During Training", fontsize=14, y=1.02)

    for ax, (key, title) in zip(axes, metrics_to_plot):
        for s_name in strategies:
            s_rows = [r for r in rows if r["strategy"] == s_name]
            x = [int(r["timestep"]) for r in s_rows]
            y = [float(r[key]) for r in s_rows]
            color = STRATEGY_COLORS.get(s_name, "#95a5a6")
            ax.plot(x, y, label=s_name, color=color, linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "eval_during_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate strategy-conditioned agent")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model .pt file (default: latest in results/poc)")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    config = PPOConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(config.results_dir, "checkpoints", "model_final.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    obs_dim = 8 + config.n_strategies  # LunarLander obs + strategy
    act_dim = 4
    model = ActorCritic(obs_dim, act_dim, config.hidden_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # Run evaluation
    data = run_evaluation(model, config, device, n_episodes=args.episodes)
    print_table(data)

    # Save results
    plot_dir = os.path.join(config.results_dir, "plots")
    save_eval_csv(data, os.path.join(config.results_dir, "eval_results.csv"))

    # Generate all plots
    plot_strategy_comparison(data, plot_dir)
    plot_training_curves(
        os.path.join(config.results_dir, "logs", "training.csv"), plot_dir
    )
    plot_eval_over_training(
        os.path.join(config.results_dir, "logs", "evaluation.csv"), config, plot_dir
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
