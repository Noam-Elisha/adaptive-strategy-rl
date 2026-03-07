# Introduction
Rocket league is a very complex multiplayer game requiring a high level of individual skill, as well as team coordination and overall strategy/planning. Currently the best bots have mastered all of these aspects of the game. The one skill that bots lack is adaptability. They are trained with one strategy in mind and that is the strategy which they play always.

My belief is that it is possible to make a bot that can switch between multiple different strategies. Once we have a bot that is able to change strategy, we can add either a second network to choose the strategy, or use a human "coach" to update the strategy (or mix of strategies) that the bot uses in real time.

I think the way to make this happen is as follows:

## Stage 1 - General Training:
First, the bot needs to learn how to play the game. It needs to learn how to drive, jump, hit the ball, aerial, pass, etc. We can use commonly known methods/tricks in the RLGym community to shape the rewards in this stage to get the bot to a high level naturally. 

Once the bot can play the game on its own and has a decently high skill level in the game comes stage 2.

## Stage 2 - Strategy Training:

In this stage we train the different strategies of the bot. To do this we alter the architecture/algorithm as follows:

### Architecture

To make the bot behave differently with different strategies we append a normalized strategy input vector (perhaps one-hot encoded, or perhaps with float values to mix strategies together) to the observation layer, where the values correspond to the weight of that strategy.

### Rewards

To match the reward function to the strategy training, we feed in the normalized strategy input vector to the reward function and add the dot product of the vector and the rewards from our various different strategy evaluation functions.

As the strategy input vector affects both the observation layer and the reward function, we can train the different strategies in different ways. Either one a time, or mixing with random (or not) values.

# This Project

In this project we will aim to prove that this method of training works on a small scale proof of concept project using smaller, more efficient, popular gym environments in the machine learning community, before then applying this method to an actual Rocket League bot using the GigaLearn (https://github.com/ZealanL/GigaLearnCPP-Leak) framework by ZealanL using the PPO algorithm.

# Proof of Concept Results

## Introduction

To validate the strategy-conditioning approach before applying it to Rocket League, we ran a proof of concept using **LunarLander-v3** from [Gymnasium](https://gymnasium.farama.org/). LunarLander is a popular RL benchmark where an agent must land a spacecraft on a pad. It is simple enough to train in minutes but has enough nuance to support meaningfully different play-styles.

We defined three strategies that a single agent can switch between:

| # | Strategy | Behaviour goal |
|---|----------|----------------|
| 0 | **Speed** | Land as quickly as possible (heavy per-step penalty, light engine cost) |
| 1 | **Fuel Efficiency** | Minimise engine usage (heavy engine cost, light time penalty) |
| 2 | **Precision** | Land as close to the centre pad as possible (x-distance penalty, centre-landing bonus) |

## Methods

**Architecture** — A shared-backbone Actor-Critic MLP (2 hidden layers of 128 units, Tanh activations, orthogonal init). The observation input is the 8-dim LunarLander state concatenated with a 3-dim strategy vector (one-hot or mixed), giving an 11-dim input. Total parameters: 18,693.

**Reward** — Each timestep produces three separate reward signals (one per strategy), each with its own shaping, engine cost, time penalty, and terminal bonuses. The agent receives `dot(strategy_vector, [R_speed, R_fuel, R_precision])`, exactly as described in the project proposal.

**Training** — PPO with GAE, 16 vectorised environments, 128 steps per rollout. Each environment samples a random one-hot strategy on episode reset, so the agent is trained on all strategies simultaneously.

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 3e-4 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip epsilon | 0.2 |
| Value coef | 0.5 |
| Entropy coef | 0.01 |
| Minibatch size | 256 |
| PPO epochs | 4 |
| Total timesteps | 1,500,000 |
| Training time | ~351s (CPU) |

The code is fully vectorised with PyTorch and GPU-ready (auto-detects CUDA). All modules (`env_wrapper`, `networks`, `ppo`, `train`, `evaluate`) are self-contained and reusable.

## Results

### Training Curves

The agent learns to land successfully across all strategies, going from a mean rollout reward of -200 (crashing) to +250 (consistent landings) over 1.5M timesteps:

![Training Curves](results/poc/plots/training_curves.png)

### Strategy Differentiation Over Training

Per-strategy evaluation was run every 50K steps. The "Fuel Used" panel is the clearest signal — the speed strategy (red) consistently burns more fuel than fuel_efficiency (green):

![Eval During Training](results/poc/plots/eval_during_training.png)

### Final Evaluation (50 episodes per strategy, greedy policy)

| Strategy | Mean Reward | Ep. Length | Fuel Used | Landing \|x\| | Success Rate |
|----------|-------------|------------|-----------|---------------|-------------|
| **Speed** | 193.7 | 110 | **63.4** | 0.263 | 72% |
| **Fuel Efficiency** | 169.1 | 102 | **51.2** | 0.228 | **82%** |
| **Precision** | 181.2 | **100** | 50.6 | **0.240** | 76% |

![Strategy Comparison](results/poc/plots/strategy_comparison.png)

**Key observations:**
- The **speed** strategy uses ~24% more fuel than fuel efficiency, confirming it fires engines more aggressively to land faster.
- The **fuel efficiency** strategy achieves the highest success rate (82%) while using the least fuel (51.2), showing that conservative engine usage leads to more controlled landings.
- The **precision** strategy produces the shortest episodes and moderate fuel usage, balancing between speed and efficiency.
- All three strategies are produced by the **same network** — only the strategy input vector changes.

### Conclusion

The proof of concept confirms that **strategy-vector conditioning works**: a single neural network can learn distinct behaviours by appending a strategy vector to the observation and weighting reward signals with the same vector. This validates the core mechanism proposed for the Rocket League bot and gives us confidence to proceed with Stage 1 of Rocket League training.

# Stage 1 Results: Rocket League — General Training

## Introduction

With the POC validated, Stage 1 applies the same strategy-conditioning architecture to a real Rocket League bot. The goal is to train a 1v1 bot that learns game fundamentals — driving, jumping, hitting the ball, aerials, scoring, and boost management — while having the strategy-conditioning wiring already in place for Stage 2.

We use the [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak) framework, which provides high-performance C++ training via [RocketSim](https://github.com/ZealanL/RocketSim) (a physics replica of Rocket League) and PPO with LibTorch.

## Methods

**Architecture** — The strategy-conditioning mechanism is identical to the POC: a strategy vector is appended to observations (`StrategyObsBuilder`), and rewards are computed as `dot(strategy_vector, per_strategy_sums)` (`StrategyReward`). Stage 1 uses a single "general" strategy (`N_STRATEGIES = 1`), so the strategy vector is always `[1.0]`. The actor-critic network has a shared head (256×2) with separate policy (256×3) and critic (256×3) heads, all with ReLU activations and layer normalization. Total parameters: **516,443**.

**Reward design** — Following RLGym-PPO community best practices for early-stage training. The reward function heavily weights ball contact (`StrongTouchReward` at 60.0, `TouchBallReward` at 10.0) to bootstrap learning, with supporting signals for approach (`VelocityPlayerToBallReward`), shooting (`VelocityBallToGoalReward`, zero-sum), game events (`GoalReward` at 150.0, `BumpReward`, `DemoReward`), and auxiliary skills (`AirReward`, `SpeedReward`, `PickupBoostReward`, `SaveBoostReward`).

**Training** — PPO with GAE, 32 parallel 1v1 Soccar environments on GPU (CUDA). Each environment uses `RandomState` (randomized ball and car velocities) for diverse starting conditions, with episode resets on goal or 15 seconds without a ball touch.

| Hyperparameter | Value |
|----------------|-------|
| Parallel environments | 32 |
| Tick skip | 8 (15 actions/second) |
| Timesteps per iteration | 50,000 |
| Batch size | 50,000 |
| PPO epochs | 2 |
| Learning rate | 1.5e-4 (policy & critic) |
| Entropy scale | 0.035 |
| Clip range | 0.2 |
| GAE gamma / lambda | 0.99 / 0.95 |
| Throughput | ~38,000–48,000 steps/second |

**Note on GPU usage:** GigaLearnCPP runs the RocketSim physics simulation on CPU and only uses the GPU for neural network inference and PPO gradient updates. With a 516K parameter model this is minimal GPU work — the bottleneck is CPU-side simulation.

## Results

The agent trained for **32M timesteps** (~631 iterations, ~14 minutes). Average step reward increased **4.6x** over training, from 0.28 (random behaviour) to 1.30:

| Timesteps | Avg Step Reward | Policy Entropy | Phase |
|-----------|----------------|----------------|-------|
| 58K | 0.28 | 0.770 | Random movement |
| 2.5M | 0.33 | 0.773 | Learning to approach ball |
| 5.0M | 0.83 | 0.752 | Consistent ball contact |
| 7.6M | 1.01 | 0.731 | Strong touches, some goals |
| 10.1M | 1.12 | 0.730 | Regular scoring |
| 15.1M | 1.18 | 0.714 | Improving shot quality |
| 20.2M | 1.31 | 0.710 | Plateau, refining play |
| 25.3M | 1.22 | 0.700 | Continued refinement |
| 32.0M | 1.30 | 0.690 | Stable, well-rounded play |

**Key observations:**
- The reward curve shows rapid initial learning (0.28 → 0.83 in the first 5M steps) as the bot learns to approach and hit the ball.
- A second phase of improvement (5M → 15M steps) corresponds to learning strong touches and goal scoring.
- The reward plateaus around 1.2–1.3 after 15M steps, indicating the bot has learned the core skills covered by the reward function.
- Policy entropy decreases steadily from 0.77 to 0.69, showing the agent moving from exploration to more focused action selection.

### Conclusion

Stage 1 confirms that the strategy-conditioning architecture works correctly in the Rocket League setting. The bot learned to drive, approach the ball, make strong touches, collect boost, and score goals — all while the `StrategyObsBuilder` and `StrategyReward` infrastructure is in place. Transitioning to Stage 2 (multiple strategies) requires only bumping `N_STRATEGIES` in `StrategyConfig.h` and adding new reward rows — no architectural changes needed.

## Running the Rocket League Bot

Full build instructions (including LibTorch download, CMake patching, and collision mesh dumping) are in [`rocket_league/README.md`](rocket_league/README.md). Quick-start if already built:

```cmd
:: From the repo root, in a Developer Command Prompt for VS 2022
cd rocket_league

:: Set Python home (required for GigaLearnCPP's embedded pybind11)
:: Find yours with: python -c "import sys; print(sys.prefix)"
set PYTHONHOME=C:\path\to\your\Python313

:: Run training
build\RocketLeagueStrategyBot.exe
```

Training prints per-iteration metrics. Press **Q** to save and quit. Checkpoints are saved to `checkpoints/` every 5M timesteps.

To rebuild after code changes:
```cmd
cd rocket_league
build.bat
```

## Running the Proof of Concept

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
pip install -r requirements.txt
```

This installs PyTorch, Gymnasium with Box2D, NumPy, and Matplotlib. If you have a CUDA-capable GPU, install the CUDA version of PyTorch first for faster training (the code auto-detects GPU availability).

### Train

```bash
python -m proof_of_concept.train
```

Training runs for 1.5M timesteps (~6 minutes on CPU). Evaluation logs are printed every 50K steps. You can override defaults with flags:

```bash
python -m proof_of_concept.train --total_timesteps 500000 --n_envs 8
```

Outputs are saved to `results/poc/`:
- `logs/training.csv` — per-rollout metrics (reward, losses, entropy, FPS)
- `logs/evaluation.csv` — per-strategy eval metrics over training
- `checkpoints/model_final.pt` — trained model weights

### Evaluate

After training, generate the comparison table and plots:

```bash
python -m proof_of_concept.evaluate
```

This loads `results/poc/checkpoints/model_final.pt`, runs 50 greedy episodes per strategy, and saves plots to `results/poc/plots/`. To use a different checkpoint or episode count:

```bash
python -m proof_of_concept.evaluate --checkpoint path/to/model.pt --episodes 100
```