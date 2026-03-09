# Rocket League Strategy Bot — Stage 1: General Training

Stage 1 of the Adaptive Strategy RL project. Trains a 1v1 Rocket League bot using the [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak) framework with PPO.

The strategy-conditioning architecture is wired in from day one (observation builder appends a strategy vector, reward function uses dot-product weighting), but Stage 1 trains a single "general" strategy to learn game fundamentals: driving, jumping, hitting the ball, aerials, scoring, and boost management.

## Architecture

### Strategy Conditioning (same mechanism as the POC)

- **`StrategyObsBuilder`** wraps `AdvancedObs` (109 features) and appends an N-dimensional strategy vector, giving 110 total obs dimensions in Stage 1
- **`StrategyReward`** holds a 2D table of `[strategy x reward_function]` weights. Each step computes per-strategy reward sums, then returns `dot(strategy_vector, strategy_sums)`
- **`StrategyConfig`** defines `N_STRATEGIES = 1` for Stage 1. Stage 2 bumps this and adds reward rows — no restructuring needed

### Neural Network

Actor-critic with shared head, trained via PPO:

| Component | Layer Sizes | Activation | Layer Norm |
|-----------|-------------|------------|------------|
| Shared Head | 256, 256 | ReLU | Yes |
| Policy | 256, 256, 256 | ReLU | Yes |
| Critic | 256, 256, 256 | ReLU | Yes |

Total parameters: **516,443**. Optimizer: Adam.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Game mode | 1v1 Soccar |
| Parallel environments | 32 |
| Tick skip | 8 (15 actions/second) |
| State setter | `RandomState` (random ball & car positions/velocities) |
| Terminal conditions | No touch for 15s, goal scored |
| Timesteps per iteration | 50,000 |
| Batch size | 50,000 |
| PPO epochs | 2 |
| Learning rate | 1.5e-4 (policy & critic) |
| Entropy scale | 0.035 |
| Clip range | 0.2 |
| GAE gamma/lambda | 0.99 / 0.95 |

### Reward Design

Rewards are defined in `src/main.cpp` (`BuildGeneralRewards()`) and can be edited from the dashboard. All reward classes are in `src/CustomRewards.h` and the built-in RLGymCPP rewards.

## Building and Running

### Prerequisites

- Windows 10/11
- Visual Studio 2022 Community (MSVC toolchain)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
- Python 3.13 (GigaLearnCPP embeds Python via pybind11 for metrics)
- Python 3.11 (for RLBot test games — separate from the embedded Python)
- [Ninja](https://github.com/ninja-build/ninja/releases) in `ninja_bin/`
- Rocket League (for dumping collision meshes and test games)

### Step 1: Clone with submodules

```bash
git clone --recursive https://github.com/Noam-Elisha/adaptive-strategy-rl.git
cd adaptive-strategy-rl/rocket_league
```

### Step 2: Download LibTorch

Download the **LibTorch C++ (CUDA 11.8)** pre-built from [pytorch.org](https://pytorch.org/get-started/locally/) (select: LibTorch, C++/Java, CUDA 11.8, Windows). Extract the `libtorch/` folder to:

```
rocket_league/GigaLearnCPP-Leak/GigaLearnCPP/libtorch/
```

### Step 3: Patch LibTorch CMake files

CUDA 12.x removed the standalone `nvToolsExt` library. Two CMake files inside LibTorch need patching:

**File: `GigaLearnCPP-Leak/GigaLearnCPP/libtorch/share/cmake/Caffe2/public/cuda.cmake`**

1. Replace `enable_language(CUDA)` with a skip message
2. After `find_package(CUDAToolkit REQUIRED)`, add an nvToolsExt compatibility target
3. Replace the `try_run` CUDA version check with a skip message

**File: `GigaLearnCPP-Leak/GigaLearnCPP/libtorch/share/cmake/Torch/TorchConfig.cmake`**

Replace the `NVTOOLEXT_HOME` block with direct CUDAToolkit include paths.

See the previous version of this README in git history for the exact patch snippets.

### Step 4: Dump collision meshes

Download [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper), run it with Rocket League open, and copy the 16 `.cmf` files into `rocket_league/collision_meshes/soccar/`.

### Step 5: Build

```cmd
cd rocket_league
build.bat
```

Output: `build/RocketLeagueStrategyBot.exe`.

### Step 6: Train

```cmd
cd rocket_league
train.bat
```

This launches the web dashboard at `http://localhost:8050` and opens it in your browser. The dashboard is the primary interface for all training operations — no need to run the exe directly.

## Web Dashboard

The dashboard at `http://localhost:8050` provides:

### Training Control
- **Start/Stop/Kill** training from the browser
- Stop saves the current checkpoint gracefully; Kill terminates immediately
- Real-time training status display

### Live Metrics
Real-time charts updated every iteration:
- Reward, policy entropy, steps/second
- Policy and critic update magnitudes
- Ball touch %, player speed, boost level
- Ball speed, boost usage, goal speed
- Collection/inference/env step/PPO timing breakdowns

### Bot Management
- Create, switch between, and delete bots
- Each bot has independent checkpoints, metrics history, and config
- Per-bot hyperparameter editor (PPO settings, network size, training params)

### Build & Deploy
- **Rebuild** — runs `build.bat` with streaming console output in the dashboard
- **Edit Rewards** — opens `main.cpp` and `CustomRewards.h` in your editor
- **Build for RLBot** — packages the bot for RLBot deployment
- **Test Game** — launches a live 1v1 match in Rocket League with the trained bot

### Notes
- Attach timestamped notes to training points (e.g., "changed reward weights at 5M steps")
- Notes appear as vertical markers on all metric charts
- Persisted across sessions

### Test Game
- Launches the trained bot in a live Rocket League match via RLBot
- Streams output from both the bot exe and RLBot to the dashboard
- Stop button kills all processes (bot, RLBot, Rocket League) and cleans up

## Project Structure

```
rocket_league/
  CMakeLists.txt                 # Build configuration
  build.bat                      # Windows build script (CMake + Ninja + MSVC)
  train.bat                      # Launches dashboard + opens browser
  GigaLearnCPP-Leak/             # Git submodule (ML framework)
    GigaLearnCPP/                # Core: Learner, PPO, models, LibTorch
      RLGymCPP/                  # RL environment, rewards, obs, state setters
    RLBotCPP/                    # RLBot TCP integration for deployment
  src/
    main.cpp                     # Entry point, env factory, reward config
    CustomRewards.h              # Custom reward functions (25+ rewards)
    StrategyConfig.h             # Strategy system constants (N_STRATEGIES)
    StrategyObsBuilder.h         # AdvancedObs + strategy vector
    StrategyReward.h             # Per-strategy weighted reward aggregator
  monitor/                       # Web dashboard
    server.py                    # Backend (metrics, bot mgmt, task runner)
    static/                      # Frontend (Chart.js, real-time updates)
  collision_meshes/soccar/       # RocketSim arena data (gitignored)
  checkpoints/                   # Model saves per bot (gitignored)
  ninja_bin/                     # Ninja build tool
  build/                         # Compiled binaries (gitignored)
```

## Editing Rewards

Rewards are configured in two files:

- **`src/main.cpp`** — `BuildGeneralRewards()` sets which rewards are active and their weights
- **`src/CustomRewards.h`** — defines all custom reward classes

Use the "Edit Rewards" button in the dashboard to open both files, then "Rebuild" to compile changes.
