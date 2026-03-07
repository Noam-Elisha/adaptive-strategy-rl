# Rocket League Strategy Bot — Stage 1: General Training

Stage 1 of the Adaptive Strategy RL project. Trains a 1v1 Rocket League bot using the [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak) framework with PPO.

The strategy-conditioning architecture is wired in from day one (observation builder appends a strategy vector, reward function uses dot-product weighting), but Stage 1 trains a single "general" strategy to learn game fundamentals: driving, jumping, hitting the ball, aerials, scoring, and boost management.

## Architecture

### Strategy Conditioning (same mechanism as the POC)

The bot uses the same strategy-vector conditioning validated in the LunarLander proof of concept:

- **`StrategyObsBuilder`** wraps `AdvancedObs` (109 features) and appends an N-dimensional strategy vector, giving 110 total obs dimensions in Stage 1
- **`StrategyReward`** holds a 2D table of `[strategy × reward_function]` weights. Each step computes per-strategy reward sums, then returns `dot(strategy_vector, strategy_sums)`
- **`StrategyConfig`** defines `N_STRATEGIES = 1` for Stage 1. Stage 2 bumps this and adds reward rows — no restructuring needed

### Neural Network

Actor-critic with shared head, trained via PPO:

| Component | Layer Sizes | Activation | Layer Norm |
|-----------|-------------|------------|------------|
| Shared Head | 256, 256 | ReLU | Yes |
| Policy | 256, 256, 256 | ReLU | Yes |
| Critic | 256, 256, 256 | ReLU | Yes |

Total parameters: **516,443**. Optimizer: Adam.

### Reward Design

Following best practices from the RLGym-PPO community guide. Heavy emphasis on ball contact for early learning:

| Reward | Weight | Purpose |
|--------|--------|---------|
| `StrongTouchReward(20, 100)` | 60.0 | Strong hits (primary signal) |
| `TouchBallReward` | 10.0 | Any ball contact |
| `VelocityPlayerToBallReward` | 4.0 | Move toward ball |
| `FaceBallReward` | 0.5 | Orient toward ball |
| `VelocityBallToGoalReward` (zero-sum) | 2.0 | Ball toward goal |
| `GoalReward` | 150.0 | Score a goal |
| `BumpReward` (zero-sum) | 20.0 | Bumps |
| `DemoReward` (zero-sum) | 80.0 | Demolitions |
| `AirReward` | 0.25 | Aerial capability |
| `SpeedReward` | 0.3 | Encourage movement |
| `PickupBoostReward` | 10.0 | Collect boost pads |
| `SaveBoostReward` | 0.2 | Boost conservation |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Game mode | 1v1 Soccar |
| Parallel environments | 32 |
| Tick skip | 8 (15 actions/second) |
| State setter | `RandomState` (random ball & car speeds) |
| Terminal conditions | No touch for 15s, goal scored |
| Timesteps per iteration | 50,000 |
| Batch size | 50,000 |
| PPO epochs | 2 |
| Learning rate | 1.5e-4 (policy & critic) |
| Entropy scale | 0.035 |
| Clip range | 0.2 |
| GAE gamma/lambda | 0.99 / 0.95 |

## Building

### Prerequisites

- Windows 10/11
- Visual Studio 2022 (MSVC)
- CUDA Toolkit 12.x
- CMake 3.8+
- Ninja build tool

### Setup

1. **Clone with submodules:**
   ```bash
   git clone --recursive https://github.com/Noam-Elisha/adaptive-strategy-rl.git
   cd adaptive-strategy-rl/rocket_league
   ```

2. **Download LibTorch:** Get the C++ CUDA 11.8 pre-built from [pytorch.org](https://pytorch.org/get-started/locally/) and extract to `GigaLearnCPP-Leak/GigaLearnCPP/libtorch/`.

3. **Collision meshes:** Use [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper) to dump `.cmf` files from Rocket League into `collision_meshes/soccar/`.

4. **Build:**
   ```bash
   # Option A: Use build.bat (sets up MSVC environment automatically)
   build.bat

   # Option B: Manual cmake with Ninja
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
   cmake --build build --config RelWithDebInfo -j 8
   ```

### Running

```bash
# Set Python home for embedded pybind11
export PYTHONHOME="/path/to/python3.13"

cd rocket_league
./build/RocketLeagueStrategyBot.exe
```

Training saves checkpoints to `checkpoints/` every 5M timesteps. Press Q to save and quit early.

## Project Structure

```
rocket_league/
├── CMakeLists.txt                 # Build configuration
├── build.bat                      # Windows build automation
├── GigaLearnCPP-Leak/             # Git submodule (framework)
├── src/
│   ├── main.cpp                   # Entry point, env factory, PPO config
│   ├── StrategyConfig.h           # Strategy system constants
│   ├── StrategyObsBuilder.h       # AdvancedObs + strategy vector
│   └── StrategyReward.h           # Per-strategy weighted rewards
├── collision_meshes/soccar/       # RocketSim arena data (gitignored)
└── checkpoints/                   # Model saves (gitignored)
```
