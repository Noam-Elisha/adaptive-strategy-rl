# Rocket League Strategy Bot — Stage 1: General Training

Stage 1 of the Adaptive Strategy RL project. Trains a 1v1 Rocket League bot using the [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak) framework with PPO.

The strategy-conditioning architecture is wired in from day one (observation builder appends a strategy vector, reward function uses dot-product weighting), but Stage 1 trains a single "general" strategy to learn game fundamentals: driving, jumping, hitting the ball, aerials, scoring, and boost management.

## Architecture

### Strategy Conditioning (same mechanism as the POC)

The bot uses the same strategy-vector conditioning validated in the LunarLander proof of concept:

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

## Building and Running

### Prerequisites

- Windows 10/11
- Visual Studio 2022 Community (MSVC toolchain)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
- Python 3.13 (GigaLearnCPP embeds Python via pybind11 for metrics)
- Rocket League (for dumping collision meshes)

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

CUDA 12.x removed the standalone `nvToolsExt` library and changed some toolchain defaults. Two CMake files inside LibTorch need patching:

**File: `GigaLearnCPP-Leak/GigaLearnCPP/libtorch/share/cmake/Caffe2/public/cuda.cmake`**

1. Replace `enable_language(CUDA)` with a skip message (no `.cu` files to compile):
   ```cmake
   # PATCHED: Skip enable_language(CUDA) since we have no .cu files to compile.
   message(STATUS "Skipping enable_language(CUDA) - no .cu files to compile")
   ```

2. After `find_package(CUDAToolkit REQUIRED)`, add an nvToolsExt compatibility target:
   ```cmake
   if(NOT TARGET CUDA::nvToolsExt)
     message(STATUS "CUDA::nvToolsExt not found, creating header-only NVTX3 compatibility target")
     add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
     set_target_properties(CUDA::nvToolsExt PROPERTIES
       INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIR}")
   endif()
   ```

3. Replace the `try_run` CUDA version check with a skip message:
   ```cmake
   message(STATUS "Caffe2: Skipping CUDA header version check (no CUDA compilation needed)")
   ```

**File: `GigaLearnCPP-Leak/GigaLearnCPP/libtorch/share/cmake/Torch/TorchConfig.cmake`**

In the MSVC CUDA section, replace the `NVTOOLEXT_HOME` block with:
```cmake
if(MSVC)
  set(TORCH_CUDA_LIBRARIES ${CUDA_LIBRARIES})
  list(APPEND TORCH_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIR}")
  find_library(CAFFE2_NVRTC_LIBRARY caffe2_nvrtc PATHS "${TORCH_INSTALL_PREFIX}/lib")
  list(APPEND TORCH_CUDA_LIBRARIES ${CAFFE2_NVRTC_LIBRARY})
```

### Step 4: Dump collision meshes

Download [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper), run it with Rocket League open, and copy the resulting `.cmf` files into:

```
rocket_league/collision_meshes/soccar/
```

You should have 16 `.cmf` files (mesh_0.cmf through mesh_15.cmf).

### Step 5: Download Ninja

Download [Ninja](https://github.com/ninja-build/ninja/releases) and place `ninja.exe` in `rocket_league/ninja_bin/`.

### Step 6: Build

Open a **Developer Command Prompt for VS 2022** (or run `vcvarsall.bat x64` first), then:

```cmd
cd rocket_league
build.bat
```

This configures and builds with Ninja + MSVC. The output binary is `build/RocketLeagueStrategyBot.exe`.

### Step 7: Run training

```cmd
set PYTHONHOME=C:\path\to\your\Python313
cd rocket_league
build\RocketLeagueStrategyBot.exe
```

Find your Python 3.13 install path — if installed via the Windows Store, it's typically:
```
C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_<version>_x64__qbz5n2kfra8p0
```

You can find it with: `python -c "import sys; print(sys.prefix)"`

Training will print per-iteration metrics (reward, entropy, steps/second). Press **Q** to save the current checkpoint and quit. Checkpoints are saved to `checkpoints/` every 5M timesteps automatically.

**Tuning `numGames`:** Edit `src/main.cpp` line 160 to change the number of parallel environments. More games = faster data collection but higher CPU usage. 32 works well for a mid-range system; use 64+ for high-end CPUs.

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
