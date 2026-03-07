"""Gymnasium wrapper that adds strategy conditioning to any environment.

The wrapper:
1. Extends the observation with a strategy vector.
2. Replaces the default reward with a strategy-weighted combination of
   per-strategy reward signals.

For LunarLander the three strategies are:
    0 - Speed:           land as fast as possible
    1 - Fuel efficiency: minimise engine usage
    2 - Precision:       land as close to the centre pad as possible
"""

import gymnasium
import numpy as np


class StrategyWrapper(gymnasium.Wrapper):
    """Wraps a LunarLander env with strategy-conditioned rewards.

    Args:
        env:             base gymnasium environment
        n_strategies:    number of strategy dimensions
        fixed_strategy:  if set, use this strategy every episode (for eval);
                         otherwise sample a random one-hot on each reset
    """

    def __init__(self, env, n_strategies: int = 3, fixed_strategy=None):
        super().__init__(env)
        self.n_strategies = n_strategies
        self.fixed_strategy = fixed_strategy

        # Extend observation space: [env_obs | strategy_vector]
        low = np.concatenate([env.observation_space.low,
                              np.zeros(n_strategies, dtype=np.float32)])
        high = np.concatenate([env.observation_space.high,
                               np.ones(n_strategies, dtype=np.float32)])
        self.observation_space = gymnasium.spaces.Box(low, high, dtype=np.float32)

        self.strategy = np.zeros(n_strategies, dtype=np.float32)
        self._prev_shaping = None
        # Metrics tracked per episode
        self._episode_fuel = 0.0
        self._episode_steps = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.fixed_strategy is not None:
            self.strategy = np.array(self.fixed_strategy, dtype=np.float32)
        else:
            # Random one-hot
            self.strategy = np.zeros(self.n_strategies, dtype=np.float32)
            self.strategy[np.random.randint(self.n_strategies)] = 1.0

        self._prev_shaping = self._shaping(obs)
        self._episode_fuel = 0.0
        self._episode_steps = 0
        return self._augmented_obs(obs), info

    def step(self, action):
        obs, _default_reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, action, terminated)
        self._episode_steps += 1

        info["strategy"] = self.strategy.copy()
        info["episode_fuel"] = self._episode_fuel
        info["episode_steps"] = self._episode_steps
        if terminated or truncated:
            info["landing_x"] = float(obs[0])
            info["landed"] = bool(terminated and obs[6] >= 0.5 and obs[7] >= 0.5)

        return self._augmented_obs(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @staticmethod
    def _shaping(obs):
        """Potential-based shaping (mirrors LunarLander internals)."""
        x, y, vx, vy, angle, ang_vel, ll, rl = obs
        return (
            -100.0 * np.sqrt(x * x + y * y)
            - 100.0 * np.sqrt(vx * vx + vy * vy)
            - 100.0 * abs(angle)
            + 10.0 * ll
            + 10.0 * rl
        )

    def _compute_reward(self, obs, action, terminated):
        x = float(obs[0])
        ll, rl = float(obs[6]), float(obs[7])

        # Shaping differential
        cur_shaping = self._shaping(obs)
        shaping_delta = (cur_shaping - self._prev_shaping
                         if self._prev_shaping is not None else 0.0)
        self._prev_shaping = cur_shaping

        is_landed = terminated and ll >= 0.5 and rl >= 0.5
        is_crashed = terminated and not is_landed

        # Engine costs
        main = float(action == 2)
        side = float(action in (1, 3))
        self._episode_fuel += main * 1.0 + side * 0.3

        # --- Per-strategy reward signals ---
        rewards = np.zeros(self.n_strategies, dtype=np.float32)

        # Strategy 0: SPEED — land fast, don't worry about fuel
        rewards[0] = shaping_delta - 0.5                         # heavy time penalty
        rewards[0] -= 0.10 * main + 0.01 * side                 # light fuel cost
        if is_landed:
            rewards[0] += 200.0
        elif is_crashed:
            rewards[0] -= 100.0

        # Strategy 1: FUEL EFFICIENCY — conserve fuel above all
        rewards[1] = shaping_delta - 0.05                        # mild time penalty
        rewards[1] -= 0.80 * main + 0.20 * side                 # heavy fuel cost
        if is_landed:
            rewards[1] += 100.0
        elif is_crashed:
            rewards[1] -= 100.0

        # Strategy 2: PRECISION — land on the centre pad
        rewards[2] = shaping_delta - 0.10                        # moderate time
        rewards[2] -= 0.30 * main + 0.03 * side
        rewards[2] -= 0.50 * abs(x)                              # x-distance penalty
        if is_landed:
            rewards[2] += 100.0 * (1.0 + max(0.0, 1.0 - 5.0 * abs(x)))
        elif is_crashed:
            rewards[2] -= 100.0

        return float(np.dot(self.strategy, rewards))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _augmented_obs(self, obs):
        return np.concatenate([obs, self.strategy]).astype(np.float32)
