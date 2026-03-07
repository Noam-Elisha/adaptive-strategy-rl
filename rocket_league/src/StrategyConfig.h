#pragma once

#include <vector>
#include <string>

// ============================================================================
// Strategy-conditioning configuration
// ============================================================================
// Stage 1: A single "general" strategy (N_STRATEGIES = 1).
// Stage 2: Bump N_STRATEGIES and add per-strategy reward rows.
//
// The strategy vector is appended to every observation and used to
// dot-product weight per-strategy reward signals — exactly matching
// the mechanism validated in the LunarLander proof of concept.
// ============================================================================

namespace Strategy {

	// --- Change this when adding strategies in Stage 2 ---
	constexpr int N_STRATEGIES = 1;

	// Human-readable names (indices match the vector positions)
	inline const std::vector<std::string>& GetStrategyNames() {
		static const std::vector<std::string> names = {
			"general"        // index 0
			// Stage 2 examples:
			// "aggressive",  // index 1
			// "defensive",   // index 2
			// "aerial",      // index 3
		};
		return names;
	}

	// Build a one-hot strategy vector for a given index
	inline std::vector<float> OneHot(int strategyIdx) {
		std::vector<float> vec(N_STRATEGIES, 0.f);
		if (strategyIdx >= 0 && strategyIdx < N_STRATEGIES)
			vec[strategyIdx] = 1.f;
		return vec;
	}

	// Build the default Stage-1 strategy vector (all weight on "general")
	inline std::vector<float> DefaultVector() {
		return OneHot(0);
	}

} // namespace Strategy
