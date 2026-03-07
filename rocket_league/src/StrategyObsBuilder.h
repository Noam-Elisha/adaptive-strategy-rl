#pragma once

#include <RLGymCPP/OBSBuilders/AdvancedObs.h>
#include "StrategyConfig.h"

// ============================================================================
// StrategyObsBuilder
// ============================================================================
// Wraps AdvancedObs and appends the strategy conditioning vector to each
// observation.  During Stage 1 the vector is always [1.0] (single strategy).
// During Stage 2 each environment can receive a different vector, making
// the policy learn strategy-dependent behaviour.
// ============================================================================

class StrategyObsBuilder : public RLGC::ObsBuilder {
public:
	StrategyObsBuilder(std::vector<float> strategyVec = Strategy::DefaultVector())
		: _strategyVec(std::move(strategyVec)),
		  _inner() {}

	virtual void Reset(const RLGC::GameState& initialState) override {
		_inner.Reset(initialState);
	}

	virtual RLGC::FList BuildObs(const RLGC::Player& player,
	                              const RLGC::GameState& state) override {
		// Get base observation from AdvancedObs
		RLGC::FList obs = _inner.BuildObs(player, state);

		// Append strategy conditioning vector
		for (float v : _strategyVec)
			obs.push_back(v);

		return obs;
	}

	// Allow changing the strategy at runtime (useful for Stage 2)
	void SetStrategy(const std::vector<float>& vec) { _strategyVec = vec; }
	const std::vector<float>& GetStrategy() const { return _strategyVec; }

private:
	std::vector<float> _strategyVec;
	RLGC::AdvancedObs _inner;
};
