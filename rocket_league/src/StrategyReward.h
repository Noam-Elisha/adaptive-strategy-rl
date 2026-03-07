#pragma once

#include <RLGymCPP/Rewards/Reward.h>
#include "StrategyConfig.h"
#include <numeric>   // std::inner_product

// ============================================================================
// StrategyReward
// ============================================================================
// Holds a 2-D table of reward functions:
//   strategies[s] = vector of WeightedReward for strategy index s
//
// Each step computes the weighted reward sum for EVERY strategy, then
// returns  dot(strategy_vector, per_strategy_sums).
//
// During Stage 1 there is only one strategy row, so this is equivalent
// to a normal weighted reward sum.  Stage 2 adds rows for new strategies.
// ============================================================================

struct StrategyRewardRow {
	std::vector<RLGC::WeightedReward> rewards;
};

class StrategyReward : public RLGC::Reward {
public:
	// strategyVec: current conditioning vector for this environment
	StrategyReward(std::vector<float> strategyVec,
	               std::vector<StrategyRewardRow> rows)
		: _strategyVec(std::move(strategyVec)),
		  _rows(std::move(rows)) {}

	virtual void Reset(const RLGC::GameState& initialState) override {
		for (auto& row : _rows)
			for (auto& wr : row.rewards)
				wr.reward->Reset(initialState);
	}

	virtual void PreStep(const RLGC::GameState& state) override {
		for (auto& row : _rows)
			for (auto& wr : row.rewards)
				wr.reward->PreStep(state);
	}

	virtual float GetReward(const RLGC::Player& player,
	                         const RLGC::GameState& state,
	                         bool isFinal) override {
		// Compute per-strategy reward sums
		std::vector<float> strategySums(_rows.size(), 0.f);
		for (size_t s = 0; s < _rows.size(); s++) {
			float sum = 0.f;
			for (auto& wr : _rows[s].rewards)
				sum += wr.weight * wr.reward->GetReward(player, state, isFinal);
			strategySums[s] = sum;
		}

		// Dot product with strategy vector
		float result = 0.f;
		size_t n = std::min(_strategyVec.size(), strategySums.size());
		for (size_t i = 0; i < n; i++)
			result += _strategyVec[i] * strategySums[i];

		return result;
	}

	virtual std::string GetName() override {
		return "StrategyReward";
	}

	void SetStrategy(const std::vector<float>& vec) { _strategyVec = vec; }
	const std::vector<float>& GetStrategy() const { return _strategyVec; }

private:
	std::vector<float> _strategyVec;
	std::vector<StrategyRewardRow> _rows;
};
