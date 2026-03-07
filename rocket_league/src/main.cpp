// ============================================================================
// Rocket League Strategy Bot — Stage 1: General Training
// ============================================================================
// Trains a 1v1 Rocket League bot using GigaLearnCPP with PPO.
// The strategy-conditioning architecture (StrategyObsBuilder + StrategyReward)
// is already wired in, but Stage 1 uses a single "general" strategy.
//
// Reward design follows best practices from the RLGym-PPO Guide:
//   - Heavy touch reward to learn ball contact first
//   - Velocity rewards for approach and shooting
//   - Moderate goal reward (not too high to avoid noise)
//   - Air, boost, and movement rewards for well-rounded play
// ============================================================================

#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/OBSBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include "StrategyConfig.h"
#include "StrategyObsBuilder.h"
#include "StrategyReward.h"

using namespace GGL;
using namespace RLGC;

// ----------------------------------------------------------------------------
// Build the "general" strategy reward set (Stage 1)
// Following the RLGym-PPO-Guide recommendations for early training
// ----------------------------------------------------------------------------
StrategyRewardRow BuildGeneralRewards() {
	StrategyRewardRow row;
	row.rewards = {
		// ---- Ball contact (primary signal for early learning) ----
		{ new StrongTouchReward(20, 100), 60.f },    // Reward strong hits
		{ new TouchBallReward(),          10.f },     // Any touch is good

		// ---- Approach the ball ----
		{ new VelocityPlayerToBallReward(), 4.f },    // Move toward ball
		{ new FaceBallReward(),             0.5f },   // Orient toward ball

		// ---- Shooting ----
		{ new ZeroSumReward(
		      new VelocityBallToGoalReward(), 1), 2.f }, // Ball toward goal

		// ---- Game events ----
		{ new GoalReward(),   150.f },  // Score (already zero-sum internally)
		{ new ZeroSumReward(
		      new BumpReward(), 0.5f),  20.f },  // Bumps
		{ new ZeroSumReward(
		      new DemoReward(), 0.5f),  80.f },  // Demos

		// ---- Movement & air ----
		{ new AirReward(),    0.25f },  // Maintain aerial capability
		{ new SpeedReward(),  0.3f  },  // Encourage movement

		// ---- Boost management ----
		{ new PickupBoostReward(), 10.f },  // Collect boost pads
		{ new SaveBoostReward(),   0.2f },  // Don't waste boost
	};
	return row;
}

// ----------------------------------------------------------------------------
// Environment factory — called once per parallel game instance
// ----------------------------------------------------------------------------
EnvCreateResult EnvCreateFunc(int index) {
	// Build strategy-conditioned reward
	// Stage 1: single row, strategy vector = [1.0]
	std::vector<StrategyRewardRow> rewardRows = { BuildGeneralRewards() };
	auto strategyVec = Strategy::DefaultVector();

	// Wrap in StrategyReward (returns weighted sum via dot product)
	auto* strategyReward = new StrategyReward(strategyVec, std::move(rewardRows));

	// The StrategyReward goes into the WeightedReward list with weight 1.0
	// (all weighting is handled internally by StrategyReward)
	std::vector<WeightedReward> rewards = {
		{ strategyReward, 1.0f }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(15),   // Reset if no ball touch for 15 seconds
		new GoalScoreCondition()    // Reset on goal
	};

	// Create 1v1 Soccar arena
	auto arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder   = new StrategyObsBuilder(strategyVec);
	result.stateSetter  = new RandomState(true, true, false);  // Random ball speed, car speed, not forced on ground
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;
	result.arena = arena;

	return result;
}

// ----------------------------------------------------------------------------
// Metrics callback — logged to wandb/console every iteration
// ----------------------------------------------------------------------------
void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	// Only compute expensive metrics 25% of steps to preserve performance
	bool doExpensive = (rand() % 4) == 0;

	for (auto& state : states) {
		if (doExpensive) {
			for (auto& player : state.players) {
				// Movement
				report.AddAvg("Player/Speed", player.vel.Length());
				report.AddAvg("Player/In Air", !player.isOnGround);

				// Ball interaction
				report.AddAvg("Player/Ball Touch", player.ballTouchedStep);
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Toward Ball",
				              RS_MAX(0, player.vel.Dot(dirToBall)));

				// Boost
				report.AddAvg("Player/Boost", player.boost);

				// Touch quality
				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}

		// Always track goals (cheap metric)
		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

// ----------------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
	// Initialize RocketSim with collision meshes
	// Meshes must be dumped from Rocket League using RLArenaCollisionDumper
	RocketSim::Init("./collision_meshes");

	// ---- Learner configuration ----
	LearnerConfig cfg = {};

	cfg.deviceType = LearnerDeviceType::GPU_CUDA;  // Will fallback to CPU if CUDA fails

	cfg.tickSkip    = 8;
	cfg.actionDelay = cfg.tickSkip - 1;  // Standard value

	// Parallel environments — tune to your CPU (256 for high-end, 64 for moderate)
	cfg.numGames = 32;  // 32 parallel games for RTX 2060

	cfg.randomSeed = 42;  // Reproducible; set to -1 for random

	// ---- PPO hyperparameters ----
	int tsPerItr = 50'000;
	cfg.ppo.tsPerItr     = tsPerItr;
	cfg.ppo.batchSize    = tsPerItr;
	cfg.ppo.miniBatchSize = 50'000;

	cfg.ppo.epochs       = 2;
	cfg.ppo.entropyScale = 0.035f;  // Normalized entropy scale

	cfg.ppo.gaeGamma  = 0.99f;
	cfg.ppo.gaeLambda = 0.95f;
	cfg.ppo.clipRange  = 0.2f;

	cfg.ppo.policyLR = 1.5e-4f;
	cfg.ppo.criticLR = 1.5e-4f;

	// ---- Network architecture ----
	cfg.ppo.sharedHead.layerSizes = { 256, 256 };
	cfg.ppo.policy.layerSizes     = { 256, 256, 256 };
	cfg.ppo.critic.layerSizes     = { 256, 256, 256 };

	auto optim = ModelOptimType::ADAM;
	cfg.ppo.policy.optimType     = optim;
	cfg.ppo.critic.optimType     = optim;
	cfg.ppo.sharedHead.optimType = optim;

	auto activation = ModelActivationType::RELU;
	cfg.ppo.policy.activationType     = activation;
	cfg.ppo.critic.activationType     = activation;
	cfg.ppo.sharedHead.activationType = activation;

	bool layerNorm = true;
	cfg.ppo.policy.addLayerNorm     = layerNorm;
	cfg.ppo.critic.addLayerNorm     = layerNorm;
	cfg.ppo.sharedHead.addLayerNorm = layerNorm;

	// ---- Logging & checkpoints ----
	cfg.sendMetrics         = false;  // Disable wandb for now (requires Python setup)
	cfg.renderMode          = false;
	cfg.checkpointFolder    = "checkpoints";
	cfg.tsPerSave           = 5'000'000;  // Save every 5M steps
	cfg.addRewardsToMetrics = true;

	// ---- Create and start learner ----
	printf("Creating learner...\n"); fflush(stdout);
	try {
		Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);
		printf("Starting training...\n"); fflush(stdout);
		learner->Start();
	} catch (const std::exception& e) {
		printf("ERROR: %s\n", e.what()); fflush(stdout);
		return 1;
	} catch (...) {
		printf("ERROR: Unknown exception\n"); fflush(stdout);
		return 1;
	}

	return EXIT_SUCCESS;
}
