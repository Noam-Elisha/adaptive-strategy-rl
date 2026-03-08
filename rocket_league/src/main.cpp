// ============================================================================
// Rocket League Strategy Bot — Stage 1: General Training
// ============================================================================
// Trains a Rocket League bot using GigaLearnCPP with PPO.
// The strategy-conditioning architecture (StrategyObsBuilder + StrategyReward)
// is already wired in, but Stage 1 uses a single "general" strategy.
//
// All hyperparameters are loaded from checkpoints/{bot}/bot_config.json.
// If the config file is missing, hardcoded defaults are used.
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
#include "RLBotClient.h"

#include <GigaLearnCPP/Util/InferUnit.h>
#include <rlbot/platform.h>
#include <nlohmann/json.hpp>

#include <string>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace GGL;
using namespace RLGC;
using json = nlohmann::json;

// ============================================================================
// Bot configuration — loaded from bot_config.json per bot
// ============================================================================
struct BotConfig {
	// Gamemode
	std::string gamemode = "1v1";

	// Training
	int numGames   = 32;
	int tickSkip   = 8;
	int randomSeed = 42;
	int64_t tsPerSave = 5'000'000;

	// PPO
	int64_t tsPerItr      = 50'000;
	int64_t batchSize     = 50'000;
	int64_t miniBatchSize = 50'000;
	int     epochs        = 2;
	float   entropyScale  = 0.035f;
	float   gaeGamma      = 0.99f;
	float   gaeLambda     = 0.95f;
	float   clipRange     = 0.2f;
	float   policyLR      = 1.5e-4f;
	float   criticLR      = 1.5e-4f;

	// Network architecture (layer sizes)
	std::vector<int> sharedHead = { 256, 256 };
	std::vector<int> policy     = { 256, 256, 256 };
	std::vector<int> critic     = { 256, 256, 256 };

	// Reward weights
	float rwStrongTouch         = 60.f;
	float rwTouchBall           = 0.f;
	float rwVelocityPlayerToBall = 10.f;
	float rwFaceBall            = 0.f;
	float rwVelocityBallToGoal  = 2.f;
	float rwGoal                = 20.f;
	float rwBump                = 20.f;
	float rwDemo                = 80.f;
	float rwAir                 = 0.25f;
	float rwSpeed               = 5.f;
	float rwPickupBoost         = 10.f;
	float rwSaveBoost           = 5.f;

	int GetNumCars() const {
		if (gamemode == "2v2") return 4;
		if (gamemode == "3v3") return 6;
		return 2;  // 1v1 default
	}
};

static BotConfig g_botConfig;  // Global so EnvCreateFunc can access it

// ----------------------------------------------------------------------------
// Load bot_config.json — falls back to defaults if missing/malformed
// ----------------------------------------------------------------------------
static BotConfig LoadBotConfig(const std::string& botName) {
	BotConfig cfg;
	auto path = std::filesystem::path("checkpoints") / botName / "bot_config.json";

	if (!std::filesystem::exists(path)) {
		printf("No bot_config.json found for '%s', using defaults\n", botName.c_str());
		fflush(stdout);
		return cfg;
	}

	try {
		std::ifstream f(path);
		auto j = json::parse(f);

		// Gamemode
		if (j.contains("gamemode")) cfg.gamemode = j["gamemode"].get<std::string>();

		// Training
		if (j.contains("training")) {
			auto& t = j["training"];
			if (t.contains("numGames"))   cfg.numGames   = t["numGames"];
			if (t.contains("tickSkip"))   cfg.tickSkip   = t["tickSkip"];
			if (t.contains("randomSeed")) cfg.randomSeed = t["randomSeed"];
			if (t.contains("tsPerSave"))  cfg.tsPerSave  = t["tsPerSave"];
		}

		// PPO
		if (j.contains("ppo")) {
			auto& p = j["ppo"];
			if (p.contains("tsPerItr"))      cfg.tsPerItr      = p["tsPerItr"];
			if (p.contains("batchSize"))     cfg.batchSize     = p["batchSize"];
			if (p.contains("miniBatchSize")) cfg.miniBatchSize = p["miniBatchSize"];
			if (p.contains("epochs"))        cfg.epochs        = p["epochs"];
			if (p.contains("entropyScale"))  cfg.entropyScale  = p["entropyScale"];
			if (p.contains("gaeGamma"))      cfg.gaeGamma      = p["gaeGamma"];
			if (p.contains("gaeLambda"))     cfg.gaeLambda     = p["gaeLambda"];
			if (p.contains("clipRange"))     cfg.clipRange     = p["clipRange"];
			if (p.contains("policyLR"))      cfg.policyLR      = p["policyLR"];
			if (p.contains("criticLR"))      cfg.criticLR      = p["criticLR"];
		}

		// Network
		if (j.contains("network")) {
			auto& n = j["network"];
			if (n.contains("sharedHead")) cfg.sharedHead = n["sharedHead"].get<std::vector<int>>();
			if (n.contains("policy"))     cfg.policy     = n["policy"].get<std::vector<int>>();
			if (n.contains("critic"))     cfg.critic     = n["critic"].get<std::vector<int>>();
		}

		// Rewards
		if (j.contains("rewards")) {
			auto& r = j["rewards"];
			if (r.contains("strongTouch"))         cfg.rwStrongTouch         = r["strongTouch"];
			if (r.contains("touchBall"))           cfg.rwTouchBall           = r["touchBall"];
			if (r.contains("velocityPlayerToBall")) cfg.rwVelocityPlayerToBall = r["velocityPlayerToBall"];
			if (r.contains("faceBall"))            cfg.rwFaceBall            = r["faceBall"];
			if (r.contains("velocityBallToGoal"))  cfg.rwVelocityBallToGoal  = r["velocityBallToGoal"];
			if (r.contains("goal"))                cfg.rwGoal                = r["goal"];
			if (r.contains("bump"))                cfg.rwBump                = r["bump"];
			if (r.contains("demo"))                cfg.rwDemo                = r["demo"];
			if (r.contains("air"))                 cfg.rwAir                 = r["air"];
			if (r.contains("speed"))               cfg.rwSpeed               = r["speed"];
			if (r.contains("pickupBoost"))         cfg.rwPickupBoost         = r["pickupBoost"];
			if (r.contains("saveBoost"))           cfg.rwSaveBoost           = r["saveBoost"];
		}

		printf("Loaded bot_config.json: gamemode=%s, numGames=%d, tickSkip=%d\n",
		       cfg.gamemode.c_str(), cfg.numGames, cfg.tickSkip);
		fflush(stdout);

	} catch (const std::exception& e) {
		printf("WARNING: Failed to parse bot_config.json: %s (using defaults)\n", e.what());
		fflush(stdout);
	}

	return cfg;
}

// ----------------------------------------------------------------------------
// Build the "general" strategy reward set using config weights
// ----------------------------------------------------------------------------
static StrategyRewardRow BuildGeneralRewards(const BotConfig& bc) {
	StrategyRewardRow row;
	row.rewards = {
		{ new StrongTouchReward(20, 100),                        bc.rwStrongTouch },
		{ new TouchBallReward(),                                 bc.rwTouchBall },
		{ new VelocityPlayerToBallReward(),                      bc.rwVelocityPlayerToBall },
		{ new FaceBallReward(),                                  bc.rwFaceBall },
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 10), bc.rwVelocityBallToGoal },
		{ new GoalReward(),                                      bc.rwGoal },
		{ new ZeroSumReward(new BumpReward(), 0.5f),             bc.rwBump },
		{ new ZeroSumReward(new DemoReward(), 0.5f),             bc.rwDemo },
		{ new AirReward(),                                       bc.rwAir },
		{ new SpeedReward(),                                     bc.rwSpeed },
		{ new PickupBoostReward(),                               bc.rwPickupBoost },
		{ new SaveBoostReward(),                                 bc.rwSaveBoost },
	};
	return row;
}

// ----------------------------------------------------------------------------
// Environment factory — called once per parallel game instance
// ----------------------------------------------------------------------------
EnvCreateResult EnvCreateFunc(int index) {
	std::vector<StrategyRewardRow> rewardRows = { BuildGeneralRewards(g_botConfig) };
	auto strategyVec = Strategy::DefaultVector();
	auto* strategyReward = new StrategyReward(strategyVec, std::move(rewardRows));

	std::vector<WeightedReward> rewards = {
		{ strategyReward, 1.0f }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(15),
		new GoalScoreCondition()
	};

	// Create arena with correct number of cars for the gamemode
	auto arena = Arena::Create(GameMode::SOCCAR);
	int numCars = g_botConfig.GetNumCars();
	for (int i = 0; i < numCars; i++) {
		arena->AddCar(i % 2 == 0 ? Team::BLUE : Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder   = new StrategyObsBuilder(strategyVec);
	result.stateSetter  = new RandomState(true, true, false);
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;
	result.arena = arena;

	return result;
}

// ----------------------------------------------------------------------------
// Metrics callback
// ----------------------------------------------------------------------------
void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	bool doExpensive = (rand() % 4) == 0;

	for (auto& state : states) {
		if (doExpensive) {
			for (auto& player : state.players) {
				report.AddAvg("Player/Speed", player.vel.Length());
				report.AddAvg("Player/In Air", !player.isOnGround);
				report.AddAvg("Player/Ball Touch", player.ballTouchedStep);
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Toward Ball",
				              RS_MAX(0, player.vel.Dot(dirToBall)));
				report.AddAvg("Player/Boost", player.boost);
				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}
		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

// ----------------------------------------------------------------------------
// Network architecture helpers (shared between training and RLBot modes)
// ----------------------------------------------------------------------------
static PartialModelConfig MakeSharedHeadConfig(const std::vector<int>& layers = { 256, 256 }) {
	PartialModelConfig c = {};
	c.layerSizes     = layers;
	c.optimType      = ModelOptimType::ADAM;
	c.activationType = ModelActivationType::RELU;
	c.addLayerNorm   = true;
	c.addOutputLayer = false;  // shared head feeds into policy/critic, no output layer
	return c;
}

static PartialModelConfig MakePolicyConfig(const std::vector<int>& layers = { 256, 256, 256 }) {
	PartialModelConfig c = {};
	c.layerSizes     = layers;
	c.optimType      = ModelOptimType::ADAM;
	c.activationType = ModelActivationType::RELU;
	c.addLayerNorm   = true;
	return c;
}

// ----------------------------------------------------------------------------
// RLBot mode — runs the trained model inside Rocket League via RLBot
// ----------------------------------------------------------------------------
int RunRLBotMode() {
	printf("Starting RLBot mode...\n"); fflush(stdout);

	auto strategyVec = Strategy::DefaultVector();
	auto* obsBuilder = new StrategyObsBuilder(strategyVec);
	auto* actionParser = new DefaultAction();

	// Build a dummy 1v1 state to measure obs size
	GameState dummyState;
	Player p1, p2;
	p1.team = Team::BLUE;  p1.carId = 0;
	p2.team = Team::ORANGE; p2.carId = 1;
	dummyState.players = { p1, p2 };
	int obsSize = obsBuilder->BuildObs(dummyState.players[0], dummyState).size();
	printf("Obs size: %d\n", obsSize); fflush(stdout);

	auto modelsFolder = std::filesystem::path(
		rlbot::platform::GetExecutableDirectory()
	);
	printf("Loading models from: %s\n", modelsFolder.string().c_str()); fflush(stdout);

	try {
		auto* inferUnit = new InferUnit(
			obsBuilder, obsSize, actionParser,
			MakeSharedHeadConfig(), MakePolicyConfig(),
			modelsFolder, false  // CPU for RLBot
		);

		RLBotParams params = {};
		params.port        = 42653;
		params.tickSkip    = 8;
		params.actionDelay = 7;
		params.inferUnit   = inferUnit;

		printf("Bot server starting on port %d...\n", params.port); fflush(stdout);
		RLBotClient::Run(params);
	} catch (const std::exception& e) {
		printf("ERROR: %s\n", e.what()); fflush(stdout);
		return 1;
	}

	return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
	std::string botName = "default";
	bool rlbotMode = false;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--bot") == 0 && i + 1 < argc) {
			botName = argv[++i];
		} else if (strcmp(argv[i], "-dll-path") == 0 && i + 1 < argc) {
			rlbotMode = true;
			i++;
		}
	}

	if (rlbotMode) {
		return RunRLBotMode();
	}

	// ---- Load per-bot config ----
	g_botConfig = LoadBotConfig(botName);

	printf("Bot: %s | Gamemode: %s\n", botName.c_str(), g_botConfig.gamemode.c_str());
	fflush(stdout);

	RocketSim::Init("./collision_meshes");

	// ---- Learner configuration (from bot config) ----
	LearnerConfig cfg = {};

	cfg.deviceType  = LearnerDeviceType::GPU_CUDA;
	cfg.tickSkip    = g_botConfig.tickSkip;
	cfg.actionDelay = g_botConfig.tickSkip - 1;
	cfg.numGames    = g_botConfig.numGames;
	cfg.randomSeed  = g_botConfig.randomSeed;

	// PPO
	cfg.ppo.tsPerItr      = g_botConfig.tsPerItr;
	cfg.ppo.batchSize     = g_botConfig.batchSize;
	cfg.ppo.miniBatchSize = g_botConfig.miniBatchSize;
	cfg.ppo.epochs        = g_botConfig.epochs;
	cfg.ppo.entropyScale  = g_botConfig.entropyScale;
	cfg.ppo.gaeGamma      = g_botConfig.gaeGamma;
	cfg.ppo.gaeLambda     = g_botConfig.gaeLambda;
	cfg.ppo.clipRange     = g_botConfig.clipRange;
	cfg.ppo.policyLR      = g_botConfig.policyLR;
	cfg.ppo.criticLR      = g_botConfig.criticLR;

	// Network architecture
	cfg.ppo.sharedHead = MakeSharedHeadConfig(g_botConfig.sharedHead);
	cfg.ppo.policy     = MakePolicyConfig(g_botConfig.policy);
	cfg.ppo.critic.layerSizes     = g_botConfig.critic;
	cfg.ppo.critic.optimType      = ModelOptimType::ADAM;
	cfg.ppo.critic.activationType = ModelActivationType::RELU;
	cfg.ppo.critic.addLayerNorm   = true;

	// Logging & checkpoints
	cfg.sendMetrics         = false;
	cfg.renderMode          = false;
	cfg.checkpointFolder    = std::string("checkpoints/") + botName;
	cfg.tsPerSave           = g_botConfig.tsPerSave;
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
