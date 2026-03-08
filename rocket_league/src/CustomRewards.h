#pragma once

// ============================================================================
// CustomRewards.h — Reward Function Catalog & Custom Rewards
// ============================================================================
//
// This file serves two purposes:
//   1. A reference catalog of ALL available reward functions from RLGymCPP
//   2. A place to write your own custom reward functions
//
// To use any reward in BuildGeneralRewards() (main.cpp), just #include this
// file and instantiate the reward class with `new YourReward()`.
//
// ============================================================================
//
// HOW REWARDS WORK
// ----------------
// Every reward class inherits from RLGC::Reward and implements:
//
//   float GetReward(const Player& player, const GameState& state, bool isFinal)
//
// This is called every tick for every player. Return a float value:
//   - Positive = encourage this behavior
//   - Negative = discourage this behavior
//   - 0        = neutral
//
// The weight you assign in BuildGeneralRewards() multiplies the output.
// Example: { new SpeedReward(), 5.0f } means SpeedReward output * 5.
//
// WRAPPERS
// --------
// ZeroSumReward(child, teamSpirit, opponentScale):
//   Makes a reward zero-sum across teams. Each player's reward becomes:
//     ownReward*(1-teamSpirit) + avgTeamReward*teamSpirit - avgOpponentReward*opponentScale
//
//   Example: new ZeroSumReward(new BumpReward(), 0.5f)
//     teamSpirit=0.5 means 50% of reward shared with teammates
//     opponentScale defaults to 1.0 (full punishment for opponents)
//
// ============================================================================

#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>

using namespace RLGC;

// ============================================================================
// AVAILABLE REWARD CATALOG — Built-in rewards from RLGymCPP
// ============================================================================
//
// ---- Ball & Goal ----
//
//   GoalReward(float concedeScale = -1)
//     Returns 1 when the player's team scores, concedeScale when conceded.
//     Already zero-sum by design. Default concedeScale = -1.
//     Example: new GoalReward()        // +1 for goal, -1 for concede
//     Example: new GoalReward(-2)      // +1 for goal, -2 for concede
//
//   VelocityBallToGoalReward(bool ownGoal = false)
//     Dot product of ball velocity direction toward opponent's goal.
//     Returns [-1, 1]. Positive when ball moves toward their goal.
//     Set ownGoal=true to reward ball moving toward YOUR goal (defensive).
//     Example: new VelocityBallToGoalReward()
//
//   TouchBallReward()
//     Returns 1 on the tick the player touches the ball, 0 otherwise.
//     Example: new TouchBallReward()
//
//   StrongTouchReward(float minSpeedKPH = 20, float maxSpeedKPH = 130)
//     Returns 0-1 based on how hard the player hit the ball.
//     Ignores weak touches below minSpeedKPH. Maxes out at maxSpeedKPH.
//     Example: new StrongTouchReward(20, 100)   // custom range
//     Example: new StrongTouchReward()           // default 20-130 KPH
//
//   TouchAccelReward()
//     Returns 0-1 based on how much the player sped up the ball on touch.
//     Only rewards increasing ball speed (not slowing it down).
//     Max rewarded ball speed: 110 KPH.
//     Example: new TouchAccelReward()
//
// ---- Player Movement ----
//
//   SpeedReward()
//     Returns player speed as fraction of max car speed [0, 1].
//     Example: new SpeedReward()
//
//   VelocityReward(bool isNegative = false)
//     Same as SpeedReward but can be negated. Returns [-1, 1].
//     isNegative=true punishes going fast (useful for patience training).
//     Example: new VelocityReward()
//     Example: new VelocityReward(true)   // negative velocity reward
//
//   VelocityPlayerToBallReward()
//     Dot product of player velocity toward the ball. Returns [-1, 1].
//     Positive when player moves toward ball, negative when moving away.
//     Example: new VelocityPlayerToBallReward()
//
//   FaceBallReward()
//     Dot product of player's forward direction toward the ball. Returns [-1, 1].
//     +1 when facing directly at ball, -1 when facing away.
//     Example: new FaceBallReward()
//
// ---- Boost ----
//
//   PickupBoostReward()
//     Rewards picking up boost pads. Returns the sqrt-scaled increase in boost.
//     Only triggers when boost increases (not when using boost).
//     Example: new PickupBoostReward()
//
//   SaveBoostReward(float exponent = 0.5f)
//     Continuous reward for having boost. Returns boost^exponent clamped [0,1].
//     Default exponent=0.5 means sqrt(boost%), rewarding having ANY boost.
//     Higher exponent (e.g. 1.0) = linear, rewards high boost more.
//     Example: new SaveBoostReward()        // sqrt scaling
//     Example: new SaveBoostReward(1.0f)    // linear scaling
//
// ---- Aerial ----
//
//   AirReward()
//     Returns 1 when the player is in the air, 0 on the ground.
//     Example: new AirReward()
//
//   WavedashReward()
//     Returns 1 on the tick the player lands from a flip (wavedash).
//     Detects transition from (flipping + in air) to (on ground).
//     Example: new WavedashReward()
//
// ---- Events (binary per-tick) ----
//
//   PlayerGoalReward()
//     Returns 1 on the tick the player's INDIVIDUAL goal event fires.
//     NOTE: Only given to the player who last touched the ball, not the
//     whole team. Use GoalReward for team-wide goal rewards.
//     Example: new PlayerGoalReward()
//
//   AssistReward()
//     Returns 1 on the tick the player gets an assist.
//     Example: new AssistReward()
//
//   ShotReward()
//     Returns 1 on the tick the player takes a shot.
//     Example: new ShotReward()
//
//   ShotPassReward()
//     Returns 1 on the tick the player makes a shot pass.
//     Example: new ShotPassReward()
//
//   SaveReward()
//     Returns 1 on the tick the player makes a save.
//     Example: new SaveReward()
//
//   BumpReward()
//     Returns 1 on the tick the player bumps an opponent.
//     Example: new BumpReward()
//
//   BumpedPenalty()
//     Returns -1 on the tick the player gets bumped by an opponent.
//     Example: new BumpedPenalty()
//
//   DemoReward()
//     Returns 1 on the tick the player demolishes an opponent.
//     Example: new DemoReward()
//
//   DemoedPenalty()
//     Returns -1 on the tick the player gets demolished.
//     Example: new DemoedPenalty()
//
// ============================================================================


// ============================================================================
// CUSTOM REWARDS — Write your own below
// ============================================================================
//
// Template for a new reward:
//
//   class MyReward : public Reward {
//   public:
//       // Constructor — add any parameters you need
//       MyReward(float param = 1.0f) : myParam(param) {}
//
//       float GetReward(const Player& player, const GameState& state,
//                       bool isFinal) override {
//           // Your reward logic here.
//           // Return a float: positive = good, negative = bad.
//           //
//           // Useful fields:
//           //   player.pos          — Vec position
//           //   player.vel          — Vec velocity
//           //   player.boost        — float [0, 100]
//           //   player.isOnGround   — bool
//           //   player.isFlipping   — bool
//           //   player.ballTouchedStep — bool (touched ball this tick)
//           //   player.team         — Team::BLUE or Team::ORANGE
//           //   player.rotMat.forward — Vec facing direction
//           //   player.prev         — const Player* (previous tick, can be nullptr)
//           //
//           //   state.ball.pos      — Vec ball position
//           //   state.ball.vel      — Vec ball velocity
//           //   state.players       — vector of all players
//           //   state.goalScored    — bool
//           //   state.prev          — const GameState* (previous tick)
//           //
//           // Useful constants:
//           //   CommonValues::CAR_MAX_SPEED     — max car speed
//           //   CommonValues::BALL_MAX_SPEED    — max ball speed
//           //   CommonValues::ORANGE_GOAL_BACK  — Vec orange goal position
//           //   CommonValues::BLUE_GOAL_BACK    — Vec blue goal position
//           //   CommonValues::FIELD_SIZE        — Vec (half-lengths of field)
//           //
//           // Useful functions:
//           //   vec.Length()         — magnitude
//           //   vec.Normalized()    — unit vector
//           //   vec.Dot(other)      — dot product
//           //   RS_CLAMP(val, min, max)
//           //   RS_MIN(a, b) / RS_MAX(a, b)
//           //   RLGC::Math::KPHToVel(kph) — convert KPH to internal velocity
//
//           return 0.f;
//       }
//
//   private:
//       float myParam;
//   };
//
// Then add it to BuildGeneralRewards() in main.cpp:
//   { new MyReward(1.5f), 10.0f },   // weight = 10
//
// ============================================================================


// ---------------------------------------------------------------------------
// Example: Reward for being close to the ball
// ---------------------------------------------------------------------------
class DistanceToBallReward : public Reward {
public:
	float maxDist;

	// maxDist: distance at which reward drops to 0 (in unreal units, field is ~10000 long)
	DistanceToBallReward(float maxDist = 6000.f) : maxDist(maxDist) {}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		float dist = (state.ball.pos - player.pos).Length();
		return RS_CLAMP(1.f - (dist / maxDist), 0.f, 1.f);
	}
};

// ---------------------------------------------------------------------------
// Example: Reward for positioning between ball and own goal (defensive)
// ---------------------------------------------------------------------------
class DefensivePositionReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		Vec ownGoal = (player.team == Team::BLUE) ? CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;
		Vec ballToGoal = (ownGoal - state.ball.pos).Normalized();
		Vec playerToBall = (state.ball.pos - player.pos).Normalized();

		// Positive when player is between ball and own goal
		return ballToGoal.Dot(playerToBall);
	}
};

// ---------------------------------------------------------------------------
// Example: Reward for ball height (encourage aerials)
// ---------------------------------------------------------------------------
class BallHeightReward : public Reward {
public:
	float maxHeight;

	BallHeightReward(float maxHeight = 2000.f) : maxHeight(maxHeight) {}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return RS_CLAMP(state.ball.pos.y / maxHeight, 0.f, 1.f);  // Note: .y is up in RocketSim
	}
};

// ---------------------------------------------------------------------------
// Example: Penalty for staying idle (low speed for too long)
// ---------------------------------------------------------------------------
class IdlePenaltyReward : public Reward {
public:
	float speedThreshold;

	// speedThreshold: fraction of max speed below which player is "idle"
	IdlePenaltyReward(float speedThreshold = 0.1f) : speedThreshold(speedThreshold) {}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		float speedFrac = player.vel.Length() / CommonValues::CAR_MAX_SPEED;
		return (speedFrac < speedThreshold) ? -1.f : 0.f;
	}
};
