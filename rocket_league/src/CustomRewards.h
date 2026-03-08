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
// Reward for being close to the ball
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
// Reward for positioning between ball and own goal (defensive)
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
// Reward for ball height (encourage aerials)
// ---------------------------------------------------------------------------
class BallHeightReward : public Reward {
public:
	float maxHeight;

	BallHeightReward(float maxHeight = 2000.f) : maxHeight(maxHeight) {}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return RS_CLAMP(state.ball.pos.z / maxHeight, 0.f, 1.f);  // .z is up in RocketSim
	}
};

// ---------------------------------------------------------------------------
// Penalty for staying idle (low speed for too long)
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


// ---------------------------------------------------------------------------
// Reward for keeping wheels on the ground
// ---------------------------------------------------------------------------
class WheelsDownReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return player.isOnGround ? 1.f : 0.f;
	}
};


// ============================================================================
// ADVANCED CUSTOM REWARDS
// ============================================================================

// ---------------------------------------------------------------------------
// Reward for lining up behind the ball toward the opponent's goal.
// Returns [0, 1]: 1.0 when the player is directly behind the ball
// relative to the goal, i.e. a shot would go straight in.
// ---------------------------------------------------------------------------
class AlignBallToGoalReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		Vec goalTarget = (player.team == Team::BLUE)
			? CommonValues::ORANGE_GOAL_BACK
			: CommonValues::BLUE_GOAL_BACK;

		Vec ballToGoal = (goalTarget - state.ball.pos).Normalized();
		Vec playerToBall = (state.ball.pos - player.pos).Normalized();

		// Dot product: 1.0 when player→ball→goal are perfectly aligned
		float alignment = playerToBall.Dot(ballToGoal);
		return RS_MAX(0.f, alignment);
	}
};

// ---------------------------------------------------------------------------
// Reward for touching the ball while airborne (aerial plays).
// Returns 1 on the tick the player touches the ball while both
// the player and ball are off the ground.
// ---------------------------------------------------------------------------
class AerialTouchReward : public Reward {
public:
	float minBallHeight;  // minimum ball height to count as aerial

	AerialTouchReward(float minBallHeight = 300.f) : minBallHeight(minBallHeight) {}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep) return 0.f;
		if (player.isOnGround) return 0.f;
		if (state.ball.pos.z < minBallHeight) return 0.f;

		// Scale reward by ball height — higher aerials are more impressive
		float heightScale = RS_CLAMP(state.ball.pos.z / 1500.f, 0.5f, 2.f);
		return heightScale;
	}
};

// ---------------------------------------------------------------------------
// Reward for dribbling (ball on top of car, close and controlled).
// Returns [0, 1]: high when ball is close, on top of car, and
// both are moving at similar speed in similar direction.
// ---------------------------------------------------------------------------
class GroundDribbleReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		Vec ballRelPos = state.ball.pos - player.pos;

		// Ball must be above the car (z > 0) and close (within ~250 UU)
		float dist = ballRelPos.Length();
		if (dist > 300.f || ballRelPos.z < 50.f) return 0.f;

		// Proximity component [0, 1]
		float proxReward = 1.f - (dist / 300.f);

		// Velocity alignment: player and ball moving in same direction
		float playerSpeed = player.vel.Length();
		float ballSpeed = state.ball.vel.Length();
		float velAlign = 0.f;
		if (playerSpeed > 100.f && ballSpeed > 100.f) {
			velAlign = RS_MAX(0.f, player.vel.Normalized().Dot(state.ball.vel.Normalized()));
		}

		// Ball height component: reward ball being on top of car (~120-200 UU above)
		float idealHeight = 150.f;
		float heightDiff = fabsf(ballRelPos.z - idealHeight);
		float heightReward = RS_CLAMP(1.f - heightDiff / 100.f, 0.f, 1.f);

		return proxReward * 0.4f + velAlign * 0.3f + heightReward * 0.3f;
	}
};

// ---------------------------------------------------------------------------
// Continuous reward for being between ball and own goal when ball
// is heading toward the player's goal. Better than DefensivePositionReward
// because it only activates when there's defensive need.
// Returns [0, 1].
// ---------------------------------------------------------------------------
class ConditionalDefenseReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		Vec ownGoal = (player.team == Team::BLUE)
			? CommonValues::BLUE_GOAL_BACK
			: CommonValues::ORANGE_GOAL_BACK;

		// Is the ball heading toward our goal?
		Vec ballToGoal = (ownGoal - state.ball.pos).Normalized();
		float ballTowardGoal = state.ball.vel.Normalized().Dot(ballToGoal);
		if (ballTowardGoal < 0.1f) return 0.f;  // ball not heading toward our goal

		// Is the player between ball and goal?
		Vec playerToBall = (state.ball.pos - player.pos).Normalized();
		float behindBall = ballToGoal.Dot(playerToBall);

		// Scale by how fast ball is heading goalward
		float urgency = RS_CLAMP(ballTowardGoal, 0.f, 1.f);
		return RS_MAX(0.f, behindBall) * urgency;
	}
};

// ---------------------------------------------------------------------------
// Reward for positioning behind the ball relative to opponent goal,
// setting up a power shot. Returns [0, 1].
// Unlike AlignBallToGoalReward, this also considers distance —
// being close AND aligned is better.
// ---------------------------------------------------------------------------
class ShotSetupReward : public Reward {
public:
	float maxDist;

	ShotSetupReward(float maxDist = 2000.f) : maxDist(maxDist) {}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		Vec goalTarget = (player.team == Team::BLUE)
			? CommonValues::ORANGE_GOAL_BACK
			: CommonValues::BLUE_GOAL_BACK;

		Vec ballToGoal = (goalTarget - state.ball.pos).Normalized();
		Vec playerToBall = (state.ball.pos - player.pos).Normalized();

		// Alignment: are we behind the ball aiming at goal?
		float alignment = RS_MAX(0.f, playerToBall.Dot(ballToGoal));

		// Proximity: are we close enough to take the shot?
		float dist = (state.ball.pos - player.pos).Length();
		float proximity = RS_CLAMP(1.f - dist / maxDist, 0.f, 1.f);

		return alignment * proximity;
	}
};

// ---------------------------------------------------------------------------
// Reward for kickoff speed — incentivizes moving fast toward the ball
// when the ball is near center (at kickoff). Returns [0, 1].
// Only active when ball is within kickoffRadius of center.
// ---------------------------------------------------------------------------
class KickoffReward : public Reward {
public:
	float kickoffRadius;

	KickoffReward(float kickoffRadius = 500.f) : kickoffRadius(kickoffRadius) {}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		// Only active at kickoff (ball near center)
		Vec center(0, 0, 93);  // ball rest position
		float ballDistFromCenter = (state.ball.pos - center).Length();
		if (ballDistFromCenter > kickoffRadius) return 0.f;

		// Reward speed toward the ball
		Vec toBall = (state.ball.pos - player.pos).Normalized();
		float speedToBall = player.vel.Dot(toBall);
		return RS_CLAMP(speedToBall / CommonValues::CAR_MAX_SPEED, 0.f, 1.f);
	}
};

// ---------------------------------------------------------------------------
// Reward for saving boost: penalizes holding boost down while supersonic
// (wasting boost) or while driving away from the play.
// Returns [-1, 0]: 0 when not wasting, -1 when wasting heavily.
// ---------------------------------------------------------------------------
class BoostWasteReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		// Only penalize if actively boosting (timeSpentBoosting increasing)
		if (!player.prev) return 0.f;
		bool boosting = player.timeSpentBoosting > player.prev->timeSpentBoosting;
		if (!boosting) return 0.f;

		// Penalize boosting while supersonic (no speed gained)
		if (player.isSupersonic) return -1.f;

		return 0.f;
	}
};

// ---------------------------------------------------------------------------
// Reward for ball possession — being the closest player on your team
// to the ball. Encourages challenging for the ball.
// Returns 1 if you're closest on your team, 0 otherwise.
// ---------------------------------------------------------------------------
class PossessionReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		float myDist = (state.ball.pos - player.pos).Length();

		for (auto& other : state.players) {
			if (other.carId == player.carId) continue;
			if (other.team != player.team) continue;
			float otherDist = (state.ball.pos - other.pos).Length();
			if (otherDist < myDist) return 0.f;  // teammate is closer
		}
		return 1.f;
	}
};

// ---------------------------------------------------------------------------
// Reward for boosting and gaining speed
// Returns [0, 1]: rewards the player for using boost to accelerate.
// Higher when boost is actively being used AND speed is increasing.
// ---------------------------------------------------------------------------
class BoostAccelReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!player.prev) return 0.f;

		// Check if currently boosting
		bool boosting = player.timeSpentBoosting > player.prev->timeSpentBoosting;
		if (!boosting) return 0.f;

		// Check if speed is increasing
		float prevSpeed = player.prev->vel.Length();
		float currSpeed = player.vel.Length();
		float speedGain = currSpeed - prevSpeed;

		if (speedGain < 0.f) return 0.f;  // speed decreased, penalize

		// Scale reward by speed gain relative to max speed
		float maxAccel = CommonValues::CAR_MAX_SPEED;
		return RS_CLAMP(speedGain / (maxAccel * 0.1f), 0.f, 1.f);
	}
};

// ---------------------------------------------------------------------------
// Reward for moving toward the ball, clamped to [0, 1]
// Unlike VelocityPlayerToBallReward, this never goes negative —
// moving away from the ball returns 0, not -1.
// ---------------------------------------------------------------------------
class PositiveVelocityPlayerToBallReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		Vec playerToBall = (state.ball.pos - player.pos).Normalized();
		float velDot = player.vel.Normalized().Dot(playerToBall);
		return RS_MAX(0.f, velDot);
	}
};

// ---------------------------------------------------------------------------
// Reward for strong aerial touch — touching the ball while airborne
// with high impact speed. Returns [0, 1].
// ---------------------------------------------------------------------------
class StrongAerialTouchReward : public Reward {
public:
	float minBallHeight;
	float minImpactSpeed;

	StrongAerialTouchReward(float minBallHeight = 300.f, float minImpactSpeed = 500.f)
		: minBallHeight(minBallHeight), minImpactSpeed(minImpactSpeed) {}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep) return 0.f;
		if (player.isOnGround) return 0.f;
		if (state.ball.pos.z < minBallHeight) return 0.f;

		// Calculate relative impact speed (how fast player is approaching ball)
		Vec relVel = player.vel - state.ball.vel;
		float impactSpeed = relVel.Length();

		if (impactSpeed < minImpactSpeed) return 0.f;

		// Scale by impact speed and height — harder hits from higher = better
		float speedScale = RS_CLAMP(impactSpeed / (CommonValues::CAR_MAX_SPEED * 1.5f), 0.f, 1.f);
		float heightScale = RS_CLAMP(state.ball.pos.z / 2000.f, 0.5f, 1.f);

		return speedScale * heightScale;
	}
};

// ---------------------------------------------------------------------------
// Reward for increasing ball speed
// Returns [0, 1]: rewards the player for hitting the ball hard and fast.
// Higher when the ball is moving quickly toward the opponent's goal.
// ---------------------------------------------------------------------------
class BallSpeedReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		float ballSpeed = state.ball.vel.Length();
		return RS_CLAMP(ballSpeed / CommonValues::BALL_MAX_SPEED, 0.f, 1.f);
	}
};

// ---------------------------------------------------------------------------
// Reward for boosting in the air while moving toward the ball
// Returns [0, 1]: rewards aerial boost usage when positioned to challenge.
// Higher when boosting, airborne, AND moving toward the ball.
// ---------------------------------------------------------------------------
class AirBoostToBallReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!player.prev) return 0.f;

		// Must be in the air
		if (player.isOnGround) return 0.f;

		// Must be boosting
		bool boosting = player.timeSpentBoosting > player.prev->timeSpentBoosting;
		if (!boosting) return 0.f;

		// Reward moving toward ball
		Vec playerToBall = (state.ball.pos - player.pos).Normalized();
		float velToBall = player.vel.Normalized().Dot(playerToBall);

		return RS_MAX(0.f, velToBall);
	}
};

// ---------------------------------------------------------------------------
// Reward for boosting while hitting the ball
// Returns [0, 1]: rewards the player for using boost during ball contact.
// Higher when boosting AND touching the ball in the same tick.
// ---------------------------------------------------------------------------
class BoostingWhileHittingReward : public Reward {
public:
	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!player.prev) return 0.f;

		// Must be touching the ball this tick
		if (!player.ballTouchedStep) return 0.f;

		// Must be boosting
		bool boosting = player.timeSpentBoosting > player.prev->timeSpentBoosting;
		if (!boosting) return 0.f;

		return 1.f;
	}
};