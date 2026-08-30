import csv
import os

from stable_baselines3 import PPO

from game.environment import PlatformerEnv
from game.game import GameState
from game.config import DEFAULT_CONFIG


# ============================================================
# SETTINGS
# ============================================================

NUM_EPISODES = 100
MAX_EPISODE_STEPS = 1000

BASE_SEED = DEFAULT_CONFIG.seed

RESULTS_DIR = "results"

EPISODE_CSV_NAME = "ppo_results_full.csv"
if not DEFAULT_CONFIG.use_hazard_reward:

    EPISODE_CSV_NAME = "ppo_results_no_hazard_reward.csv"

elif not DEFAULT_CONFIG.use_enemy_observation:

    EPISODE_CSV_NAME = "ppo_results_no_enemy_observation.csv"

elif not DEFAULT_CONFIG.use_movement_reward:

    EPISODE_CSV_NAME = "ppo_results_no_movement_reward.csv"
EPISODE_CSV = os.path.join(
    RESULTS_DIR,
    EPISODE_CSV_NAME,
)

SUMMARY_CSV_NAME = "ppo_summary_full.csv"
if not DEFAULT_CONFIG.use_hazard_reward:

    SUMMARY_CSV_NAME = "ppo_summary_no_hazard_reward.csv"

elif not DEFAULT_CONFIG.use_enemy_observation:

    SUMMARY_CSV_NAME = "ppo_summary_no_enemy_observation.csv"

elif not DEFAULT_CONFIG.use_movement_reward:

    SUMMARY_CSV_NAME = "ppo_summary_no_movement_reward.csv"
SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    SUMMARY_CSV_NAME,
)


# ============================================================
# MODEL NAME
# ============================================================

MODEL_NAME = "ppo_platformer"

if not DEFAULT_CONFIG.use_hazard_reward:

    MODEL_NAME = "ppo_platformer_no_hazard_reward"

elif not DEFAULT_CONFIG.use_enemy_observation:

    MODEL_NAME = "ppo_platformer_no_enemy_observation"

elif not DEFAULT_CONFIG.use_movement_reward:

    MODEL_NAME = "ppo_platformer_no_movement_reward"


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = PlatformerEnv(
    max_episode_steps=MAX_EPISODE_STEPS,
    seed=BASE_SEED,
)


# ============================================================
# LOAD MODEL
# ============================================================

model = PPO.load(
    MODEL_NAME,
    env=env,
)


results = []


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 70)
print("PPO Evaluation")
print("=" * 70)
print(f"Model:       {MODEL_NAME}")
print(f"Episodes:    {NUM_EPISODES}")
print(f"Max steps:   {MAX_EPISODE_STEPS}")
print(f"Base seed:   {BASE_SEED}")
print("=" * 70)


for episode in range(NUM_EPISODES):

    # Use a different but deterministic seed
    # for each evaluation episode.
    seed = BASE_SEED + episode

    obs, info = env.reset(
        seed=seed
    )

    total_reward = 0.0
    steps = 0

    print()
    print(
        f"Starting Episode "
        f"{episode + 1}/{NUM_EPISODES} "
        f"(seed={seed})"
    )
    print("-" * 40)


    # ========================================================
    # RUN EPISODE
    # ========================================================

    while True:

        # PPO chooses the action.
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = int(action)

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += float(reward)
        steps += 1

        done = terminated or truncated


        # ----------------------------------------------------
        # Debug every 100 steps
        # ----------------------------------------------------

        if steps % 100 == 0:

            player = env.game.player

            print(
                f"Step {steps}: "
                f"action={action}, "
                f"reward={float(reward):.4f}, "
                f"player_x={player.x:.1f}, "
                f"player_y={player.y:.1f}, "
                f"vx={player.vx:.1f}, "
                f"vy={player.vy:.1f}, "
                f"grounded={player.grounded}"
            )


        # ----------------------------------------------------
        # Episode ended
        # ----------------------------------------------------

        if done:

            final_state = env.game.state
            death_reason = env.game.death_reason

            player = env.game.player

            print()
            print("Episode ended")
            print("-------------------")

            print(f"Steps:        {steps}")
            print(f"Terminated:   {terminated}")
            print(f"Truncated:    {truncated}")
            print(f"Game state:   {final_state}")
            print(f"Death reason: {death_reason}")
            print(f"Last action:  {action}")

            print(
                f"Player X:     {player.x:.1f}"
            )

            print(
                f"Player Y:     {player.y:.1f}"
            )

            print(
                f"Velocity X:   {player.vx:.1f}"
            )

            print(
                f"Velocity Y:   {player.vy:.1f}"
            )

            print(
                f"Grounded:     {player.grounded}"
            )

            break


    # ========================================================
    # DETERMINE RESULT
    # ========================================================

    if final_state == GameState.COMPLETED:

        result = "SUCCESS"

    elif final_state == GameState.DEAD:

        result = "DEATH"

    else:

        result = "TIMEOUT"


    # ========================================================
    # STORE RESULT
    # ========================================================

    results.append({
        "episode": episode + 1,
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "result": result,
        "terminated": terminated,
        "truncated": truncated,
    })


# ============================================================
# CLOSE ENVIRONMENT
# ============================================================

env.close()


# ============================================================
# SUMMARY STATISTICS
# ============================================================

successes = sum(
    r["result"] == "SUCCESS"
    for r in results
)

deaths = sum(
    r["result"] == "DEATH"
    for r in results
)

timeouts = sum(
    r["result"] == "TIMEOUT"
    for r in results
)


average_reward = (
    sum(r["total_reward"] for r in results)
    / NUM_EPISODES
)


average_steps = (
    sum(r["steps"] for r in results)
    / NUM_EPISODES
)


success_rate = (
    successes
    / NUM_EPISODES
    * 100
)


death_rate = (
    deaths
    / NUM_EPISODES
    * 100
)


timeout_rate = (
    timeouts
    / NUM_EPISODES
    * 100
)


# ============================================================
# SAVE EPISODE-LEVEL CSV
# ============================================================

with open(
    EPISODE_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "episode",
        "seed",
        "steps",
        "total_reward",
        "result",
        "terminated",
        "truncated",
    ])

    for r in results:

        writer.writerow([
            r["episode"],
            r["seed"],
            r["steps"],
            f"{r['total_reward']:.6f}",
            r["result"],
            r["terminated"],
            r["truncated"],
        ])


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

with open(
    SUMMARY_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "agent",
        "model",
        "episodes",
        "successes",
        "deaths",
        "timeouts",
        "success_rate",
        "death_rate",
        "timeout_rate",
        "average_steps",
        "average_reward",
    ])

    writer.writerow([
        "PPO",
        MODEL_NAME,
        NUM_EPISODES,
        successes,
        deaths,
        timeouts,
        f"{success_rate:.2f}",
        f"{death_rate:.2f}",
        f"{timeout_rate:.2f}",
        f"{average_steps:.2f}",
        f"{average_reward:.6f}",
    ])


# ============================================================
# FINAL REPORT
# ============================================================

print()
print()
print("=" * 70)
print("PPO Evaluation")
print("=" * 70)

for r in results:

    print(
        f"Episode {r['episode']:3d}: "
        f"steps={r['steps']:4d}, "
        f"reward={r['total_reward']:8.3f}, "
        f"result={r['result']}"
    )


print()
print("Summary")
print("-" * 40)

print(f"Model:          {MODEL_NAME}")
print(f"Episodes:       {NUM_EPISODES}")
print(f"Successes:      {successes}")
print(f"Deaths:         {deaths}")
print(f"Timeouts:       {timeouts}")
print(f"Success rate:   {success_rate:.1f}%")
print(f"Death rate:     {death_rate:.1f}%")
print(f"Timeout rate:   {timeout_rate:.1f}%")
print(f"Average steps:  {average_steps:.1f}")
print(f"Average reward: {average_reward:.3f}")

print()
print("CSV files saved:")
print(f"  {EPISODE_CSV}")
print(f"  {SUMMARY_CSV}")

print("=" * 70)