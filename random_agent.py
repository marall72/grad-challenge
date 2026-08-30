import csv
import os

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

EPISODE_CSV = os.path.join(
    RESULTS_DIR,
    "random_results.csv",
)

SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    "random_summary.csv",
)


# ============================================================
# RUN ONE EPISODE
# ============================================================

def run_episode(seed=BASE_SEED, max_steps=MAX_EPISODE_STEPS):

    env = PlatformerEnv(
        max_episode_steps=max_steps,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    total_reward = 0.0

    for step in range(max_steps):

        # Random agent chooses an action
        # without using a learned policy.
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        if step % 100 == 0:

            print(
                f"Step {step}: "
                f"action={action}, "
                f"reward={reward:.4f}, "
                f"player_x={obs[0]:.1f}, "
                f"player_y={obs[1]:.1f}"
            )

        if terminated or truncated:

            break

    # --------------------------------------------------------
    # Determine result
    # --------------------------------------------------------

    if env.game.state == GameState.COMPLETED:

        result = "SUCCESS"

    elif env.game.state == GameState.DEAD:

        result = "DEATH"

    else:

        result = "TIMEOUT"

    steps = step + 1

    env.close()

    return {
        "steps": steps,
        "total_reward": total_reward,
        "result": result,
        "terminated": terminated,
        "truncated": truncated,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    results = []

    print()
    print("=" * 70)
    print("Random Agent Evaluation")
    print("=" * 70)
    print(f"Episodes:    {NUM_EPISODES}")
    print(f"Max steps:   {MAX_EPISODE_STEPS}")
    print(f"Base seed:   {BASE_SEED}")
    print("=" * 70)

    # ========================================================
    # RUN EPISODES
    # ========================================================

    for episode in range(NUM_EPISODES):

        seed = BASE_SEED + episode

        print()
        print(
            f"Starting Episode "
            f"{episode + 1}/{NUM_EPISODES} "
            f"(seed={seed})"
        )
        print("-" * 40)

        result = run_episode(
            seed=seed,
            max_steps=MAX_EPISODE_STEPS,
        )

        result["episode"] = episode + 1
        result["seed"] = seed

        results.append(result)

        print()
        print(
            f"Episode {episode + 1}: "
            f"steps={result['steps']}, "
            f"reward={result['total_reward']:.3f}, "
            f"result={result['result']}"
        )

    # ========================================================
    # STATISTICS
    # ========================================================

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
        successes / NUM_EPISODES * 100
    )

    death_rate = (
        deaths / NUM_EPISODES * 100
    )

    timeout_rate = (
        timeouts / NUM_EPISODES * 100
    )

    # ========================================================
    # SAVE EPISODE-LEVEL CSV
    # ========================================================

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

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "agent",
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
            "Random",
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

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("Random Agent Evaluation")
    print("=" * 70)

    for r in results:

        print(
            f"Episode {r['episode']:2d}: "
            f"steps={r['steps']:4d}, "
            f"reward={r['total_reward']:8.3f}, "
            f"result={r['result']}"
        )

    print()
    print("Summary")
    print("-" * 40)

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