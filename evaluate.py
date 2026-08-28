from stable_baselines3 import PPO

from game.environment import PlatformerEnv
from game.game import GameState


NUM_EPISODES = 10
MAX_EPISODE_STEPS = 1000


# --------------------------------------------------
# Create environment
# --------------------------------------------------

env = PlatformerEnv(
    max_episode_steps=MAX_EPISODE_STEPS,
    seed=42,
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = PPO.load(
    "ppo_platformer",
    env=env,
)


results = []


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

for episode in range(NUM_EPISODES):

    obs, info = env.reset(seed=42)

    total_reward = 0.0
    steps = 0

    print()
    print(f"Starting Episode {episode + 1}")
    print("-------------------")

    while True:

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        done = terminated or truncated

        # --------------------------------------------------
        # Debug every 100 steps
        # --------------------------------------------------

        if steps % 100 == 0:

            player = env.game.player

            print(
                f"Step {steps}: "
                f"action={action}, "
                f"reward={reward:.4f}, "
                f"player_x={player.x:.1f}, "
                f"player_y={player.y:.1f}, "
                f"vx={player.vx:.1f}, "
                f"vy={player.vy:.1f}, "
                f"grounded={player.grounded}"
            )

        # --------------------------------------------------
        # Episode ended
        # --------------------------------------------------

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

            print(f"Player X:     {player.x:.1f}")
            print(f"Player Y:     {player.y:.1f}")
            print(f"Velocity X:   {player.vx:.1f}")
            print(f"Velocity Y:   {player.vy:.1f}")
            print(f"Grounded:     {player.grounded}")

            break

    # --------------------------------------------------
    # Determine result
    # --------------------------------------------------

    if final_state == GameState.COMPLETED:
        result = "SUCCESS"

    elif final_state == GameState.DEAD:
        result = "DEATH"

    else:
        result = "TIMEOUT"

    results.append({
        "episode": episode + 1,
        "steps": steps,
        "reward": total_reward,
        "result": result,
    })


env.close()


# --------------------------------------------------
# Summary
# --------------------------------------------------

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
    sum(r["reward"] for r in results)
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


# --------------------------------------------------
# Final report
# --------------------------------------------------

print()
print()
print("PPO Evaluation")
print("===================")

for r in results:

    print(
        f"Episode {r['episode']:2d}: "
        f"steps={r['steps']:4d}, "
        f"reward={r['reward']:8.3f}, "
        f"result={r['result']}"
    )


print()
print("Summary")
print("-------------------")

print(f"Episodes:       {NUM_EPISODES}")
print(f"Successes:      {successes}")
print(f"Deaths:         {deaths}")
print(f"Timeouts:       {timeouts}")
print(f"Success rate:   {success_rate:.1f}%")
print(f"Death rate:     {death_rate:.1f}%")
print(f"Timeout rate:   {timeout_rate:.1f}%")
print(f"Average steps:  {average_steps:.1f}")
print(f"Average reward: {average_reward:.3f}")