from game.environment import PlatformerEnv


def run_episode(seed=42, max_steps=1000):
    env = PlatformerEnv(
        max_episode_steps=max_steps,
        seed=seed
    )

    obs, info = env.reset(seed=seed)

    total_reward = 0.0

    for step in range(max_steps):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

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
            return {
                "steps": step + 1,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
            }

    return {
        "steps": max_steps,
        "total_reward": total_reward,
        "terminated": False,
        "truncated": True,
    }


if __name__ == "__main__":
    result = run_episode(seed=42)

    print("Random Agent Result")
    print("-------------------")
    print(f"Steps: {result['steps']}")
    print(f"Total reward: {result['total_reward']:.3f}")
    print(f"Terminated: {result['terminated']}")
    print(f"Truncated: {result['truncated']}")