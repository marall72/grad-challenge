from game.environment import PlatformerEnv


def main():
    env = PlatformerEnv(
        max_episode_steps=1000,
        seed=0,
    )

    try:
        for action in [0, 1, 2, 3]:

            print()
            print("=" * 70)
            print(f"Testing action {action}")
            print("0 = NO-OP | 1 = LEFT | 2 = RIGHT | 3 = JUMP")
            print("=" * 70)

            obs, info = env.reset(seed=0)

            previous_distance = env._compute_goal_distance()

            print(
                f"Initial: "
                f"x={env.game.player.x:.2f}, "
                f"y={env.game.player.y:.2f}, "
                f"goal_distance={previous_distance:.2f}"
            )

            for step in range(10):

                obs, reward, terminated, truncated, info = env.step(action)

                current_distance = env._compute_goal_distance()
                progress = previous_distance - current_distance

                player = env.game.player

                print(
                    f"step={step:2d} | "
                    f"action={action} | "
                    f"reward={reward:8.4f} | "
                    f"progress={progress:8.4f} | "
                    f"goal_dist={current_distance:8.2f} | "
                    f"x={player.x:7.2f} | "
                    f"y={player.y:7.2f} | "
                    f"vx={player.vx:7.2f} | "
                    f"vy={player.vy:7.2f} | "
                    f"grounded={player.grounded}"
                )

                previous_distance = current_distance

                if terminated or truncated:
                    print(
                        f"Episode ended: "
                        f"terminated={terminated}, "
                        f"truncated={truncated}"
                    )
                    break

    finally:
        env.close()


if __name__ == "__main__":
    main()