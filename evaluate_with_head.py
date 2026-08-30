from email.policy import default

import pygame
from collections import Counter

from stable_baselines3 import PPO

from game.environment import PlatformerEnv
from game.game import GameState
from game.config import DEFAULT_CONFIG, GameConfig


# ============================================================
# SETTINGS
# ============================================================

NUM_EPISODES = 1
MAX_EPISODE_STEPS = 1000
MODEL_NAME = "ppo_platformer"
if not (DEFAULT_CONFIG.use_hazard_reward):
    MODEL_NAME = "ppo_platformer_no_hazard_reward"
elif not(DEFAULT_CONFIG.use_enemy_observation):
    MODEL_NAME = "ppo_platformer_no_enemy_observation"
elif not(DEFAULT_CONFIG.use_movement_reward):
    MODEL_NAME = "ppo_platformer_no_movement_reward"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 500

FPS = 60

# True  = watch the trained PPO agent
# False = manually control the player
WATCH_AGENT = True

# Player stays around this position on screen.
CAMERA_OFFSET_X = 250


# ============================================================
# ENVIRONMENT + MODEL
# ============================================================

env = PlatformerEnv(
    max_episode_steps=MAX_EPISODE_STEPS,
    seed=DEFAULT_CONFIG.seed,
)

model = PPO.load(
    MODEL_NAME,
    env=env,
)


# ============================================================
# PYGAME
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (WINDOW_WIDTH, WINDOW_HEIGHT)
)

pygame.display.set_caption(
    "PPO Platformer Evaluation"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    "Arial",
    18,
)


# ============================================================
# EVALUATION
# ============================================================

results = []


for episode in range(NUM_EPISODES):

    obs, info = env.reset(seed=DEFAULT_CONFIG.seed)

    total_reward = 0.0
    steps = 0

    action_counts = Counter()

    running = True

    print()
    print("=" * 70)
    print(f"Starting Episode {episode + 1}")
    print("=" * 70)

    while running:

        # ----------------------------------------------------
        # Handle window events
        # ----------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False
                break

        if not running:
            break

        # ----------------------------------------------------
        # Get action
        # ----------------------------------------------------

        if WATCH_AGENT:

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = int(action)

        else:

            # Manual control for debugging
            keys = pygame.key.get_pressed()

            left = keys[pygame.K_LEFT]
            right = keys[pygame.K_RIGHT]
            jump = keys[pygame.K_SPACE]

            if right and jump:
                action = 4

            elif left and jump:
                action = 5

            elif right:
                action = 2

            elif left:
                action = 1

            elif jump:
                action = 3

            else:
                action = 0

        action_counts[action] += 1

        # ----------------------------------------------------
        # Step environment
        # ----------------------------------------------------

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += float(reward)
        steps += 1

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        player = env.game.player

        camera_x = (
            player.x
            - CAMERA_OFFSET_X
        )

        camera_x = max(
            0.0,
            camera_x,
        )

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        env.game.render(
            screen,
            camera_x=camera_x,
            camera_y=0.0,
        )

        # ----------------------------------------------------
        # Information overlay
        # ----------------------------------------------------

        state_text = (
            f"Episode: {episode + 1}    "
            f"Step: {steps}    "
            f"Reward: {reward:.3f}"
        )

        state_surface = font.render(
            state_text,
            True,
            (0, 0, 0),
        )

        screen.blit(
            state_surface,
            (10, 10),
        )

        player_text = (
            f"x={player.x:.1f}  "
            f"y={player.y:.1f}  "
            f"vx={player.vx:.1f}  "
            f"vy={player.vy:.1f}  "
            f"grounded={player.grounded}"
        )

        player_surface = font.render(
            player_text,
            True,
            (0, 0, 0),
        )

        screen.blit(
            player_surface,
            (10, 35),
        )

        action_names = {
            0: "NO-OP",
            1: "LEFT",
            2: "RIGHT",
            3: "JUMP",
            4: "RIGHT + JUMP",
            5: "LEFT + JUMP",
        }

        action_surface = font.render(
            f"Action: {action_names[action]}",
            True,
            (0, 0, 0),
        )

        screen.blit(
            action_surface,
            (10, 60),
        )

        # ----------------------------------------------------
        # Show game state
        # ----------------------------------------------------

        if env.game.state == GameState.DEAD:

            state_surface = font.render(
                f"DEAD - {env.game.death_reason}",
                True,
                (0, 0, 0),
            )

            screen.blit(
                state_surface,
                (10, 85),
            )

        elif env.game.state == GameState.COMPLETED:

            state_surface = font.render(
                "COMPLETED!",
                True,
                (0, 0, 0),
            )

            screen.blit(
                state_surface,
                (10, 85),
            )

        # ----------------------------------------------------
        # Camera information
        # ----------------------------------------------------

        camera_surface = font.render(
            f"Camera X: {camera_x:.1f}",
            True,
            (0, 0, 0),
        )

        screen.blit(
            camera_surface,
            (10, 110),
        )

        # ----------------------------------------------------
        # Goal information
        # ----------------------------------------------------

        goal_x, goal_y, goal_w, goal_h = (
            env.game.level.goal.bounds()
        )

        goal_surface = font.render(
            f"Goal X: {goal_x:.1f}",
            True,
            (0, 100, 0),
        )

        screen.blit(
            goal_surface,
            (10, 135),
        )

        # ----------------------------------------------------
        # Draw player camera position
        # ----------------------------------------------------

        pygame.draw.line(
            screen,
            (0, 0, 0),
            (
                CAMERA_OFFSET_X,
                0,
            ),
            (
                CAMERA_OFFSET_X,
                WINDOW_HEIGHT,
            ),
            1,
        )

        pygame.display.flip()

        # ----------------------------------------------------
        # Console logging
        # ----------------------------------------------------

        if steps % 25 == 0 or terminated or truncated:

            print(
                f"step={steps:4d} "
                f"action={action} "
                f"reward={reward:8.4f} "
                f"x={player.x:7.2f} "
                f"y={player.y:7.2f} "
                f"vx={player.vx:7.2f} "
                f"vy={player.vy:7.2f} "
                f"grounded={player.grounded}"
            )

        # ----------------------------------------------------
        # Episode ended
        # ----------------------------------------------------

        if terminated or truncated:

            print()
            print("Episode ended")
            print("-" * 70)

            print(f"Steps:        {steps}")
            print(f"Terminated:   {terminated}")
            print(f"Truncated:    {truncated}")
            print(f"Game state:   {env.game.state}")
            print(f"Death reason: {env.game.death_reason}")
            print(f"Total reward: {total_reward:.3f}")

            print()
            print("Action distribution")
            print("-" * 70)

            for action_id in range(6):

                count = action_counts[action_id]

                percentage = (
                    count / steps * 100
                    if steps > 0
                    else 0.0
                )

                print(
                    f"{action_id} = "
                    f"{action_names[action_id]:14s} "
                    f"{count:4d} times "
                    f"({percentage:6.2f}%)"
                )

            results.append({
                "episode": episode + 1,
                "steps": steps,
                "reward": total_reward,
                "result": (
                    "SUCCESS"
                    if env.game.state == GameState.COMPLETED
                    else "DEATH"
                    if env.game.state == GameState.DEAD
                    else "TIMEOUT"
                ),
            })

            # Keep final frame visible
            pygame.time.wait(1500)

            break

        clock.tick(FPS)

    if not running:
        break


# ============================================================
# CLEANUP
# ============================================================

env.close()

pygame.quit()


# ============================================================
# SUMMARY
# ============================================================

if results:

    print()
    print("=" * 70)
    print("PPO Evaluation Summary")
    print("=" * 70)

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
        / len(results)
    )

    average_steps = (
        sum(r["steps"] for r in results)
        / len(results)
    )

    print(f"Episodes:       {len(results)}")
    print(f"Successes:      {successes}")
    print(f"Deaths:         {deaths}")
    print(f"Timeouts:       {timeouts}")

    print(
        f"Success rate:   "
        f"{successes / len(results) * 100:.1f}%"
    )

    print(
        f"Death rate:     "
        f"{deaths / len(results) * 100:.1f}%"
    )

    print(
        f"Timeout rate:   "
        f"{timeouts / len(results) * 100:.1f}%"
    )

    print(
        f"Average steps:  "
        f"{average_steps:.1f}"
    )

    print(
        f"Average reward: "
        f"{average_reward:.3f}"
    )