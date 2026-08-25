from stable_baselines3 import PPO
from game.environment import PlatformerEnv
from game.game import GameState


env = PlatformerEnv(
    max_episode_steps=1000,
    seed=42
)

model = PPO.load("ppo_platformer", env=env)

obs, info = env.reset(seed=42)

total_reward = 0.0

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if terminated or truncated:
        break

print("PPO Evaluation")
print("-------------------")
print(f"Steps: {step + 1}")
print(f"Total reward: {total_reward:.3f}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")
print(f"Final game state: {env.game.state}")

env.close()