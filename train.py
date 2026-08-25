from stable_baselines3 import PPO
from game.environment import PlatformerEnv


env = PlatformerEnv(
    max_episode_steps=1000,
    seed=42
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
)

model.learn(total_timesteps=100_000)

model.save("ppo_platformer")

env.close()