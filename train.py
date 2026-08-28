from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from game.environment import PlatformerEnv


# ============================================================
# SETTINGS
# ============================================================

TOTAL_TIMESTEPS = 100_000

MAX_EPISODE_STEPS = 1000

SEED = 42

MODEL_NAME = "ppo_platformer"


# ============================================================
# ENVIRONMENT
# ============================================================

env = PlatformerEnv(
    max_episode_steps=MAX_EPISODE_STEPS,
    seed=SEED,
)

env = Monitor(env)


# ============================================================
# PPO MODEL
# ============================================================

model = PPO(
    policy="MlpPolicy",
    env=env,

    # Learning
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,

    # RL parameters
    gamma=0.99,
    gae_lambda=0.95,

    # Exploration
    ent_coef=0.01,

    # Value function
    vf_coef=0.5,

    # PPO clipping
    clip_range=0.2,

    # CPU
    device="cpu",

    # Logging
    verbose=1,

    seed=SEED,
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("Starting PPO training")
print("=" * 70)
print(f"Timesteps:          {TOTAL_TIMESTEPS:,}")
print(f"Max episode steps:  {MAX_EPISODE_STEPS}")
print(f"Seed:               {SEED}")
print("=" * 70)
print()


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True,
)


# ============================================================
# SAVE
# ============================================================

model.save(MODEL_NAME)

print()
print("=" * 70)
print("Training finished")
print("=" * 70)
print(f"Model saved as: {MODEL_NAME}.zip")
print("=" * 70)


# ============================================================
# CLEANUP
# ============================================================

env.close()