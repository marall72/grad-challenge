from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from game.environment import PlatformerEnv
from game.config import DEFAULT_CONFIG


# ============================================================
# SETTINGS
# ============================================================

TOTAL_TIMESTEPS = 100_000

MAX_EPISODE_STEPS = 1000

SEED = DEFAULT_CONFIG.seed


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
# TENSORBOARD
# ============================================================

LOG_DIR = "./logs/"

TB_LOG_NAME = "PPO_Platformer"

if not DEFAULT_CONFIG.use_hazard_reward:

    TB_LOG_NAME = "PPO_Platformer_no_hazard_reward"

elif not DEFAULT_CONFIG.use_enemy_observation:

    TB_LOG_NAME = "PPO_Platformer_no_enemy_observation"

elif not DEFAULT_CONFIG.use_movement_reward:

    TB_LOG_NAME = "PPO_Platformer_no_movement_reward"


# ============================================================
# ENVIRONMENT
# ============================================================

env = PlatformerEnv(
    max_episode_steps=MAX_EPISODE_STEPS,
    seed=SEED,
)

env = Monitor(env)


# ============================================================
# OBSERVATION INFORMATION
# ============================================================

obs, info = env.reset(seed=SEED)


observation_names = [
    "player_x",
    "player_y",
    "player_vx",
    "player_vy",
    "grounded",

    "goal_relative_x",
    "goal_relative_y",

    "hazard_relative_x",
    "hazard_relative_y",

    "next_platform_relative_x",
    "next_platform_relative_y",
    "next_platform_width",
    "next_platform_distance",

    "hazard_left_relative_x",
    "hazard_right_relative_x",
    "hazard_horizontal_distance",
]


# ------------------------------------------------------------
# Enemy observations
# ------------------------------------------------------------

if DEFAULT_CONFIG.use_enemy_observation:

    observation_names.extend([
        "enemy_relative_x",
        "enemy_relative_y",
        "enemy_left_relative_x",
        "enemy_right_relative_x",
        "enemy_horizontal_distance",
    ])


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("Environment Configuration")
print("=" * 70)

print(
    f"Hazard reward:         "
    f"{DEFAULT_CONFIG.use_hazard_reward}"
)

print(
    f"Enemy observation:     "
    f"{DEFAULT_CONFIG.use_enemy_observation}"
)

print(
    f"Movement reward:       "
    f"{DEFAULT_CONFIG.use_movement_reward}"
)

print(f"Seed:                  {SEED}")

print("=" * 70)


# ============================================================
# PRINT OBSERVATION INFORMATION
# ============================================================

print()
print("=" * 70)
print("Observation Information")
print("=" * 70)

print(
    f"Number of observations: "
    f"{len(obs)}"
)

print(
    f"Observation shape:      "
    f"{obs.shape}"
)

print()

for index, (name, value) in enumerate(
    zip(observation_names, obs)
):

    print(
        f"{index:2d}  "
        f"{name:<32} "
        f"{value:10.3f}"
    )

print()

print("Observation space:")
print(env.observation_space)

print("=" * 70)


# ============================================================
# VERIFY OBSERVATION COUNT
# ============================================================

expected_observation_count = (
    21
    if DEFAULT_CONFIG.use_enemy_observation
    else 16
)

if len(obs) != expected_observation_count:

    raise RuntimeError(
        "Observation count mismatch! "
        f"Expected {expected_observation_count}, "
        f"but got {len(obs)}."
    )


# ============================================================
# PPO MODEL
# ============================================================

model = PPO(
    policy="MlpPolicy",
    env=env,

    # --------------------------------------------------------
    # Learning
    # --------------------------------------------------------

    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,

    # --------------------------------------------------------
    # RL parameters
    # --------------------------------------------------------

    gamma=0.99,
    gae_lambda=0.95,

    # --------------------------------------------------------
    # Exploration
    # --------------------------------------------------------

    ent_coef=0.01,

    # --------------------------------------------------------
    # Value function
    # --------------------------------------------------------

    vf_coef=0.5,

    # --------------------------------------------------------
    # PPO clipping
    # --------------------------------------------------------

    clip_range=0.2,

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    device="cpu",

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    verbose=1,
    tensorboard_log=LOG_DIR,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("Starting PPO training")
print("=" * 70)

print(
    f"Model:               {MODEL_NAME}"
)

print(
    f"Timesteps:           "
    f"{TOTAL_TIMESTEPS:,}"
)

print(
    f"Max episode steps:   "
    f"{MAX_EPISODE_STEPS}"
)

print(
    f"Seed:                "
    f"{SEED}"
)

print(
    f"Observation count:   "
    f"{len(obs)}"
)

print(
    f"TensorBoard logs:    "
    f"{LOG_DIR}"
)

print("=" * 70)
print()


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True,
    tb_log_name=TB_LOG_NAME,
)


# ============================================================
# SAVE
# ============================================================

model.save(MODEL_NAME)


# ============================================================
# TRAINING FINISHED
# ============================================================

print()
print("=" * 70)
print("Training finished")
print("=" * 70)

print(
    f"Model saved as:     "
    f"{MODEL_NAME}.zip"
)

print(
    f"TensorBoard logs:   "
    f"{LOG_DIR}"
)

print("=" * 70)


# ============================================================
# CLEANUP
# ============================================================

env.close()