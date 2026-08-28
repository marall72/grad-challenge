from game.environment import PlatformerEnv


env = PlatformerEnv(
    max_episode_steps=1000,
    seed=42,
)

obs, info = env.reset(seed=42)

print("=" * 70)
print("PLAYER")
print("=" * 70)

p = env.game.player

print(f"x={p.x}")
print(f"y={p.y}")
print(f"w={p.w}")
print(f"h={p.h}")

print()
print("=" * 70)
print("GOAL")
print("=" * 70)

gx, gy, gw, gh = env.game.level.goal.bounds()

print(f"x={gx}")
print(f"y={gy}")
print(f"w={gw}")
print(f"h={gh}")

print()
print("=" * 70)
print("PLATFORMS")
print("=" * 70)

for i, platform in enumerate(env.game.level.platforms):

    x, y, w, h = platform.bounds()

    print(
        f"Platform {i}: "
        f"x={x:.1f}, "
        f"y={y:.1f}, "
        f"w={w:.1f}, "
        f"h={h:.1f}"
    )

print()
print("=" * 70)
print("HAZARDS")
print("=" * 70)

for i, hazard in enumerate(env.game.level.hazards):

    x, y, w, h = hazard.bounds()

    print(
        f"Hazard {i}: "
        f"x={x:.1f}, "
        f"y={y:.1f}, "
        f"w={w:.1f}, "
        f"h={h:.1f}"
    )

print()
print("=" * 70)
print("ENEMIES")
print("=" * 70)

for i, enemy in enumerate(env.game.level.enemies):

    x, y, w, h = enemy.bounds()

    print(
        f"Enemy {i}: "
        f"x={x:.1f}, "
        f"y={y:.1f}, "
        f"w={w:.1f}, "
        f"h={h:.1f}"
    )

env.close()