"""Level representation: platforms, enemies, hazards, goal, and start.

The level stores simple axis-aligned rectangles for platforms and entities.
It is intentionally minimal but suitable for simulation and collision checks.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

from .entities import Enemy, Hazard, Goal


@dataclass
class Platform:
    x: float
    y: float
    w: int
    h: int

    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class Level:
    platforms: List[Platform]
    enemies: List[Enemy]
    hazards: List[Hazard]
    goal: Goal
    start_x: float
    start_y: float

    @staticmethod
    def sample(seed: Optional[int] = None) -> "Level":
        rng = random.Random(seed)
        # Simple flat ground with a pit in the middle and a raised platform
        platforms: List[Platform] = []
        ground_y = 400
        # left ground
        platforms.append(Platform(0, ground_y, 300, 80))
        # pit/gap
        platforms.append(Platform(420, ground_y, 300, 80))
        # raised platform to cross pit (small)
        platforms.append(Platform(320, ground_y - 120, 120, 20))
        # far right ground including goal
        platforms.append(Platform(740, ground_y, 600, 80))

        # enemy: patrols on the far right ground
        enemy_min_x = 780
        enemy_max_x = 980
        enemy_x = rng.uniform(enemy_min_x, enemy_max_x)
        enemy = Enemy(x=enemy_x, y=ground_y - 48, w=32, h=48, vx=80.0, patrol_min_x=enemy_min_x, patrol_max_x=enemy_max_x)

        # hazard: a small spike pit in the left ground near center
        hazard = Hazard(x=180, y=ground_y - 20, w=40, h=20)

        # goal: flag at far right
        goal = Goal(x=1280, y=ground_y - 120, w=24, h=120)

        # start position: left side
        start_x, start_y = (40.0, ground_y - 48)

        return Level(platforms=platforms, enemies=[enemy], hazards=[hazard], goal=goal, start_x=start_x, start_y=start_y)

    def pick_spawn(self) -> Tuple[float, float]:
        return (self.start_x, self.start_y)
