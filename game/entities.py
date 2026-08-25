"""Simple entity definitions: Enemy, Hazard, Goal.

Entities are lightweight data holders and contain minimal logic so simulation
can run without rendering. Collision geometry is always axis-aligned rectangles.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Enemy:
    x: float
    y: float
    w: int
    h: int
    vx: float = 80.0
    patrol_min_x: float = 0.0
    patrol_max_x: float = 0.0

    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def update(self, dt: float) -> None:
        # Simple patrol between min and max
        self.x += self.vx * dt
        if self.x < self.patrol_min_x:
            self.x = self.patrol_min_x
            self.vx = abs(self.vx)
        elif self.x > self.patrol_max_x:
            self.x = self.patrol_max_x
            self.vx = -abs(self.vx)


@dataclass
class Hazard:
    x: float
    y: float
    w: int
    h: int

    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class Goal:
    x: float
    y: float
    w: int
    h: int

    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)
