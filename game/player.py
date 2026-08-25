"""Player class with physics, simple input interface, and collision resolution.

Player's simulation is independent from rendering. Collision geometry is an AABB
that is resolved against level platforms.
"""
from dataclasses import dataclass
from typing import Tuple

from .config import DEFAULT_CONFIG, GameConfig


@dataclass
class Player:
    x: float
    y: float
    w: int
    h: int
    vx: float = 0.0
    vy: float = 0.0
    grounded: bool = False
    config: GameConfig = DEFAULT_CONFIG

    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    # Input interface
    def apply_input(self, left: bool, right: bool, jump: bool) -> None:
        self._want_left = left
        self._want_right = right
        self._want_jump = jump

    def reset_input(self) -> None:
        self._want_left = False
        self._want_right = False
        self._want_jump = False

    def update(self, dt: float, platforms) -> None:
        # Horizontal movement
        accel = 0.0
        if getattr(self, "_want_left", False) and not getattr(self, "_want_right", False):
            accel = -self.config.accel
        elif getattr(self, "_want_right", False) and not getattr(self, "_want_left", False):
            accel = self.config.accel
        else:
            # apply friction toward zero
            if self.vx > 0:
                accel = -self.config.friction
            elif self.vx < 0:
                accel = self.config.friction

        # integrate horizontal
        self.vx += accel * dt
        # clamp speed and friction overshoot
        if not (getattr(self, "_want_left", False) or getattr(self, "_want_right", False)):
            # stop if friction would reverse velocity
            if self.vx > 0 and self.vx < 1.0:
                self.vx = 0.0
            if self.vx < 0 and self.vx > -1.0:
                self.vx = 0.0

        if self.vx > self.config.max_speed:
            self.vx = self.config.max_speed
        if self.vx < -self.config.max_speed:
            self.vx = -self.config.max_speed

        # Jumping
        if getattr(self, "_want_jump", False) and self.grounded:
            self.vy = -self.config.jump_speed
            self.grounded = False

        # Gravity
        self.vy += self.config.gravity * dt

        # Integrate
        new_x = self.x + self.vx * dt
        new_y = self.y + self.vy * dt

        # naive AABB collision resolution against platforms
        # First move horizontally then vertically
        self.x = new_x
        self._resolve_horizontal(platforms)

        self.y = new_y
        self._resolve_vertical(platforms)

        # reset one-frame inputs like jump
        self._want_jump = False

    def _resolve_horizontal(self, platforms) -> None:
        for plat in platforms:
            px, py, pw, ph = plat.bounds()
            if self._aabb_intersect(self.bounds(), (px, py, pw, ph)):
                # compute overlap on x
                if self.vx > 0:
                    # moving right, push left
                    overlap = (self.x + self.w) - px
                    self.x -= overlap
                    self.vx = 0
                elif self.vx < 0:
                    # moving left, push right
                    overlap = (px + pw) - self.x
                    self.x += overlap
                    self.vx = 0

    def _resolve_vertical(self, platforms) -> None:
        self.grounded = False
        for plat in platforms:
            px, py, pw, ph = plat.bounds()
            if self._aabb_intersect(self.bounds(), (px, py, pw, ph)):
                if self.vy > 0:
                    # falling, landed on top
                    overlap = (self.y + self.h) - py
                    self.y -= overlap
                    self.vy = 0
                    self.grounded = True
                elif self.vy < 0:
                    # hitting head
                    overlap = (py + ph) - self.y
                    self.y += overlap
                    self.vy = 0

    @staticmethod
    def _aabb_intersect(a, b) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw <= bx or ax >= bx + bw or ay + ah <= by or ay >= by + bh)

    # Optional rendering (kept out of core simulation); imports pygame only here
    def render(self, surface) -> None:
        try:
            import pygame
        except Exception:
            return
        rect = pygame.Rect(int(self.x), int(self.y), self.w, self.h)
        pygame.draw.rect(surface, (50, 150, 250), rect)
