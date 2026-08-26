"""Core Game class: holds game state, updates simulation, and manages reset/seed.

The Game class purposely separates simulation from rendering. The step() method
advances physics and game rules and returns the current episode state.
"""
from enum import Enum, auto
from typing import Optional, Dict
import random

from .config import DEFAULT_CONFIG
from .player import Player
from .level import Level
from .entities import Enemy, Hazard, Goal


class GameState(Enum):
    RUNNING = auto()
    DEAD = auto()
    COMPLETED = auto()


class Game:
    def __init__(self, config=DEFAULT_CONFIG, seed: Optional[int] = None) -> None:
        self.config = config
        self._seed = seed
        self.rng = random.Random(seed)
        self.level: Optional[Level] = None
        self.player: Optional[Player] = None
        self.state = GameState.RUNNING
        self.time = 0.0
        self.death_reason = None
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        # Allow overriding seed on reset
        if seed is not None:
            self._seed = seed
        self.rng = random.Random(self._seed)
        # Build level deterministically from seed
        self.level = Level.sample(self._seed)
        sx, sy = self.level.pick_spawn()
        self.player = Player(x=sx, y=sy, w=self.config.player_width, h=self.config.player_height, config=self.config)
        self.state = GameState.RUNNING
        self.time = 0.0
        self.death_reason = None

    def step(self, action: Optional[Dict[str, bool]] = None, dt: Optional[float] = None) -> GameState:
        """Advance simulation by dt seconds applying the given action.

        action: dict with keys 'left', 'right', 'jump' boolean values.
        dt: seconds to advance; defaults to config.time_step.
        Returns the current GameState after the step.
        """
        if dt is None:
            dt = self.config.time_step
        if self.state != GameState.RUNNING:
            return self.state

        action = action or {}
        left = bool(action.get("left", False))
        right = bool(action.get("right", False))
        jump = bool(action.get("jump", False))

        # apply inputs
        self.player.apply_input(left=left, right=right, jump=jump)

        # update enemies
        for e in self.level.enemies:
            e.update(dt)

        # update player physics and collisions
        self.player.update(dt, self.level.platforms)

        # check collisions with hazards
        px, py, pw, ph = self.player.bounds()
        # check hazards
        for h in self.level.hazards:
            hx, hy, hw, hh = h.bounds()
            if not (px + pw <= hx or px >= hx + hw or py + ph <= hy or py >= hy + hh):
                self.state = GameState.DEAD
                self.death_reason = "HAZARD"
                return self.state

        # check collisions with enemies (simple bounding box)
        for en in self.level.enemies:
            ex, ey, ew, eh = en.bounds()
            if not (px + pw <= ex or px >= ex + ew or py + ph <= ey or py >= ey + eh):
                # simple rule: touching enemy kills the player
                self.state = GameState.DEAD
                self.death_reason = "ENEMY"
                return self.state

        # check goal
        gx, gy, gw, gh = self.level.goal.bounds()
        if not (px + pw <= gx or px >= gx + gw or py + ph <= gy or py >= gy + gh):
            self.state = GameState.COMPLETED
            return self.state

        # keep time
        self.time += dt
        return self.state

    # Optional rendering helpers; these import pygame only if used
    def render(self, surface) -> None:
        try:
            import pygame
        except Exception:
            return
        # background
        surface.fill((135, 206, 235))
        # draw platforms
        for plat in self.level.platforms:
            x, y, w, h = plat.bounds()
            pygame.draw.rect(surface, (100, 100, 100), pygame.Rect(int(x), int(y), int(w), int(h)))
        # draw hazards
        for hz in self.level.hazards:
            x, y, w, h = hz.bounds()
            pygame.draw.rect(surface, (200, 50, 50), pygame.Rect(int(x), int(y), int(w), int(h)))
        # draw enemies
        for en in self.level.enemies:
            x, y, w, h = en.bounds()
            pygame.draw.rect(surface, (180, 30, 30), pygame.Rect(int(x), int(y), int(w), int(h)))
        # draw goal
        gx, gy, gw, gh = self.level.goal.bounds()
        pygame.draw.rect(surface, (250, 200, 50), pygame.Rect(int(gx), int(gy), int(gw), int(gh)))
        # draw player
        self.player.render(surface)
