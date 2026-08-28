"""Core Game class: holds game state, updates simulation, and manages reset/seed."""

from enum import Enum, auto
from typing import Optional, Dict
import random

from .config import DEFAULT_CONFIG
from .player import Player
from .level import Level


class GameState(Enum):
    RUNNING = auto()
    DEAD = auto()
    COMPLETED = auto()


class Game:

    def __init__(
        self,
        config=DEFAULT_CONFIG,
        seed: Optional[int] = None,
    ) -> None:

        self.config = config
        self._seed = seed
        self.rng = random.Random(seed)

        self.level: Optional[Level] = None
        self.player: Optional[Player] = None

        self.state = GameState.RUNNING
        self.time = 0.0
        self.death_reason = None

        self.reset(seed=seed)

    # =============================================================
    # RESET
    # =============================================================

    def reset(self, seed: Optional[int] = None) -> None:

        if seed is not None:
            self._seed = seed

        self.rng = random.Random(self._seed)

        self.level = Level.sample(self._seed)

        sx, sy = self.level.pick_spawn()

        self.player = Player(
            x=sx,
            y=sy,
            w=self.config.player_width,
            h=self.config.player_height,
            grounded=True,
            config=self.config,
        )

        self.state = GameState.RUNNING
        self.time = 0.0
        self.death_reason = None

    # =============================================================
    # STEP
    # =============================================================

    def step(
        self,
        action: Optional[Dict[str, bool]] = None,
        dt: Optional[float] = None,
    ) -> GameState:

        if dt is None:
            dt = self.config.time_step

        if self.state != GameState.RUNNING:
            return self.state

        action = action or {}

        left = bool(action.get("left", False))
        right = bool(action.get("right", False))
        jump = bool(action.get("jump", False))

        # Apply inputs
        self.player.apply_input(
            left=left,
            right=right,
            jump=jump,
        )

        # Update enemies
        for enemy in self.level.enemies:
            enemy.update(dt)

        # Update player
        self.player.update(
            dt,
            self.level.platforms,
        )

        # ---------------------------------------------------------
        # Hazard collision
        # ---------------------------------------------------------

        px, py, pw, ph = self.player.bounds()

        for hazard in self.level.hazards:

            hx, hy, hw, hh = hazard.bounds()

            if not (
                px + pw <= hx
                or px >= hx + hw
                or py + ph <= hy
                or py >= hy + hh
            ):

                self.state = GameState.DEAD
                self.death_reason = "HAZARD"

                return self.state

        # ---------------------------------------------------------
        # Enemy collision
        # ---------------------------------------------------------

        for enemy in self.level.enemies:

            ex, ey, ew, eh = enemy.bounds()

            if not (
                px + pw <= ex
                or px >= ex + ew
                or py + ph <= ey
                or py >= ey + eh
            ):

                self.state = GameState.DEAD
                self.death_reason = "ENEMY"

                return self.state

        # ---------------------------------------------------------
        # Goal collision
        # ---------------------------------------------------------

        gx, gy, gw, gh = self.level.goal.bounds()

        if not (
            px + pw <= gx
            or px >= gx + gw
            or py + ph <= gy
            or py >= gy + gh
        ):

            self.state = GameState.COMPLETED

            return self.state

        self.time += dt

        return self.state

    # =============================================================
    # RENDER
    # =============================================================

    def render(
        self,
        surface,
        camera_x: float = 0.0,
        camera_y: float = 0.0,
    ) -> None:

        try:
            import pygame
        except Exception:
            return

        # ---------------------------------------------------------
        # Background
        # ---------------------------------------------------------

        surface.fill(
            (135, 206, 235)
        )

        # ---------------------------------------------------------
        # Platforms
        # ---------------------------------------------------------

        for platform in self.level.platforms:

            x, y, w, h = platform.bounds()

            screen_x = int(x - camera_x)
            screen_y = int(y - camera_y)

            pygame.draw.rect(
                surface,
                (100, 100, 100),
                pygame.Rect(
                    screen_x,
                    screen_y,
                    int(w),
                    int(h),
                ),
            )

        # ---------------------------------------------------------
        # Hazards
        # ---------------------------------------------------------

        for hazard in self.level.hazards:

            x, y, w, h = hazard.bounds()

            screen_x = int(x - camera_x)
            screen_y = int(y - camera_y)

            pygame.draw.rect(
                surface,
                (200, 50, 50),
                pygame.Rect(
                    screen_x,
                    screen_y,
                    int(w),
                    int(h),
                ),
            )

            # Draw hazard label
            font = pygame.font.Font(None, 20)

            label = font.render(
                "HAZARD",
                True,
                (255, 255, 255),
            )

            surface.blit(
                label,
                (
                    screen_x,
                    screen_y - 20,
                ),
            )

        # ---------------------------------------------------------
        # Enemies
        # ---------------------------------------------------------

        for enemy in self.level.enemies:

            x, y, w, h = enemy.bounds()

            screen_x = int(x - camera_x)
            screen_y = int(y - camera_y)

            pygame.draw.rect(
                surface,
                (180, 30, 30),
                pygame.Rect(
                    screen_x,
                    screen_y,
                    int(w),
                    int(h),
                ),
            )

        # ---------------------------------------------------------
        # Goal
        # ---------------------------------------------------------

        gx, gy, gw, gh = self.level.goal.bounds()

        goal_x = int(gx - camera_x)
        goal_y = int(gy - camera_y)

        # Goal pole
        pygame.draw.rect(
            surface,
            (40, 120, 40),
            pygame.Rect(
                goal_x,
                goal_y,
                6,
                int(gh),
            ),
        )

        # Goal flag
        pygame.draw.polygon(
            surface,
            (30, 180, 60),
            [
                (goal_x + 6, goal_y),
                (goal_x + 50, goal_y + 15),
                (goal_x + 6, goal_y + 30),
            ],
        )

        # Goal base
        pygame.draw.rect(
            surface,
            (40, 120, 40),
            pygame.Rect(
                goal_x - 8,
                goal_y + int(gh) - 8,
                22,
                8,
            ),
        )

        # Goal label
        font = pygame.font.Font(None, 28)

        goal_label = font.render(
            "GOAL",
            True,
            (0, 100, 0),
        )

        surface.blit(
            goal_label,
            (
                goal_x - 10,
                goal_y - 32,
            ),
        )

        # ---------------------------------------------------------
        # Player
        # ---------------------------------------------------------

        # Player.render() uses its own world coordinates,
        # so we draw the player manually with camera offset.

        p = self.player

        player_x = int(p.x - camera_x)
        player_y = int(p.y - camera_y)

        pygame.draw.rect(
            surface,
            (40, 80, 220),
            pygame.Rect(
                player_x,
                player_y,
                int(p.w),
                int(p.h),
            ),
        )