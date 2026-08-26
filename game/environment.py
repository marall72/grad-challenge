"""Gymnasium-compatible environment wrapper around the core Game class.

PlatformerEnv converts the game's input system into a discrete RL action space
and exposes a numeric observation containing player state, goal information,
nearby hazards/enemies, and the next platform.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .game import Game, GameState


class PlatformerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.game = Game(seed=seed)
        self.max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0

        # Actions:
        # 0 = no-op
        # 1 = left
        # 2 = right
        # 3 = jump
        # 4 = right + jump
        # 5 = left + jump
        self.action_space = spaces.Discrete(6)

        # Observation:
        #
        # 0  player x
        # 1  player y
        # 2  player vx
        # 3  player vy
        # 4  grounded
        #
        # 5  goal relative x
        # 6  goal relative y
        #
        # 7  nearest enemy relative x
        # 8  nearest enemy relative y
        #
        # 9  nearest hazard relative x
        # 10 nearest hazard relative y
        #
        # 11 next platform relative x
        # 12 next platform relative y
        # 13 next platform width
        #
        # 14 distance to next platform
        #
        # Total = 15 values
        low = np.array(
            [
                0.0,
                -1000.0,
                -2000.0,
                -2000.0,
                0.0,
                -5000.0,
                -5000.0,
                -5000.0,
                -5000.0,
                -5000.0,
                -5000.0,
                -5000.0,
                -5000.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        high = np.array(
            [
                5000.0,
                2000.0,
                2000.0,
                2000.0,
                1.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                5000.0,
                10000.0,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

        self._prev_goal_dist: Optional[float] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:

        if seed is not None:
            self.game.reset(seed=seed)
        else:
            self.game.reset()

        self._elapsed_steps = 0
        self._prev_goal_dist = self._compute_goal_distance()

        obs = self._get_obs()

        return obs, {"seed": self.game._seed}

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        # Convert discrete action to game input.
        act = {
            "left": False,
            "right": False,
            "jump": False,
        }
        
        if action == 1:
            act["left"] = True
        
        elif action == 2:
            act["right"] = True
        
        elif action == 3:
            act["jump"] = True
        
        elif action == 4:
            act["right"] = True
            act["jump"] = True
        
        elif action == 5:
            act["left"] = True
            act["jump"] = True

        # Advance game simulation.
        state = self.game.step(action=act)

        obs = self._get_obs()

        # Small penalty for every time step.
        reward = -0.01

        terminated = False
        truncated = False

        if state == GameState.COMPLETED:
            # Strong positive reward for completing the level.
            reward += 100.0
            terminated = True

        elif state == GameState.DEAD:
            # Strong negative reward for dying.
            reward -= 100.0
            terminated = True

        else:
            # Reward progress toward the goal.
            current_goal_dist = self._compute_goal_distance()

            if self._prev_goal_dist is not None:
                progress = self._prev_goal_dist - current_goal_dist

                # Stronger signal than the previous 0.01 coefficient.
                reward += float(progress * 0.05)

            self._prev_goal_dist = current_goal_dist

        self._elapsed_steps += 1

        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True

        info = {}

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _get_obs(self) -> np.ndarray:
        p = self.game.player
        lvl = self.game.level

        # ---------------------------------------------------------
        # Player
        # ---------------------------------------------------------

        px = float(p.x)
        py = float(p.y)
        pvx = float(p.vx)
        pvy = float(p.vy)

        grounded = 1.0 if p.grounded else 0.0

        # ---------------------------------------------------------
        # Goal
        # ---------------------------------------------------------

        gx, gy, gw, gh = lvl.goal.bounds()

        goal_rel_x = float(gx - px)
        goal_rel_y = float(gy - py)

        # ---------------------------------------------------------
        # Nearest enemy
        # ---------------------------------------------------------

        if len(lvl.enemies) > 0:

            nearest_enemy = None
            min_dist = float("inf")

            for enemy in lvl.enemies:

                ex = float(enemy.x)
                ey = float(enemy.y)

                dx = ex - px
                dy = ey - py

                distance = abs(dx) + abs(dy)

                if distance < min_dist:
                    min_dist = distance
                    nearest_enemy = (dx, dy)

            enemy_rel_x, enemy_rel_y = nearest_enemy

        else:
            enemy_rel_x = 0.0
            enemy_rel_y = 0.0

        # ---------------------------------------------------------
        # Nearest hazard
        # ---------------------------------------------------------

        if len(lvl.hazards) > 0:

            nearest_hazard = None
            min_dist = float("inf")

            for hazard in lvl.hazards:

                hx = float(hazard.x)
                hy = float(hazard.y)

                dx = hx - px
                dy = hy - py

                distance = abs(dx) + abs(dy)

                if distance < min_dist:
                    min_dist = distance
                    nearest_hazard = (dx, dy)

            hazard_rel_x, hazard_rel_y = nearest_hazard

        else:
            hazard_rel_x = 0.0
            hazard_rel_y = 0.0

        # ---------------------------------------------------------
        # Next platform
        # ---------------------------------------------------------

        next_platform = self._get_next_platform(px)

        if next_platform is not None:

            platform_x, platform_y, platform_w, platform_h = (
                next_platform.bounds()
            )

            platform_rel_x = float(platform_x - px)
            platform_rel_y = float(platform_y - py)
            platform_width = float(platform_w)

            platform_center_x = platform_x + platform_w / 2.0
            platform_center_y = platform_y + platform_h / 2.0

            player_center_x = px + p.w / 2.0
            player_center_y = py + p.h / 2.0

            dx = platform_center_x - player_center_x
            dy = platform_center_y - player_center_y

            platform_distance = float(np.sqrt(dx * dx + dy * dy))

        else:

            platform_rel_x = 0.0
            platform_rel_y = 0.0
            platform_width = 0.0
            platform_distance = 0.0

        # ---------------------------------------------------------
        # Observation
        # ---------------------------------------------------------

        obs = np.array(
            [
                px,
                py,
                pvx,
                pvy,
                grounded,
                goal_rel_x,
                goal_rel_y,
                enemy_rel_x,
                enemy_rel_y,
                hazard_rel_x,
                hazard_rel_y,
                platform_rel_x,
                platform_rel_y,
                platform_width,
                platform_distance,
            ],
            dtype=np.float32,
        )

        return np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high,
        )

    def _get_next_platform(self, player_x: float):
        """Return the nearest platform that is ahead of the player."""

        candidates = []

        for platform in self.game.level.platforms:

            # Platform starts ahead of the player.
            if platform.x > player_x + 1.0:
                candidates.append(platform)

        if not candidates:
            return None

        # Choose the closest platform ahead.
        candidates.sort(key=lambda p: p.x)

        return candidates[0]

    def _compute_goal_distance(self) -> float:
        p = self.game.player

        gx, gy, gw, gh = self.game.level.goal.bounds()

        player_center_x = p.x + p.w / 2.0
        player_center_y = p.y + p.h / 2.0

        goal_center_x = gx + gw / 2.0
        goal_center_y = gy + gh / 2.0

        dx = goal_center_x - player_center_x
        dy = goal_center_y - player_center_y

        return float(np.sqrt(dx * dx + dy * dy))

    def close(self):
        return None