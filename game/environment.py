"""Gymnasium-compatible environment wrapper around the core Game class.

PlatformerEnv implements the gymnasium.Env API. It keeps the game simulation
separate from the environment wrapper and converts discrete actions into the
game's input dictionary. Observation is a flat numeric vector describing the
player, goal, nearest enemy, and nearest hazard.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .game import Game, GameState


class PlatformerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1000, seed: Optional[int] = None) -> None:
        super().__init__()
        self.game = Game(seed=seed)
        self.max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0

        # Actions: Discrete(4) -> 0=no-op, 1=left, 2=right, 3=jump
        self.action_space = spaces.Discrete(4)

        # Observation vector of 11 floats described in the spec
        # Define reasonable finite bounds for each element
        # player x: [0, 5000]
        # player y: [-1000, 2000]
        # vx, vy: [-2000, 2000]
        # grounded: [0,1]
        # goal rel x,y: [-5000,5000]
        # nearest enemy rel x,y: [-5000,5000]
        # nearest hazard rel x,y: [-5000,5000]
        low = np.array([
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
        ], dtype=np.float32)
        high = np.array([
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
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # internal tracking for reward shaping
        self._prev_goal_dist = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        # Seed the underlying game and reset
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
        # translate action to game input dict
        act = {"left": False, "right": False, "jump": False}
        if action == 1:
            act["left"] = True
        elif action == 2:
            act["right"] = True
        elif action == 3:
            act["jump"] = True

        state = self.game.step(action=act)

        obs = self._get_obs()

        reward = -0.01  # per-step penalty

        terminated = False
        truncated = False

        if state == GameState.COMPLETED:
            reward += 100.0
            terminated = True
        elif state == GameState.DEAD:
            reward -= 100.0
            terminated = True
        else:
            # progress reward: change in distance to goal (positive if getting closer)
            cur_dist = self._compute_goal_distance()
            if self._prev_goal_dist is not None:
                progress = (self._prev_goal_dist - cur_dist) * 0.01
                reward += float(progress)
            self._prev_goal_dist = cur_dist

        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True

        info = {}
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        p = self.game.player
        lvl = self.game.level
        # player
        px = float(p.x)
        py = float(p.y)
        pvx = float(p.vx)
        pvy = float(p.vy)
        grounded = 1.0 if p.grounded else 0.0

        # goal relative position (goal.x - player.x, goal.y - player.y)
        gx, gy, gw, gh = lvl.goal.bounds()
        goal_rel_x = float(gx - px)
        goal_rel_y = float(gy - py)

        # nearest enemy relative
        if len(lvl.enemies) > 0:
            nearest = None
            min_dist = float("inf")
            for e in lvl.enemies:
                ex = float(e.x)
                ey = float(e.y)
                dx = ex - px
                dy = ey - py
                d = abs(dx) + abs(dy)
                if d < min_dist:
                    min_dist = d
                    nearest = (dx, dy)
            enemy_rel_x, enemy_rel_y = nearest
        else:
            enemy_rel_x = 0.0
            enemy_rel_y = 0.0

        # nearest hazard relative
        if len(lvl.hazards) > 0:
            nearest = None
            min_dist = float("inf")
            for h in lvl.hazards:
                hx = float(h.x)
                hy = float(h.y)
                dx = hx - px
                dy = hy - py
                d = abs(dx) + abs(dy)
                if d < min_dist:
                    min_dist = d
                    nearest = (dx, dy)
            hazard_rel_x, hazard_rel_y = nearest
        else:
            hazard_rel_x = 0.0
            hazard_rel_y = 0.0

        obs = np.array([
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
        ], dtype=np.float32)

        # clip to observation space bounds to be safe
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _compute_goal_distance(self) -> float:
        p = self.game.player
        gx, gy, gw, gh = self.game.level.goal.bounds()
        # Euclidean distance between player center and goal center
        pcx = p.x + p.w / 2.0
        pcy = p.y + p.h / 2.0
        gcx = gx + gw / 2.0
        gcy = gy + gh / 2.0
        dx = gcx - pcx
        dy = gcy - pcy
        return (dx * dx + dy * dy) ** 0.5

    def close(self):
        return None
