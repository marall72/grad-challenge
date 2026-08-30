"""Gymnasium environment for training PPO on the platformer game."""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game.config import DEFAULT_CONFIG

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

        # =========================================================
        # ACTIONS
        # =========================================================
        #
        # 0 = NO-OP
        # 1 = LEFT
        # 2 = RIGHT
        # 3 = JUMP
        # 4 = RIGHT + JUMP
        # 5 = LEFT + JUMP
        #
        self.action_space = spaces.Discrete(6)

        # =========================================================
        # OBSERVATION
        # =========================================================
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
        # 7  enemy relative x
        # 8  enemy relative y
        #
        # 9  hazard relative x
        # 10 hazard relative y
        #
        # 11 next platform relative x
        # 12 next platform relative y
        # 13 next platform width
        # 14 next platform distance
        #
        # 15 hazard left edge relative x
        # 16 hazard right edge relative x
        # 17 horizontal distance to hazard
        #
        # 18 enemy left edge relative x
        # 19 enemy right edge relative x
        # 20 horizontal distance to enemy
        #

        low = [
                0.0,       # player x
                -1000.0,   # player y
                -2000.0,   # player vx
                -2000.0,   # player vy
                0.0,       # grounded

                -5000.0,   # goal relative x
                -5000.0,   # goal relative y

                -5000.0,   # hazard relative x
                -5000.0,   # hazard relative y

                -5000.0,   # platform relative x
                -5000.0,   # platform relative y
                0.0,       # platform width
                0.0,       # platform distance

                -5000.0,   # hazard left
                -5000.0,   # hazard right
                0.0,       # hazard horizontal distance
            ]

        high = [
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
                10000.0,

                5000.0,
                5000.0,
                5000.0,
            ]

        if DEFAULT_CONFIG.use_enemy_observation:
            low.extend([
                -5000.0,   # enemy relative x
                -5000.0,   # enemy relative y
                -5000.0,   # enemy left
                -5000.0,   # enemy right
                0.0,       # enemy horizontal distance
            ])

            high.extend([
                5000.0,   # enemy relative x
                5000.0,   # enemy relative y
                5000.0,    # enemy left
                5000.0,    # enemy right
                5000.0,    # enemy horizontal distance
            ])

        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

        # =========================================================
        # REWARD STATE
        # =========================================================

        self._prev_goal_dist: Optional[float] = None
        self._prev_player_x: Optional[float] = None

        self._passed_hazards = set()
        self._hazard_attempted = set()

        self._passed_enemies = set()
        self._enemy_attempted = set()

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:

        super().reset(seed=seed)

        if seed is not None:
            self.game.reset(seed=seed)
        else:
            self.game.reset()

        self._elapsed_steps = 0

        self._prev_goal_dist = self._compute_goal_distance()
        self._prev_player_x = float(self.game.player.x)

        self._passed_hazards = set()
        self._hazard_attempted = set()

        self._passed_enemies = set()
        self._enemy_attempted = set()

        obs = self._get_obs()

        return obs, {
            "seed": self.game._seed,
        }

    # ============================================================
    # STEP
    # ============================================================

    def step(self, action: int):

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # ---------------------------------------------------------
        # Convert discrete action to game input
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # State before simulation
        # ---------------------------------------------------------

        player_before = self.game.player

        previous_x = float(player_before.x)
        previous_y = float(player_before.y)
        previous_grounded = bool(player_before.grounded)

        previous_goal_dist = self._prev_goal_dist

        # ---------------------------------------------------------
        # Advance game
        # ---------------------------------------------------------

        state = self.game.step(action=act)

        # ---------------------------------------------------------
        # Observation
        # ---------------------------------------------------------

        obs = self._get_obs()

        # ---------------------------------------------------------
        # Base reward
        # ---------------------------------------------------------

        reward = -0.02

        terminated = False
        truncated = False

        # =========================================================
        # SUCCESS
        # =========================================================

        if state == GameState.COMPLETED:

            reward += 100.0
            terminated = True

        # =========================================================
        # DEATH
        # =========================================================

        elif state == GameState.DEAD:

            reward -= 100.0
            terminated = True

        # =========================================================
        # ALIVE
        # =========================================================

        else:

            p = self.game.player

            current_x = float(p.x)
            current_y = float(p.y)

            dx = current_x - previous_x

            if (DEFAULT_CONFIG.use_movement_reward):
                # =====================================================
                # 1. REAL HORIZONTAL MOVEMENT
                # =====================================================
                #is moving right
                if dx > 0.0:

                    reward += min(
                        dx * 0.035,
                        0.40,
                    )

                #is standing still
                elif dx == 0:
                    reward -= 1

                #moved backwards
                elif dx < 0.0:
                    reward -= min(
                        abs(dx) * 0.040,
                        0.55,
                    )

                #did right jump
                if action == 4 and dx > 0.0:
                    reward += 0.035

            # =====================================================
            # 3. GOAL PROGRESS
            # =====================================================

            current_goal_dist = self._compute_goal_distance()

            if previous_goal_dist is not None:

                goal_progress = (
                    previous_goal_dist
                    - current_goal_dist
                )

                reward += goal_progress * 0.008

            self._prev_goal_dist = current_goal_dist

            # =====================================================
            # 4. HAZARD HANDLING
            # =====================================================
            player_left = float(p.x)
            player_right = float(p.x + p.w)
            if(DEFAULT_CONFIG.use_hazard_reward):
                for index, hazard in enumerate(
                    self.game.level.hazards
                ):

                    if index in self._passed_hazards:
                        continue

                    hx, hy, hw, hh = hazard.bounds()

                    hazard_left = float(hx)
                    hazard_right = float(hx + hw)

                    distance_to_hazard = (
                        hazard_left - player_right
                    )

                    player_over_hazard = (
                        player_right > hazard_left
                        and player_left < hazard_right
                    )

                    completely_past_hazard = (
                        player_left > hazard_right
                    )

                    # -------------------------------------------------
                    # Hazard attempt
                    # -------------------------------------------------

                    if (
                        distance_to_hazard >= 0.0
                        and distance_to_hazard <= 160.0
                    ):

                        self._hazard_attempted.add(index)

                # -------------------------------------------------
                # Approaching hazard
                # -------------------------------------------------

                    if index in self._hazard_attempted:

                        if (
                            distance_to_hazard > 0.0
                            and dx > 0.0
                        ):

                            reward += min(
                                dx * 0.02,
                                0.2,
                            )

                        if (
                            distance_to_hazard > 0.0
                            and dx < 0.0
                        ):

                            reward -= min(
                                abs(dx) * 0.045,
                                0.40,
                            )

                    # -------------------------------------------------
                    # Jumping above hazard
                    # -------------------------------------------------

                    if (
                        index in self._hazard_attempted
                        and player_over_hazard
                        and not p.grounded
                    ):

                        crossing_key = (
                            index,
                            "airborne",
                        )

                        if crossing_key not in self._passed_hazards:

                            reward += 0.2

                    # -------------------------------------------------
                    # Successful crossing
                    # -------------------------------------------------

                    if completely_past_hazard:

                        self._passed_hazards.add(index)

                        reward += 36.0

                    # -------------------------------------------------
                    # Do not turn around above hazard
                    # -------------------------------------------------

                    if (
                        index in self._hazard_attempted
                        and player_over_hazard
                        and not p.grounded
                        and dx < 0.0
                    ):

                        reward -= abs(dx) * 0.060

                    # -------------------------------------------------
                    # Standing immediately before hazard
                    # -------------------------------------------------

                    if (
                        p.grounded
                        and 0.0 <= distance_to_hazard <= 70.0
                    ):

                        reward -= 0.08

                    # -------------------------------------------------
                    # Encourage RIGHT + JUMP near hazard
                    # -------------------------------------------------

                    if (
                        action == 4 and distance_to_hazard <= 70.0
                    ):

                        reward += 0.3

            # =====================================================
            # 5. ENEMY HANDLING
            # =====================================================

            for index, enemy in enumerate(
                self.game.level.enemies
            ):

                if index in self._passed_enemies:
                    continue

                ex, ey, ew, eh = enemy.bounds()

                enemy_left = float(ex)
                enemy_right = float(ex + ew)

                # -------------------------------------------------
                # Horizontal distance to enemy
                # -------------------------------------------------

                distance_to_enemy = (
                    enemy_left - player_right
                )

                player_over_enemy = (
                    player_right > enemy_left
                    and player_left < enemy_right
                )

                completely_past_enemy = (
                    player_left > enemy_right
                )

                # -------------------------------------------------
                # Enemy attempt
                # -------------------------------------------------

                if (
                    distance_to_enemy >= 0.0
                    and distance_to_enemy <= 180.0
                ):

                    self._enemy_attempted.add(index)

                # -------------------------------------------------
                # Enemy is ahead
                # -------------------------------------------------

                if index in self._enemy_attempted:

                    # Keep moving toward the enemy.
                    # Do not reward stopping.

                    if (
                        distance_to_enemy > 0.0
                        and dx > 0.0
                    ):

                        reward += min(
                            dx * 0.010,
                            0.10,
                        )

                    # Strong penalty for simply stopping
                    # directly before an enemy.

                    if (
                        p.grounded
                        and 0.0 <= distance_to_enemy <= 70.0
                    ):

                        reward -= 0.15

                    # -------------------------------------------------
                    # Encourage jumping when enemy is close
                    # -------------------------------------------------

                    if (
                        0.0 <= distance_to_enemy <= 70.0
                        and action == 4
                    ):

                        reward += 0.30

                # -------------------------------------------------
                # Player is crossing enemy horizontally
                # -------------------------------------------------

                if (
                    index in self._enemy_attempted
                    and player_over_enemy
                    and not p.grounded
                ):

                    reward += 0.15

                # -------------------------------------------------
                # Successfully passed enemy
                # -------------------------------------------------

                if completely_past_enemy:

                    self._passed_enemies.add(index)

                    reward += 25.0

                # -------------------------------------------------
                # Do not turn around while crossing enemy
                # -------------------------------------------------

                if (
                    index in self._enemy_attempted
                    and player_over_enemy
                    and not p.grounded
                    and dx < 0.0
                ):

                    reward -= abs(dx) * 0.050

            # =====================================================
            # 6. UPDATE PREVIOUS X
            # =====================================================

            self._prev_player_x = current_x

        # =========================================================
        # STEP COUNTER
        # =========================================================

        self._elapsed_steps += 1

        # =========================================================
        # TIME LIMIT
        # =========================================================

        if self._elapsed_steps >= self.max_episode_steps:

            truncated = True

        # =========================================================
        # INFO
        # =========================================================

        info = {
            "game_state": self.game.state,
            "death_reason": self.game.death_reason,
            "goal_distance": self._compute_goal_distance(),
            "elapsed_steps": self._elapsed_steps,
            "passed_hazards": len(
                self._passed_hazards
            ),
            "total_hazards": len(
                self.game.level.hazards
            ),
            "passed_enemies": len(
                self._passed_enemies
            ),
            "total_enemies": len(
                self.game.level.enemies
            ),
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ============================================================
    # OBSERVATION
    # ============================================================

    def _get_obs(self) -> np.ndarray:

        p = self.game.player
        lvl = self.game.level

        px = float(p.x)
        py = float(p.y)
        pvx = float(p.vx)
        pvy = float(p.vy)

        grounded = (
            1.0
            if p.grounded
            else 0.0
        )

        # =========================================================
        # GOAL
        # =========================================================

        gx, gy, gw, gh = lvl.goal.bounds()

        goal_rel_x = float(gx - px)
        goal_rel_y = float(gy - py)

        # =========================================================
        # NEAREST ENEMY
        # =========================================================

        nearest_enemy = None
        nearest_enemy_dist = float("inf")

        player_center_x = (
            px + float(p.w) / 2.0
        )

        player_center_y = (
            py + float(p.h) / 2.0
        )

        for enemy in lvl.enemies:

            ex, ey, ew, eh = enemy.bounds()

            enemy_center_x = (
                float(ex) + float(ew) / 2.0
            )

            enemy_center_y = (
                float(ey) + float(eh) / 2.0
            )

            dx = (
                enemy_center_x
                - player_center_x
            )

            dy = (
                enemy_center_y
                - player_center_y
            )

            distance = abs(dx) + abs(dy)

            if distance < nearest_enemy_dist:

                nearest_enemy_dist = distance

                nearest_enemy = (
                    float(ex),
                    float(ey),
                    float(ew),
                    float(eh),
                )

        if nearest_enemy is not None:

            ex, ey, ew, eh = nearest_enemy

            enemy_rel_x = (
                float(ex) - px
            )

            enemy_rel_y = (
                float(ey) - py
            )

            enemy_left_rel_x = (
                float(ex) - px
            )

            enemy_right_rel_x = (
                float(ex + ew) - px
            )

            player_right = (
                px + float(p.w)
            )

            if player_right < ex:

                enemy_horizontal_distance = (
                    float(ex) - player_right
                )

            elif px > ex + ew:

                enemy_horizontal_distance = (
                    px - float(ex + ew)
                )

            else:

                enemy_horizontal_distance = 0.0

        else:

            enemy_rel_x = 0.0
            enemy_rel_y = 0.0
            enemy_left_rel_x = 0.0
            enemy_right_rel_x = 0.0
            enemy_horizontal_distance = 0.0

        # =========================================================
        # NEAREST HAZARD
        # =========================================================

        nearest_hazard = None
        nearest_hazard_dist = float("inf")

        for hazard in lvl.hazards:

            hx, hy, hw, hh = hazard.bounds()

            hazard_center_x = (
                float(hx)
                + float(hw) / 2.0
            )

            hazard_center_y = (
                float(hy)
                + float(hh) / 2.0
            )

            dx = (
                hazard_center_x
                - player_center_x
            )

            dy = (
                hazard_center_y
                - player_center_y
            )

            distance = abs(dx) + abs(dy)

            if distance < nearest_hazard_dist:

                nearest_hazard_dist = distance

                nearest_hazard = (
                    float(hx),
                    float(hy),
                    float(hw),
                    float(hh),
                )

        if nearest_hazard is not None:

            hx, hy, hw, hh = nearest_hazard

            hazard_rel_x = (
                float(hx) - px
            )

            hazard_rel_y = (
                float(hy) - py
            )

            hazard_left_rel_x = (
                float(hx) - px
            )

            hazard_right_rel_x = (
                float(hx + hw) - px
            )

            player_right = (
                px + float(p.w)
            )

            if player_right < hx:

                horizontal_distance = (
                    float(hx) - player_right
                )

            elif px > hx + hw:

                horizontal_distance = (
                    px - float(hx + hw)
                )

            else:

                horizontal_distance = 0.0

        else:

            hazard_rel_x = 0.0
            hazard_rel_y = 0.0
            hazard_left_rel_x = 0.0
            hazard_right_rel_x = 0.0
            horizontal_distance = 0.0

        # =========================================================
        # NEXT PLATFORM
        # =========================================================

        next_platform = self._get_next_platform(px)

        if next_platform is not None:

            (
                platform_x,
                platform_y,
                platform_w,
                platform_h,
            ) = next_platform.bounds()

            platform_rel_x = (
                float(platform_x) - px
            )

            platform_rel_y = (
                float(platform_y) - py
            )

            platform_width = float(
                platform_w
            )

            platform_center_x = (
                float(platform_x)
                + float(platform_w) / 2.0
            )

            platform_center_y = (
                float(platform_y)
                + float(platform_h) / 2.0
            )

            player_center_x = (
                px + float(p.w) / 2.0
            )

            player_center_y = (
                py + float(p.h) / 2.0
            )

            dx = (
                platform_center_x
                - player_center_x
            )

            dy = (
                platform_center_y
                - player_center_y
            )

            platform_distance = float(
                np.sqrt(
                    dx * dx + dy * dy
                )
            )

        else:

            platform_rel_x = 0.0
            platform_rel_y = 0.0
            platform_width = 0.0
            platform_distance = 0.0

        # =========================================================
        # BUILD OBSERVATION
        # =========================================================

        obs = [
                px,
                py,
                pvx,
                pvy,
                grounded,

                goal_rel_x,
                goal_rel_y,

                hazard_rel_x,
                hazard_rel_y,

                platform_rel_x,
                platform_rel_y,
                platform_width,
                platform_distance,

                hazard_left_rel_x,
                hazard_right_rel_x,
                horizontal_distance,
            ]

        if DEFAULT_CONFIG.use_enemy_observation:
            obs.extend([
                enemy_rel_x,
                enemy_rel_y,
                enemy_left_rel_x,
                enemy_right_rel_x,
                enemy_horizontal_distance,
            ])

        obs = np.array(obs, dtype=np.float32)

        return np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high,
        )

    # ============================================================
    # NEXT PLATFORM
    # ============================================================

    def _get_next_platform(
        self,
        player_x: float,
    ):

        candidates = []

        for platform in self.game.level.platforms:

            if platform.x > player_x + 1.0:

                candidates.append(platform)

        if not candidates:

            return None

        candidates.sort(
            key=lambda p: p.x
        )

        return candidates[0]

    # ============================================================
    # GOAL DISTANCE
    # ============================================================

    def _compute_goal_distance(self) -> float:

        p = self.game.player

        gx, gy, gw, gh = (
            self.game.level.goal.bounds()
        )

        player_center_x = (
            p.x + p.w / 2.0
        )

        goal_center_x = (
            gx + gw / 2.0
        )

        dx = (
            goal_center_x
            - player_center_x
        )

        return float(abs(dx))

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self):
        return None