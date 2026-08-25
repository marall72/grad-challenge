"""Configuration constants and simple typed config for the game."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    gravity: float = 1500.0  # px/s^2
    accel: float = 2000.0  # horizontal acceleration px/s^2
    max_speed: float = 300.0  # horizontal max speed px/s
    jump_speed: float = 650.0  # initial jump velocity px/s
    friction: float = 1800.0  # deceleration when no input px/s^2
    player_width: int = 32
    player_height: int = 48
    time_step: float = 1.0 / 60.0


DEFAULT_CONFIG = GameConfig()
