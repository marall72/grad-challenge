"""Game package for the simple 2D platformer core engine.

Keep rendering optional and separate from simulation logic.
"""

from .game import Game
from .player import Player
from .level import Level
from .entities import Enemy, Hazard, Goal

__all__ = ["Game", "Player", "Level", "Enemy", "Hazard", "Goal"]
