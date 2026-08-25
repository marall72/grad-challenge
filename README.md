Simple 2D platformer core engine (minimal)

This repository contains a small, original 2D platformer core suitable for
wrapping later as a Gymnasium environment. The implementation focuses on
simulation separate from rendering and supports deterministic seeds and
headless execution.

Project layout:

project/
├── game/             # core engine
│   ├── __init__.py
│   ├── config.py
│   ├── game.py
│   ├── player.py
│   ├── level.py
│   └── entities.py
├── tests/            # tests go here
├── requirements.txt
└── README.md

Notes:
- Do not import pygame in simulation code; rendering functions import pygame
  only when called.
- Use Game.reset(seed=...) for deterministic episodes.

installing requirements: python -m pip install -r requirements.txt