import pygame

from game.game import Game
from game.config import DEFAULT_CONFIG, GameConfig

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

FPS = 60

# Player stays around this position on screen.
CAMERA_OFFSET_X = 250


def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT)
    )

    pygame.display.set_caption(
        "RL Platformer"
    )

    clock = pygame.time.Clock()

    game = Game(seed=DEFAULT_CONFIG.seed)

    running = True

    while running:

        dt = clock.tick(FPS) / 1000.0

        # =========================================================
        # EVENTS
        # =========================================================

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        # =========================================================
        # INPUT
        # =========================================================

        keys = pygame.key.get_pressed()

        action = {
            "left": (
                keys[pygame.K_LEFT]
                or keys[pygame.K_a]
            ),

            "right": (
                keys[pygame.K_RIGHT]
                or keys[pygame.K_d]
            ),

            "jump": (
                keys[pygame.K_SPACE]
                or keys[pygame.K_UP]
            ),
        }

        # =========================================================
        # UPDATE
        # =========================================================

        game.step(
            action,
            dt,
        )

        # =========================================================
        # CAMERA
        # =========================================================

        camera_x = (
            game.player.x
            - CAMERA_OFFSET_X
        )

        # Never move camera before x=0.
        camera_x = max(
            0.0,
            camera_x,
        )

        # =========================================================
        # RENDER
        # =========================================================

        game.render(
            screen,
            camera_x=camera_x,
            camera_y=0.0,
        )

        # =========================================================
        # UI
        # =========================================================

        font = pygame.font.Font(
            None,
            24,
        )

        # Player X
        player_text = font.render(
            f"Player X: {game.player.x:.0f}",
            True,
            (0, 0, 0),
        )

        screen.blit(
            player_text,
            (10, 10),
        )

        # Goal X
        goal_x, goal_y, goal_w, goal_h = (
            game.level.goal.bounds()
        )

        goal_text = font.render(
            f"Goal X: {goal_x:.0f}",
            True,
            (0, 100, 0),
        )

        screen.blit(
            goal_text,
            (10, 40),
        )

        # Game state
        state_text = font.render(
            f"State: {game.state.name}",
            True,
            (0, 0, 0),
        )

        screen.blit(
            state_text,
            (10, 70),
        )

        # =========================================================
        # CAMERA POSITION INDICATOR
        # =========================================================

        camera_text = font.render(
            f"Camera X: {camera_x:.0f}",
            True,
            (0, 0, 0),
        )

        screen.blit(
            camera_text,
            (10, 100),
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()