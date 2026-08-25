import pygame

from game.game import Game, GameState


def main():
    pygame.init()

    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("RL Platformer")

    clock = pygame.time.Clock()

    game = Game(seed=42)

    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        action = {
            "left": keys[pygame.K_LEFT] or keys[pygame.K_a],
            "right": keys[pygame.K_RIGHT] or keys[pygame.K_d],
            "jump": keys[pygame.K_SPACE] or keys[pygame.K_UP],
        }

        game.step(action, dt)

        game.render(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()