from typing import Protocol
from vector import Vector
import pygame


class GameController(Protocol):
    def update(self) -> Vector:
        pass


class HumanController(GameController):
    def __init__(self, game):
        self.game = game
        self.game.controller = self
        pygame.init()
        self.screen = pygame.display.set_mode((game.grid.x * game.scale, game.grid.y * game.scale))
        self.clock = pygame.time.Clock()
        self.color_snake_head = (0, 255, 0)
        self.color_food = (255, 0, 0)

    def __del__(self):
        pygame.quit()

    def update(self) -> Vector:
        next_move = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    next_move = Vector(-1, 0)
                if event.key == pygame.K_RIGHT:
                    next_move = Vector(1, 0)
                if event.key == pygame.K_UP:
                    next_move = Vector(0, -1)
                if event.key == pygame.K_DOWN:
                    next_move = Vector(0, 1)
        self.screen.fill('black')
        for i, p in enumerate(self.game.snake.body):
            pygame.draw.rect(self.screen, (0, max(128, 255 - i * 12), 0), self.block(p))
        pygame.draw.rect(self.screen, self.color_food, self.block(self.game.food.p))
        pygame.display.flip()
        self.clock.tick(10)
        return next_move

    def block(self, obj):
        return (obj.x * self.game.scale,
                obj.y * self.game.scale,
                self.game.scale,
                self.game.scale)
