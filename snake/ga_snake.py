#!/usr/bin/env python


from snake import SnakeGame
from ga_controller import GAController


if __name__ == '__main__':
    game = SnakeGame()
    controller = GAController(game)
    game.run()
