#!/usr/bin/env python


from snake import SnakeGame
from game_controller import HumanController


if __name__ == '__main__':
    game = SnakeGame()
    controller = HumanController(game)
    game.run()
