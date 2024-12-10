import numpy as np
from scipy.special import softmax
import random
from typing import Tuple, Sequence
import pygame
import math
from snake import Vector, Snake, Food, SnakeGame
from collections import deque
from math import tanh



# --------------------
# Observationer og fitness funktion
# --------------------

def update_input_tables(snake, food, grid):
    head = snake.p
    food_pos = food.p

    # Direction tabeller
    FoodTable = [0, 0, 0, 0]  # N, S, E, W
    ObstacleTable = [0, 0, 0, 0]  # N, S, E, W

    
    if food_pos.y < head.y:
        FoodTable[0] = (head.y - food_pos.y)  # North
    elif food_pos.y > head.y:
        FoodTable[1] = (food_pos.y - head.y)  # South
    if food_pos.x > head.x:
        FoodTable[2] = (food_pos.x - head.x)  # East
    elif food_pos.x < head.x:
        FoodTable[3] = (head.x - food_pos.x)  # West

    # Obstacles (walls and body)
    directions = {
        0: Vector(head.x, head.y - 1),  # North
        1: Vector(head.x, head.y + 1),  # South
        2: Vector(head.x + 1, head.y),  # East
        3: Vector(head.x - 1, head.y),  # West
    }
    for i, direction in directions.items():
        if not direction.within(grid) or direction in snake.body:
            ObstacleTable[i] = 1

    return FoodTable + ObstacleTable

def fitness_function(agent, steps, food_count, last_food_step, snake, game):
    """
    Fitnessfunktion, der opdaterer agentens belønning og straf baseret på dens adfærd.
    """
    reward = 0

    # 1. Belønning for at spise mad
    food_reward = food_count * 100  # +100 for hver mad der er spist
    reward += food_reward

    # 2. Straff for at ramme væggen eller sig selv
    if not snake.p.within(game.grid) or snake.cross_own_tail:
        reward -= 1000  # Store straf hvis agenten rammer væggen eller sig selv

    # 3. Belønning for at tage skridt (fremad)
    # Straf for skridt uden at spise mad, men giv en lille belønning for hver skridt taget
    step_penalty = -1 * steps  # -1 point per skridt, uanset hvad
    reward += step_penalty

    # 4. Hvis agenten tager mere end 200 skridt uden at spise ny mad (straff)
    if steps - last_food_step > 200:  # Hvis det er mere end 200 skridt siden sidste mad
        reward -= 50  # -50 ekstra straf for at tage for mange skridt uden mad

    # 5. Belønning for at undgå væggen
    # Hvis agenten undgår væggen (ved at ikke være ude af grænserne), giv en lille belønning
    if snake.p.within(game.grid) and not snake.cross_own_tail:
        reward += 1  # +1 for at undgå væggen og selv overleve

    # 6. Straf for at dø uden at have spist mad
    death_penalty = -100 if food_count == 0 and steps < 400 else 0  # Hvis agenten dør uden at have spist mad
    reward += death_penalty

    return reward







# --------------------
# Neural net og agent-logik
# --------------------

class SimpleModel:
    def __init__(self, *, dims: Tuple[int, ...]):
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self.DNA = []
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                self.DNA.append(np.random.rand(dim, dims[i + 1]))  # Initialisering af vægte
    '''
    def update(self, obs: Sequence) -> np.ndarray:
        x = np.array(obs, dtype=np.float32)  # Input observationer
        for i, layer in enumerate(self.DNA):
            x = x @ layer  # Matrixmultiplikation med vægtmatricerne
            if i != len(self.DNA) - 1:  # Anvend aktiveringsfunktion på skjulte lag
                x = np.tanh(x)  # Du kan skifte dette til ReLU eller en anden funktion
        return softmax(x)  # Brug softmax på output for at få sandsynligheder
    '''

    def update(self, obs: Sequence) -> Tuple[int, ...]:
        x = obs
        for i, layer in enumerate(self.DNA):
            if not i == 0:
                x = np.tanh(x)
            x = x @ layer
        return softmax(x)
    

    def action(self, obs: Sequence):
        return self.update(obs).argmax()  # Vælg den handling, der giver den højeste sandsynlighed


    def mutate(self, mutation_rate) -> bool:
        """Mutation funktion der tilfældigt ændrer værdien af en vægt i DNA"""
        mutation_occurred = False
        if random.random() < mutation_rate:
            random_layer = random.randint(0, len(self.DNA) - 1)
            row = random.randint(0, self.DNA[random_layer].shape[0] - 1)
            col = random.randint(0, self.DNA[random_layer].shape[1] - 1)
            self.DNA[random_layer][row][col] = random.uniform(-1, 1)  # Random mutation
            mutation_occurred = True  # Mutation skete
        return mutation_occurred


    def __add__(self, other):
        """Crossover operation der blander DNA fra to agenter"""
        baby_DNA = []
        for mom, dad in zip(self.DNA, other.DNA):
            if random.random() > 0.5:
                baby_DNA.append(mom)  # Vælg tilfældigt morens eller farens lag
            else:
                baby_DNA.append(dad)
        baby = type(self)(dims=self.dims)  # Opretter et nyt barn
        baby.DNA = baby_DNA  # Tildeler blandet DNA
        return baby




# --------------------
# Simulering og træning
# --------------------

def simulate_game(agent, game, render=False):
    snake = Snake(game=game)
    food = Food(game=game)
    steps = 0
    food_count = 0
    last_food_step = 0  # Denne variabel bruges til at holde styr på, hvornår mad blev spist
    last_food_distance = ((food.p.x - snake.p.x)**2 + (food.p.y - snake.p.y)**2)**0.5

    while steps < 500:
        obs = update_input_tables(snake, food, game.grid)
        action = agent.action(obs)
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]
        snake.move()

        if not snake.p.within(game.grid) or snake.cross_own_tail:
            break

        current_food_distance = ((food.p.x - snake.p.x)**2 + (food.p.y - snake.p.y)**2)**0.5

        # Hvis agenten spiser mad
        if snake.p == food.p:
            snake.add_score()
            food = Food(game=game)
            food_count += 1  # Øg madspisningen
            last_food_step = steps  # Opdater hvornår mad sidst blev spist
            last_food_distance = ((food.p.x - snake.p.x)**2 + (food.p.y - snake.p.y)**2)**0.5

        # Render spillet (hvis render=True)
        if render:
            game.screen.fill((0, 0, 0))
            pygame.draw.rect(game.screen, game.color_food, game.block(food.p))
            for segment in snake.body:
                pygame.draw.rect(game.screen, game.color_snake_head, game.block(segment))
            pygame.display.update()
            game.clock.tick(10)

        steps += 1

    return steps, food_count, last_food_step, snake, game  # Returnér de nødvendige data







def train_agents(agents, generations, mutation_rate):
    try:
        pygame.init()
        for generation in range(generations):
            fitness_scores = []
            max_food = 0
            avg_food = 0
            max_fitness = -float("inf")
            avg_fitness = 0

            best_agent = None
            best_food_count = 0  # Variabel til at holde styr på agenten med flest food

            for agent in agents:
                game = SnakeGame()
                steps, food_count, last_food_step, snake, game = simulate_game(agent, game, render=False)  # Simuler spillet uden rendering
                fitness = fitness_function(agent, steps, food_count, last_food_step, snake, game)  # Beregn fitness
                fitness_scores.append(fitness)

                # Spor agenten med flest food
                if food_count > best_food_count:
                    best_food_count = food_count
                    best_agent = agent

                # Spor agenten med den højeste fitness
                if fitness > max_fitness:
                    max_fitness = fitness
                    best_agent = agent

                max_food = max(max_food, food_count)
                avg_food += food_count
                max_fitness = max(max_fitness, fitness)
                avg_fitness += fitness

            avg_food /= len(agents)
            avg_fitness /= len(agents)

            # Print information about training progress
            print(f"----------------------------------Generation {generation + 1}:------------------------------------------")
            print(f"Max Food: {max_food}, Avg Food: {avg_food}")
            print(f"Max Fitness: {max_fitness}, Avg Fitness: {avg_fitness}")
            
            # Sort agents by fitness
            sorted_agents = [agent for _, agent in sorted(zip(fitness_scores, agents), 
                                                          key=lambda pair: pair[0], reverse=True)]
            top_agents = sorted_agents[:len(sorted_agents) // 2]

            # Information about breeding and mutation
            num_breeded = 0  # Number of bred agents
            num_mutated = 0  # Number of mutated agents

            # Create new agents via crossover and mutation
            new_agents = []

            # 50% from crossover of the top agents
            while len(new_agents) < len(agents) // 2:
                parent1, parent2 = random.sample(top_agents, 2)
                child = parent1 + parent2  # Crossover
                num_breeded += 1  # Count the breeding
                mutation_occurred = child.mutate(mutation_rate)  # Check if mutation occurred

                if mutation_occurred:
                    num_mutated += 1  # Count the mutation

                new_agents.append(child)

            # 50% new agents with random initialization
            while len(new_agents) < len(agents):
                new_agent = SimpleModel(dims=(8, 12, 4))  # Randomly initialize a new agent
                new_agents.append(new_agent)

            agents = new_agents

            # Print breeding and mutation statistics for this generation
            print(f"Generation {generation + 1} - Breed stats: {num_breeded} agents bred, {num_mutated} agents mutated")
            

    finally:
        pygame.quit()



if __name__ == "__main__":
    population_size = 100
    generations = 75
    mutation_rate = 0.1
    agents = [SimpleModel(dims=(8, 12, 4)) for _ in range(population_size)]
    train_agents(agents, generations, mutation_rate)