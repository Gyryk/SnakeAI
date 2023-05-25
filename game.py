# Import Libraries
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

font = pygame.font.Font('./assets/DejaVuSerif.ttf', 32)


# Store Direction as set of preset values
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Constants
point = namedtuple('point', ['x', 'y'])
BLOCK_SIZE = 16
SPEED = 120
# Colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 200)
GREEN = (0, 200, 0)
LIME = (0, 200, 55)
CYAN = (0, 155, 200)
CRIMSON = (155, 0, 50)
DARK_GREEN = (0, 55, 0)
BROWN = (92, 64, 51)


class snakeAI:
    def __init__(self, w=1280, h=960):
        self.w = w
        self.h = h

        # Initialise Display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("AI plays Snake (:")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Draw Horizontal Walls
        for block in range(0, 1280, BLOCK_SIZE):
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(block, 0, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BROWN, pygame.Rect(block + random.randint(0, 4), random.randint(0, 8),
                                                              random.randint(1, 4), random.randint(1, 6)))
            pygame.draw.rect(self.display, BROWN, pygame.Rect(block + random.randint(8, 12), random.randint(0, 8),
                                                              random.randint(1, 4), random.randint(1, 6)))

            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(block, 960-BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BROWN, pygame.Rect(block + random.randint(0, 4), 960-random.randint(6, 14),
                                                              random.randint(1, 4), random.randint(1, 6)))
            pygame.draw.rect(self.display, BROWN, pygame.Rect(block + random.randint(8, 12), 960-random.randint(6, 14),
                                                              random.randint(1, 4), random.randint(1, 6)))

        # Draw Vertical Walls
        for block in range(0, 960, BLOCK_SIZE):
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(0, block, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BROWN, pygame.Rect(random.randint(0, 4), block + random.randint(0, 8),
                                                              random.randint(1, 4), random.randint(1, 6)))
            pygame.draw.rect(self.display, BROWN, pygame.Rect(random.randint(8, 12), block + random.randint(0, 8),
                                                              random.randint(1, 4), random.randint(1, 6)))

            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(1280-BLOCK_SIZE, block, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BROWN, pygame.Rect(1280-random.randint(10, 14), block + random.randint(0, 8),
                                                              random.randint(1, 4), random.randint(1, 6)))
            pygame.draw.rect(self.display, BROWN, pygame.Rect(1280-random.randint(10, 14), block + random.randint(0, 8),
                                                              random.randint(1, 4), random.randint(1, 6)))

        # Initialise Game Variables
        self.direction = Direction.RIGHT
        self.head = point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      point(self.head.x - BLOCK_SIZE, self.head.y),
                      point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    # Place food in valid pixels
    def _place_food(self):
        x = random.randint(BLOCK_SIZE, (self.w-(BLOCK_SIZE*2))//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(BLOCK_SIZE, (self.h-(BLOCK_SIZE*2))//BLOCK_SIZE) * BLOCK_SIZE
        self.food = point(x, y)
        if self.food in self.snake:
            self._place_food()

    # Frame Executes
    def play_step(self, action):
        self.frame_iteration += 1

        # Get User Input
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                quit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

        # Move Snake
        self._move(action)
        self.snake.insert(0, self.head)

        # Game Over Check
        reward = 0

        game_over = False
        if self.is_collision():
            game_over = True
            reward -= 10
            return reward, game_over, self.score
        elif self.frame_iteration > 120*len(self.snake):
            game_over = True
            reward -= 10
            return reward, game_over, self.score

        # Food Logic
        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
        else:
            self.snake.pop()

        # Update UI and Clock
        self._update_ui()
        self.clock.tick(SPEED)

        # Return Game State
        return reward, game_over, self.score

    def is_collision(self, p=None):
        if p is None:
            p = self.head

        # Snake hits horizontal boundary
        if p.x > self.w - (BLOCK_SIZE*2) or p.x < BLOCK_SIZE:
            return True
        # Snake hits vertical boundary
        if p.y > self.h - (BLOCK_SIZE*2) or p.y < BLOCK_SIZE:
            return True
        # Snake hits body
        if p in self.snake[1:]:
            return True

        return False

    # Interface Updates
    def _update_ui(self):
        # self.display.fill(BLACK)

        # Background
        for blockX in range(BLOCK_SIZE, 1280-BLOCK_SIZE, BLOCK_SIZE):
            for blockY in range(BLOCK_SIZE, 960-BLOCK_SIZE, BLOCK_SIZE):
                pygame.draw.rect(self.display, LIME, pygame.Rect(blockX, blockY, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, WHITE, pygame.Rect(blockX + 6, blockY + 6, 4, 4))

        # Snake
        for p in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, CYAN, pygame.Rect(p.x+2, p.y+2, 12, 12))
            pygame.draw.rect(self.display, BLUE, pygame.Rect(p.x+4, p.y+4, 8, 8))

        # Apple
        pygame.draw.rect(self.display, CRIMSON, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x+3, self.food.y+3, 10, 10))
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x+1, self.food.y+1, 4, 4))

        # Send Updates
        text = font.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [32, 32])
        pygame.display.flip()

    # Move the snake around
    def _move(self, action):
        # actions = [straight, right, left]
        x = self.head.x
        y = self.head.y

        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        inx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clockwise[inx]
        elif np.array_equal(action, [0, 1, 0]):
            new_inx = (inx + 1) % 4
            new_direction = clockwise[new_inx]
        elif np.array_equal(action, [0, 0, 1]):
            new_inx = (inx - 1) % 4
            new_direction = clockwise[new_inx]

        self.direction = new_direction

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = point(x, y)


# TODO: Turn boundaries into teleports instead of death barriers, Increase snake speed when size increases
# TODO: Create new sprites for everything

# TODO: Make flappy bird in the same format
