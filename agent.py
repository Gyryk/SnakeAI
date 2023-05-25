# Import Libraries
import torch
import random
import numpy as np
from collections import deque
from game import point, snakeAI, Direction
from model import linearQNet, QTrainer
from graphing import plot

# Constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        # Exploration vs Exploitation Variable, in model evaluation mode it needs to have maximum accuracy
        self.epsilon = 0
        # Discount Rate
        self.gamma = 0.995
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = linearQNet(11, 512, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # Store all relevant game points and directions
        head = game.snake[0]
        point_r = point(head.x + 16, head.y)
        point_l = point(head.x-16, head.y)
        point_u = point(head.x, head.y-16)
        point_d = point(head.x, head.y+16)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Ahead
            (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_r and game.is_collision(point_d)) or (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)),

            # Danger Left
            (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or (dir_d and game.is_collision(point_r)),

            # Moving Direction
            dir_r, dir_l, dir_u, dir_d,

            # Food Right
            game.food.x > game.head.x,
            # Food Left
            game.food.x < game.head.x,
            # Food Up
            game.food.y < game.head.y,
            # Food Down
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # popleft() if runs out of memory ^

    # Training from past games
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        # Group similar data-types and train
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    # Training for current game
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Clamp Epsilon between 16 and 256 to maintain a small amount of randomness and keep training
        # self.epsilon = max(16, 256 - self.n_games)
        final_move = [0, 0, 0]

        if random.randint(0, 400) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


# Load AI Model
def loadModel(agentObj):
    agentObj.model.load_state_dict(torch.load("./models/300games_model_new.pth"))
    agentObj.model.eval()

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_current_mean_scores = []
    total_score = 0
    high_score = 0
    agent = Agent()
    game = snakeAI()

    while True:
        loadModel(agent)

        # Get current state
        old_state = agent.get_state(game)

        # Calculate optimal action
        final_action = agent.get_action(old_state)

        # Get updated values
        reward, done, score = game.play_step(final_action)
        new_state = agent.get_state(game)

        # Train Short Memory
        agent.train_short_memory(old_state, final_action, reward, new_state, done)

        # Store Memory
        agent.store_memory(old_state, final_action, reward, new_state, done)

        if done:
            # Train Replay Memory for Experience Replay
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > high_score:
                high_score = score
                agent.model.save()

            print('Game',  agent.n_games, '| Score:', score, '| High Score:', high_score)

            # Call plotting script
            plot_scores.append(score)

            total_score += score
            mean_score = total_score/agent.n_games
            mean_score = int(mean_score*1000)/1000
            plot_mean_scores.append(mean_score)

            current_mean_score = getAverage(plot_scores)
            plot_current_mean_scores.append(current_mean_score)

            minY = np.min(plot_scores)
            maxY = np.max(plot_scores)

            plot(plot_scores, plot_mean_scores, plot_current_mean_scores, minY, maxY)

def getAverage(array):
    total_score = 0
    if len(array) >= 16:
        for i in range(1, 16):
            total_score += array[-i]
        mean = total_score/16
    else:
        for i in range(len(array)):
            total_score += array[i]

        mean = total_score/len(array)
    mean = int(mean*1000)/1000

    return mean


# Start Executes
if __name__ == '__main__':
    train()

# TODO: fix the running into itself and looping
