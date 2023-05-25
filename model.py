# Import Libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class linearQNet(nn.Module):
    # Initialise Reinforcement Learning Model
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # Linear Unit Function
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    # Save Current State
    def save(self, file_name='new_snake_model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    # Initialise Machine Learning Trainer
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # Trainer Logic
    def train_step(self, state, action, reward, new_state, done):
        # Get (n, x)
        state = torch.tensor(state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Get (1, x)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicted Q with current state
        pred = self.model(state)

        # New Q Prediction
        target = pred.clone()
        for inx in range(len(done)):
            newQ = reward[inx]
            if not done[inx]:
                newQ = reward[inx] + self.gamma * torch.max(self.model(new_state[inx]))
            
            target[inx][torch.argmax(action[inx]).item()] = newQ

        # Mean Squared Error
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()

