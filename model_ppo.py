from torch import nn
from torch.nn import functional as F
import numpy as np

from global_variables import MIN_STD


class ActorPPO(nn.Module):
    def __init__(self, state_size, action_size, fc1=64, fc2=64, fc3=64) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(self.state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        # self.fc3 = nn.Linear(fc2, fc3)

        self.mu_output = nn.Linear(fc2, self.action_size)
        # self.sigma_output = nn.Linear(fc3, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        mu = self.mu_output(x)
        # sigma = F.relu(self.sigma_output(x)) + MIN_STD
        return mu#, sigma


class CriticPPO(nn.Module):
    def __init__(self, state_size, fc1=64, fc2=64, fc3=64) -> None:
        super().__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(self.state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        # self.fc3 = nn.Linear(fc2, fc3)
        self.value_output = nn.Linear(fc2, 1)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        value = self.value_output(x)
        return value
