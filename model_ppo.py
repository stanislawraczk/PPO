import torch
from torch import nn
from torch.nn import functional as F


class ActorPPO(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=256, fc2_size=256, fc3_size=256, initial_logstd_scaling = 0) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(self.state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.mu_output = nn.Linear(fc3_size, self.action_size)
        self.logstd = nn.Parameter(torch.zeros(self.action_size) - initial_logstd_scaling)
        self.reset_paramethers()


    def reset_paramethers(self):
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.orthogonal_(self.fc1.weight, gain=.5)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.orthogonal_(self.fc2.weight, gain=.5)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.orthogonal_(self.fc3.weight, gain=.5)
        nn.init.constant_(self.mu_output.bias, 0)
        nn.init.orthogonal_(self.mu_output.weight, gain=.5)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu_output(x)
        return mu


class CriticPPO(nn.Module):
    def __init__(self, state_size, fc1_size=256, fc2_size=256, fc3_size=256) -> None:
        super().__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(self.state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.value_output = nn.Linear(fc3_size, 1)
        self.reset_paramethers()


    def reset_paramethers(self):
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.orthogonal_(self.fc1.weight, gain=.5)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.orthogonal_(self.fc2.weight, gain=.5)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.orthogonal_(self.fc3.weight, gain=.5)
        nn.init.constant_(self.value_output.bias, 0)
        nn.init.orthogonal_(self.value_output.weight, gain=.5)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_output(x)
        return value
