from torch import nn
from torch.nn import functional as F


class ActorPPO(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=1024, fc2_size=1024, fc3_size=512) -> None:
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(self.state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.mu_output = nn.Linear(fc3_size, self.action_size)
        self.logstd = nn.Linear(fc3_size, self.action_size)
        self.reset_paramethers()


    def reset_paramethers(self):
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.orthogonal_(self.fc1.weight, gain=.2)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.orthogonal_(self.fc2.weight, gain=.2)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.orthogonal_(self.fc3.weight, gain=.2)
        nn.init.constant_(self.mu_output.bias, 0)
        nn.init.orthogonal_(self.mu_output.weight, gain=.2)
        nn.init.orthogonal_(self.logstd.weight, gain=.2)
        nn.init.constant_(self.logstd.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu_output(x)
        logstd = self.logstd(x)
        return mu, logstd


class CriticPPO(nn.Module):
    def __init__(self, state_size, fc1_size=1024, fc2_size=1024, fc3_size=512) -> None:
        super().__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(self.state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.value_output = nn.Linear(fc3_size, 1)
        self.reset_paramethers()


    def reset_paramethers(self):
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.orthogonal_(self.fc1.weight, gain=.2)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.orthogonal_(self.fc2.weight, gain=.2)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.orthogonal_(self.fc3.weight, gain=.2)
        nn.init.constant_(self.value_output.bias, 0)
        nn.init.orthogonal_(self.value_output.weight, gain=.2)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_output(x)
        return value
