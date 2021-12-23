import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyGradientNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super().__init__()

        self.fc = nn.Sequential(
                        nn.Linear(*input_dims, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, n_actions)
                        )

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc(state)

        return x
