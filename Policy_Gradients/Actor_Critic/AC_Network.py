import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AC_Network(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=2048, fc2_dims=1536):
        super().__init__()

        self.fc = nn.Sequential(
                            nn.Linear(*input_dims, fc1_dims),
                            nn.ReLU(),
                            nn.Linear(fc1_dims, fc2_dims),
                            nn.ReLU()
                            )
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc(state)
        pi = self.pi(x)
        V = self.v(x)

        return (pi, V)
