import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from utils import get_path

class Value(nn.Module):
    def __init__(self, input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims, lr=3e-4):
        super().__init__()

        self.input_dims = input_dims
        self.action_dims = action_dims
        self.name = name
        self.chkptdir = chkptdir
        self.chkptfile = get_path(chkptdir, name)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.v = nn.Linear(self.fc2_dims, 1)


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        ret = self.v(x)

        return ret

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chkptfile+'_value.zip')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chkptfile+'_value.zip', map_location={'cuda:0': 'cpu'}))


class Critic(nn.Module):
    def __init__(self, input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims, lr=3e-4):
        super().__init__()

        self.input_dims = input_dims
        self.action_dims = action_dims
        self.name = name
        self.chkptdir = chkptdir
        self.chkptfile = get_path(chkptdir, name)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(input_dims[0]+action_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state,action],dim=1)))
        x = F.relu(self.fc2(x))
        ret = self.q(x)

        return ret

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chkptfile+'_critic.zip')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chkptfile+'_critic.zip', map_location={'cuda:0': 'cpu'}))



class Actor(nn.Module):
    def __init__(self, input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims, max_action, lr=0.001):
        super().__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.name = name
        self.chkptdir = chkptdir
        self.chkptfile = get_path(chkptdir, name)
        self.reparam_noise = 1e-6
        self.max_action = max_action

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims


        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, action_dims)
        self.sigma = nn.Linear(self.fc2_dims, action_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mu(x)
        stddev = T.clamp(self.sigma(x), self.reparam_noise, 1)

        return mean, stddev

    def sample_normal(self, state, reparameterize = True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        actions = None
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1- action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chkptfile+'_actor.zip')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chkptfile+'_actor.zip', map_location={'cuda:0': 'cpu'}))
