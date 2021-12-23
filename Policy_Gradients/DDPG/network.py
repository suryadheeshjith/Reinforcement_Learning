import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import get_path


class Critic(nn.Module):
    def __init__(self, input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims, lr=0.001, weight_decay=0.01):
        super().__init__()

        self.input_dims = input_dims
        self.action_dims = action_dims
        self.name = name
        self.chkptdir = chkptdir
        self.chkptfile = get_path(chkptdir, name)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims


        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.action_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights(self):
        val1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        val2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        val3 = 1./np.sqrt(self.action_value.weight.data.size()[0])# Not mentioned in paper but doing it anyway

        self.fc1.weight.data.uniform_(-val1, val1)
        self.fc1.bias.data.uniform_(-val1, val1)

        self.fc2.weight.data.uniform_(-val2, val2)
        self.fc2.bias.data.uniform_(-val2, val2)

        self.q.weight.data.uniform_(-0.003, 0.003)
        self.q.bias.data.uniform_(-0.003, 0.003)

        self.action_value.weight.data.uniform_(-val3, val3)
        self.action_value.bias.data.uniform_(-val3, val3)

    def forward(self, state, action):
        x = self.fc1(state)
        x = F.relu(self.bn1(x))

        x = self.fc2(x)
        x = self.bn2(x)

        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(action_value, x))

        ret = self.q(state_action_value)

        return ret

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chkptfile+'_critic')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chkptfile+'_critic'))



class Actor(nn.Module):
    def __init__(self, input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims, lr=0.001):
        super().__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.name = name
        self.chkptdir = chkptdir
        self.chkptfile = get_path(chkptdir, name)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims


        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)


        self.mu = nn.Linear(self.fc2_dims, action_dims)

        self.init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights(self):
        val1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        val2 = 1./np.sqrt(self.fc2.weight.data.size()[0])

        self.fc1.weight.data.uniform_(-val1, val1)
        self.fc1.bias.data.uniform_(-val1, val1)

        self.fc2.weight.data.uniform_(-val2, val2)
        self.fc2.bias.data.uniform_(-val2, val2)

        self.mu.weight.data.uniform_(-0.003, 0.003)
        self.mu.bias.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(self.bn1(x))

        x = self.fc2(x)
        x = F.relu(self.bn2(x))

        x = T.tanh(self.mu(x)) # -1 to +1 bound actions

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chkptfile+'_actor')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chkptfile+'_actor'))
