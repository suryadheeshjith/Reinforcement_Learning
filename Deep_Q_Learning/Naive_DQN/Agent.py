import gym
import numpy as np
from Network import DQN
import torch as T

class Agent():

    def __init__(self, input_dims, n_actions, lr=0.0001, gamma=0.99, eps_max=1.0, eps_min=0.01, eps_dec= 1e-5):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr=lr
        self.gamma=gamma
        self.eps=eps_max
        self.eps_min=eps_min
        self.eps_dec=eps_dec

        self.action_space = [i for i in range(self.n_actions)]
        self.Q = DQN(self.lr, self.n_actions, self.input_dims)

    def choose_action(self,state):

        if(np.random.random()<=self.eps):
            return np.random.choice(self.action_space)

        else:
            state = T.tensor(state, dtype=T.float).to(self.Q.device)
            actions =  self.Q.forward(state)
            return T.argmax(actions).item()

    def dec_eps(self):
        self.eps = max(self.eps_min, self.eps-self.eps_dec)

    def learn(self, state, action, reward, state_ ):

        self.Q.optimizer.zero_grad()
        state_t = T.tensor(state, dtype=T.float).to(self.Q.device)
        action_t = T.tensor(action).to(self.Q.device)
        reward_t = T.tensor(reward).to(self.Q.device)
        state_t_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred =  self.Q.forward(state_t)[action_t] # Get the predictions for action value at a state for only the action that we took.
        q_next = self.Q.forward(state_t_).max() #self.Q.forward(states_).max().detach()

        q_target = reward + self.gamma*q_next

        loss = self.Q.loss(q_target,q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()

        self.dec_eps()
