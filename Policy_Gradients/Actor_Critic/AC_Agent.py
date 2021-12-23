import numpy as np
import torch.nn.functional as F
import torch as T
from AC_Network import AC_Network


class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions =4 , gamma=0.99):
        self.lr = lr
        self.gamma = gamma

        self.actor_critic = AC_Network(lr, input_dims, n_actions, fc1_dims, fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor(observation, dtype = T.float).to(self.actor_critic.device)
        pi, _ = self.actor_critic.forward(state)
        probs = F.softmax(pi, dim=-1)
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        self.log_prob = action_probs.log_prob(action)
        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()
        statet = T.tensor(state, dtype = T.float).to(self.actor_critic.device)
        rewardt = T.tensor(reward, dtype = T.float).to(self.actor_critic.device)
        state_t = T.tensor(state_, dtype = T.float).to(self.actor_critic.device)

        _, critic_statevalue = self.actor_critic.forward(statet)
        _, critic_state_value = self.actor_critic.forward(state_t)

        delta = reward + self.gamma * critic_state_value * (1-int(done)) - critic_statevalue#1

        actor_loss = -self.log_prob * delta
        critic_loss = delta**2

        (actor_loss+critic_loss).backward()
        self.actor_critic.optimizer.step()

        self.log_prob = None
