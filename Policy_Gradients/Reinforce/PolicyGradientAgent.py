import numpy as np
import torch as T
import torch.nn.functional as F
from PolicyGradientNetwork import PolicyGradientNetwork

class PolicyGradientAgent():
    def __init__(self, lr, input_dims, n_actions=4, gamma =0.99):
        self.gamma = gamma
        self.lr = lr

        self.reward_memory = []
        self.action_memory= []

        self.policy = PolicyGradientNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        state = T.Tensor(observation).to(self.policy.device)
        probs = F.softmax(self.policy.forward(state), dim=-1)
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)


    def learn(self):

        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            g=0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                g += discount*self.reward_memory[k]
                discount*= self.gamma
            G[t] = g

        G = T.tensor(G, dtype = T.float).to(self.policy.device)

        loss =0
        for g, logprob in zip(G, self.action_memory):
            loss+= -g*logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
