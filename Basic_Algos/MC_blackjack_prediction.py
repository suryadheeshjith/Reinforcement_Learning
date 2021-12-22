import gym
import numpy as np


class Agent():
    def __init__(self, gamma=0.99):
        self.V = {}
        self.sum_space = [i for i in range(4,22)]
        self.dealer_show_card_space = [i for i in range(1,11)]
        self.ace_space = [False, True]
        self.action_space = [0,1] # Fold, Hit

        self.state_space = []
        self.returns = {}
        self.state_visited = {}
        self.memory = []
        self.gamma = gamma

        self.init_vals()

    def init_vals(self):
        for i in self.sum_space:
            for j in self.dealer_show_card_space:
                for k in self.ace_space:
                    self.V[(i,j,k)] = 0
                    self.returns[(i,j,k)] = []
                    self.state_visited[(i,j,k)] = 0
                    self.state_space.append((i,j,k))

    def policy(self, state):
        total, _, _ = state
        action = 0 if total >=20 else 1
        return action

    def update_V(self):
        for idx, (state, _) in enumerate(self.memory):
            G = 0
            if self.state_visited[state]==0:
                self.state_visited[state] += 1
                discount = 1
                for t, (_, reward) in enumerate(self.memory[idx:]):
                    G += reward*discount
                    self.returns[state].append(G)
                    discount *= self.gamma

        for state, _ in self.memory:
            self.V[state] = np.mean(self.returns[state])

        for state in self.state_space:
            self.state_visited[state] = 0

        self.memory = []


if __name__ == "__main__":
    num_games = 50000
    env = gym.make('Blackjack-v0')
    agent = Agent()

    for i in range(num_games):
        if (i+1)%500==0:
            print("Episode {0}".format(i+1))

        observation = env.reset()
        done = False

        while not done:
            action = agent.policy(observation)
            observation_, reward, done, info = env.step(action)
            agent.memory.append((observation, reward))
            observation = observation_
        agent.update_V()


    print(agent.V[(21,3,True)])
    print(agent.V[(20,1,True)])
