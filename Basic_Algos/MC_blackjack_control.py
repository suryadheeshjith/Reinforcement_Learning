import gym
import numpy as np
import random
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, gamma=0.99, epsilon = 0.001):
        self.Q = {}
        self.policy_dict = {}

        self.sum_space = [i for i in range(4,22)]
        self.dealer_show_card_space = [i for i in range(1,11)]
        self.ace_space = [False, True]
        self.action_space = [0,1] # Fold, Hit

        self.action_state_space = []
        self.action_returns = {}
        self.action_state_visited = {}
        self.memory = []

        self.gamma = gamma
        self.eps = epsilon

        self.init_vals()

    def init_vals(self):
        for i in self.sum_space:
            for j in self.dealer_show_card_space:
                for k in self.ace_space:
                    self.policy_dict[(i,j,k)] = []
                    for action in self.action_space:
                        self.Q[((i,j,k),action)] = 0
                        self.action_returns[((i,j,k),action)] = []
                        self.action_state_visited[((i,j,k),action)] = 0
                        self.action_state_space.append(((i,j,k),action))
                        self.policy_dict[(i,j,k)].append(1/len(self.action_space))


    def choose_action(self, state):
        action = np.random.choice(self.action_space, p = self.policy_dict[state])
        return action


    def update_Q(self):
        for idx, (state, action, _) in enumerate(self.memory):
            G = 0
            discount = 1
            if self.action_state_visited[(state,action)]==0:
                self.action_state_visited[(state,action)] += 1
                for t, (_, _, reward) in enumerate(self.memory[idx:]):
                    G += reward*discount
                    self.action_returns[(state,action)].append(G)
                    discount *= self.gamma

        for state, action, _ in self.memory:
            self.Q[(state,action)] = np.mean(self.action_returns[(state,action)])
            self.update_policy(state)

        for (state,action) in self.action_state_visited.keys():
            self.action_state_visited[(state,action)] = 0

        self.memory = []

    def update_policy(self, state):
        actions = [self.Q[(state,a)] for a in self.action_space]
        a_max = np.argmax(actions)
        n_actions = len(self.action_space)
        probs = []
        for action in self.action_space:
            prob = 1 - self.eps + self.eps / n_actions if action == a_max else self.eps/n_actions
            probs.append(prob)
        self.policy_dict[state] = probs


if __name__ == "__main__":
    num_games = 200000
    env = gym.make('Blackjack-v0')
    agent = Agent()
    agent2 = Agent_Phil()
    win_lose_draw = {-1:0, 0:0, 1:0}
    win_rates = []


    for i in range(num_games):
        if i>0 and i%1000==0:
            pct = win_lose_draw[1]/i
            win_rates.append(pct)
        if (i+1) % 50000 ==0:
            rates = win_rates[-1] if win_rates else 0.0
            print("Episode {0} : Win rate {1}".format(i+1, rates))

        observation = env.reset()
        done = False

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.memory.append((observation, action, reward))
            observation = observation_

        agent.update_Q()

        win_lose_draw[reward] += 1

    plt.plot(win_rates)
    plt.show()
