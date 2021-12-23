import gym
import numpy as np
import matplotlib.pyplot as plt

class Agent():

    def __init__(self,lr,gamma,eps_max,eps_min,eps_dec,n_actions, n_states):
        self.Q = {}
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_max
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.n_actions = n_actions
        self.n_states = n_states
        self.init_Q_table()


    def init_Q_table(self):
        for i in range(self.n_states):
            for j in range(self.n_actions):
                self.Q[(i,j)] = 0.0


    def choose_action(self,state):

        if(np.random.random()<self.eps):
            return np.random.choice([i for i in range(self.n_actions)])

        else:
            actions =  np.array([self.Q[(state, j)] for j in range(self.n_actions)])
            return np.argmax(actions)



    def learn(self, state, action, reward, state_ ):

        actions = np.array([self.Q[(state_, j)] for j in range(self.n_actions)])
        a_max = np.argmax(actions)
        self.Q[(state,action)] += self.lr*((reward+self.gamma*self.Q[(state_, a_max)])-self.Q[(state,action)])

        self.dec_eps()


    def dec_eps(self):
        self.eps = max(self.eps_min, self.eps*self.eps_dec)



if __name__=='__main__':
    env = gym.make('FrozenLake-v0')
    lr = 0.001
    gamma = 0.9
    eps_max = 1.0
    eps_min = 0.01
    eps_dec = 0.9999995
    n_actions= 4
    n_states=16
    agent = Agent(lr,gamma,eps_max,eps_min,eps_dec,n_actions, n_states)

    scores = []
    win_pct_list = []
    n_games = 500000

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_)
            score+=reward
            observation = observation_

        scores.append(score)

        if i%100==0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if i%1000==0:
                print('Episode : ',i,' win_pct %.2f'%win_pct,' epsilon %.2f'%agent.eps)

    plt.plot(win_pct_list)
    plt.show()
