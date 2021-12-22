import numpy as np
import gym
import matplotlib.pyplot as plt


class Agent():
    def __init__(self, n_actions, state_space, gamma=0.99, lr=0.01, eps_start=1.0, eps_end=0.1, eps_dec=0.000004):
        self.n_actions = n_actions
        self.state_space = state_space
        self.gamma = gamma
        self.lr = lr
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec

        self.actions = [i for i in range(n_actions)]
        self.Q = {}

        self.init_Q()

    def init_Q(self):
        for state in self.state_space:
            for action in self.actions:
                self.Q[(state,action)] = 0.0

    def max_action(self, state):
        actions = [self.Q[(state,a)] for a in self.actions]
        return np.argmax(actions)

    def choose_action(self, state):
        r = np.random.random()
        if r < self.eps:
            return np.random.choice(self.actions)

        else:
            return self.max_action(state)

    def update_eps(self):
        self.eps = max(self.eps_end, self.eps - self.eps_dec)

    def learn(self, state, action, reward, state_):
        max_action = self.max_action(state_)
        self.Q[(state,action)] += self.lr * (reward + self.gamma*self.Q[(state_,max_action)] - self.Q[(state,action)])
        self.update_eps()

class StateDigitizer():
    def __init__(self, bounds = (2.4, 4, 0.209, 4), n_bins=10):
        self.position_space = np.linspace(-1*bounds[0], bounds[0], n_bins)
        self.velocity_space = np.linspace(-1*bounds[1], bounds[1], n_bins)
        self.pole_angle_space = np.linspace(-1*bounds[2], bounds[2], n_bins)
        self.pole_velocity_space = np.linspace(-1*bounds[3], bounds[3], n_bins)

        self.states = self.get_state_space()

    def get_state_space(self):
        state_space = []
        for i in range(len(self.position_space)+1):
            for j in range(len(self.velocity_space)+1):
                for k in range(len(self.pole_angle_space)+1):
                    for l in range(len(self.pole_velocity_space)+1):
                        state_space.append((i,j,k,l))
        return state_space

    def digitize(self, observation):
        x, x_dot, theta, theta_dot = observation
        d_x = int(np.digitize(x, self.position_space))
        d_x_dot = int(np.digitize(x_dot, self.velocity_space))
        d_theta = int(np.digitize(theta, self.pole_angle_space))
        d_theta_dot = int(np.digitize(theta_dot, self.pole_velocity_space))

        return (d_x, d_x_dot, d_theta, d_theta_dot)


def plot_learning_curve(scores, x):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of scores')
    plt.show()



if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    n_games = 50000

    digitizer = StateDigitizer()
    agent = Agent(n_actions=2, state_space = digitizer.states)

    scores = []


    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        state = digitizer.digitize(obs)


        while not done:
            action = agent.choose_action(state)
            obs_, reward, done, info = env.step(action)
            state_ = digitizer.digitize(obs_)
            agent.learn(state, action, reward, state_)
            state = state_
            score+= reward

        if i%5000 == 0:
            print('episode ', i, 'score %.1f' % score, 'epsilon %.2f' % agent.eps)

        scores.append(score)


    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(scores, x)
