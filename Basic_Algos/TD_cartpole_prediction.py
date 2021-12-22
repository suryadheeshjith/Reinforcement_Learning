import numpy as np
import gym

def simple_policy(state):
    action = 0 if state < 5 else 1
    return action

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    alpha = 0.1
    gamma = 0.99

    states = np.linspace(-0.2094, 0.2094, 10) # Must be between -12 to 12 degrees ~ 0.2094 rad
    print(states)
    V = {}
    for state in range(len(states)+1): # State corresponds to index of bin. 0 is <states[0], 1 is >states[0] and <states[1], 10 is >states[9]
        V[state] = 0

    for i in range(5000):
        obs = env.reset() # Cart position, Cart velocity, Pole angle, Tip of pole velocity
        done = False
        while not done:
            state = int(np.digitize(obs[2], states))
            action = simple_policy(state)
            obs_, reward, done, info = env.step(action)
            state_ = int(np.digitize(obs_[2], states))
            V[state] = V[state] + alpha*(reward + gamma*V[state_] - V[state])
            obs = obs_

    for state in V:
        print(state, '%.3f' % V[state])
