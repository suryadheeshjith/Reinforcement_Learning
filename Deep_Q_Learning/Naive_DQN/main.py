import gym
import numpy as np
from utils import plot_learning_curve
from Agent import Agent

if __name__=='__main__':

    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []

    agent = Agent(input_dims = env.observation_space.shape, n_actions = env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, obs_)
            score+=reward
            obs = obs_

        scores.append(score)
        eps_history.append(agent.eps)

        if i%100==0:
            avg_score = np.mean(scores[-100:])
            print('Episode ',i,' score %.1f avg score %.1f epsilon %.2f' % (score, avg_score,agent.eps))


    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
