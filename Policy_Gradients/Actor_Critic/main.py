import gym
import matplotlib.pyplot as plt
import numpy as np

from AC_Agent import Agent

def plot_curve(scores, x):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):i+1])
    plt.plot(x, running_avg)
    plt.show()

if __name__ == "__main__":

    n_games =3000
    env = gym.make('LunarLander-v2')
    agent = Agent(lr = 5e-6, input_dims=[8], fc1_dims = 256, fc2_dims = 256)


    scores = []
    for i in range(n_games):
        done=False
        observation = env.reset()
        score =0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            score += reward
            observation = observation_
            if i % 100 == 0:
                env.render()

        scores.append(score)

        if (i+1) % 100 ==0:
            avg_score = np.mean(scores[-100:])
            print('Episode ', i+1, 'score %.2f' % score, 'avg score %.2f' % avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_curve(scores, x)
