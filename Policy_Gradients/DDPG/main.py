import numpy as np
import gym

from utils import plot_curve
from agent import Agent

if __name__ == "__main__":

    np.random.seed(0)
    n_games =1000
    load_chkpt = False
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims = env.observation_space.shape, tau=0.001,
                    action_dims = env.action_space.shape[0], name = 'DDPG', chkptdir= 'models') # lr = 5e-6, input_dims=[8], fc1_dims = 256, fc2_dims = 256

    if load_chkpt:
        agent.load_models()


    scores = []
    best_score = env.reward_range[0]
    n_steps = 0

    for i in range(n_games):
        done=False
        observation = env.reset()
        score =0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            if not load_chkpt:
                agent.learn()

            if load_chkpt or i % 100 == 0:
                env.render()

            score += reward
            observation = observation_
            n_steps+=1


        scores.append(score)
        agent.noise.reset()

        avg_score = np.mean(scores[-100:])
        print('Episode ', i+1, 'score %.2f' % score, 'avg score %.2f' % avg_score)

        
        if i >= 99:
            if avg_score > best_score:
                best_score = avg_score
                if not load_chkpt:
                    agent.save_models()


        if load_chkpt and n_steps>30:
            break

    x = [i+1 for i in range(len(scores))]
    plot_curve(scores, x)
