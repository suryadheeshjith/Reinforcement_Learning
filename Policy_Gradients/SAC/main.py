import pybullet_envs
import numpy as np
import gym

from utils import plot_curve
from agent import Agent

if __name__ == "__main__":

    np.random.seed(0)

    #env_id = 'LunarLanderContinuous-v2'
    #env_id = 'BipedalWalker-v2'
    #env_id = 'AntBulletEnv-v0'
    env_id = 'InvertedPendulumBulletEnv-v0'
    #env_id = 'CartPoleContinuousBulletEnv-v0'

    n_games =250
    load_chkpt = True
    env = gym.make(env_id)
    agent = Agent(env, alpha=3e-4, beta=3e-4, input_dims = env.observation_space.shape,
                    action_dims = env.action_space.shape[0], name = 'SAC_'+env_id)

    if load_chkpt:
        agent.load_models()
        env.render(mode='human')


    scores = []
    best_score = env.reward_range[0]
    n_steps = 0

    for i in range(n_games):
        done=False
        observation = env.reset()
        score =0
        while not done:
            action = agent.choose_action(observation, load_chkpt)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            if not load_chkpt:
                agent.learn()

            score += reward
            observation = observation_
            n_steps+=1


        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('Episode ', i+1, 'score %.2f' % score, 'avg score %.2f' % avg_score)


        if avg_score > best_score:
            best_score = avg_score
            if not load_chkpt:
                agent.save_models()


        if load_chkpt and n_steps>60000:
            break

    x = [i+1 for i in range(len(scores))]
    plot_curve(scores, x)
