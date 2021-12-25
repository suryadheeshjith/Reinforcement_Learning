
import numpy as np
import gym

from utils import plot_curve
from agent import Agent

if __name__ == "__main__":

    np.random.seed(0)
    n_games =1500
    load_chkpt = True
    env = gym.make('BipedalWalker-v3')
    agent = Agent(env, alpha=0.001, beta=0.001, input_dims = env.observation_space.shape,
                    action_dims = env.action_space.shape[0], name = 'TD3')

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

        avg_score = np.mean(scores[-100:])
        print('Episode ', i+1, 'score %.2f' % score, 'avg score %.2f' % avg_score)


        if i>99 and avg_score > best_score:
            best_score = avg_score
            if not load_chkpt:
                agent.save_models()


        if load_chkpt and n_steps>6000:
            break

    x = [i+1 for i in range(len(scores))]
    plot_curve(scores, x)
