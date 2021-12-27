import numpy as np
import os
import matplotlib.pyplot as plt

# Replay Buffer
class ReplayBuffer():
    def __init__(self, mem_size, input_dims, actions_dims):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((mem_size,*input_dims))
        self.new_state_memory = np.zeros((mem_size,*input_dims))
        self.reward_memory = np.zeros(mem_size)
        self.action_memory = np.zeros((mem_size,actions_dims))
        self.terminal_memory = np.zeros(mem_size, dtype=np.bool)


    def store_transition(self, state, action, reward, state_, done):
        indx = self.mem_cntr % self.mem_size
        self.state_memory[indx] = state
        self.new_state_memory[indx] = state_
        self.reward_memory[indx] = reward
        self.action_memory[indx] = action
        self.terminal_memory[indx] = done

        self.mem_cntr +=1

    def sample_buffer(self, batch_size):
        max_indx = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_indx, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

# Path for checkpoints
def get_path(chkpt_dir, name):
    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)
    checkpoint_file = os.path.join(chkpt_dir, name)
    return checkpoint_file


def plot_curve(scores, x):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):i+1])
    plt.plot(x, running_avg)
    plt.show()
