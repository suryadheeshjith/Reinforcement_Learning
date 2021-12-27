import numpy as np
import torch as T
import torch.nn.functional as F

from utils import OUActionNoise, ReplayBuffer
from network import Critic, Actor

class Agent():
    def __init__(self, alpha, beta, input_dims, tau,
                    action_dims, name, chkptdir, gamma = 0.99, fc1_dims = 400, fc2_dims = 300,
                    max_buffer_size = 1000000, batch_size = 64):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size

        self.memory = ReplayBuffer(max_buffer_size, input_dims, action_dims)
        self.noise = OUActionNoise(mu = np.zeros(action_dims))

        self.actor = Actor(input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims, alpha)
        self.critic = Critic(input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims, beta)

        self.target_actor = Actor(input_dims, action_dims, name+'_target', chkptdir, fc1_dims, fc2_dims, alpha)
        self.target_critic = Critic(input_dims, action_dims, name+'_target', chkptdir, fc1_dims, fc2_dims, beta)

        self.update_network_parameters(tau = 1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_dict = dict(actor_params)
        critic_dict = dict(critic_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in target_actor_dict:
            target_actor_dict[name] = tau*actor_dict[name].clone() + (1-tau)*target_actor_dict[name].clone() # clone used to make a copy

        for name in target_critic_dict:
            target_critic_dict[name] = tau*critic_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()

        self.target_actor.load_state_dict(target_actor_dict)
        self.target_critic.load_state_dict(target_critic_dict)

    def choose_action(self, observation):
        self.actor.eval() # Dropout and BatchNorm (and maybe some custom modules) behave differently during training and evaluation. So need to set here cause no learning done here
        state = T.tensor([observation], dtype = T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype = T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()[0] # mu_prime is a pytorch tensor

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition( state, action, reward, state_, done)


    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        statest = T.tensor(states, dtype = T.float).to(self.actor.device)
        actionst = T.tensor(actions, dtype = T.float).to(self.actor.device)
        rewardst = T.tensor(rewards, dtype = T.float).to(self.actor.device)
        states_t = T.tensor(states_, dtype = T.float).to(self.actor.device)
        donest = T.tensor(dones).to(self.actor.device)

        # Calculation of target Q
        target_actions = self.target_actor.forward(states_t)
        critic_values_ = self.target_critic.forward(states_t,target_actions)
        critic_values_[donest] = 0.0
        critic_values_ = critic_values_.view(-1)
        target = rewardst + self.gamma * critic_values_
        target = target.view(self.batch_size, 1)

        # Calculation of Current Q
        critic_values = self.critic.forward(statest,actionst)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_values)
        critic_loss.backward()
        self.critic.optimizer.step()


        self.actor.optimizer.zero_grad()
        actor_loss = - self.critic.forward(statest, self.actor.forward(statest)) # First, negative due to gradient ascent. Second, be
                                                                                 # careful not to get confused with critic_values. Actions
                                                                                 # must be sampled from actor clearly shown in algorithm
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()

        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()

        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
