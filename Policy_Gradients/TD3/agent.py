import numpy as np
import torch as T
import torch.nn.functional as F

from utils import ReplayBuffer
from network import Critic, Actor

class Agent():
    def __init__(self, env, alpha, beta, input_dims,
                    action_dims, name, chkptdir='models', tau = 5e-3,  gamma = 0.99, fc1_dims = 400, fc2_dims = 300,
                    max_buffer_size = 1000000, batch_size = 100, update_actor_interval = 2, warmup = 1000, noise =0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size

        self.input_dims = input_dims
        self.action_dims = action_dims

        self.update_actor_interval = update_actor_interval
        self.warmup = warmup
        self.noise = noise
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.learn_step_cntr = 0
        self.time_step = 0

        self.memory = ReplayBuffer(max_buffer_size, input_dims, action_dims)

        self.actor = Actor(input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims, alpha)
        self.critic1 = Critic(input_dims, action_dims, name+'_1', chkptdir, fc1_dims, fc2_dims, beta)
        self.critic2 = Critic(input_dims, action_dims, name+'_2', chkptdir, fc1_dims, fc2_dims, beta)

        self.target_actor = Actor(input_dims, action_dims, name+'_target', chkptdir, fc1_dims, fc2_dims, alpha)
        self.target_critic1 = Critic(input_dims, action_dims, name+'_target1', chkptdir, fc1_dims, fc2_dims, beta)
        self.target_critic2 = Critic(input_dims, action_dims, name+'_target2', chkptdir, fc1_dims, fc2_dims, beta)

        self.update_network_parameters(tau = 1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params1 = self.critic1.named_parameters()
        critic_params2 = self.critic2.named_parameters()

        target_actor_params = self.target_actor.named_parameters()
        target_critic_params1 = self.target_critic1.named_parameters()
        target_critic_params2 = self.target_critic2.named_parameters()

        actor_dict = dict(actor_params)
        critic_dict1 = dict(critic_params1)
        critic_dict2 = dict(critic_params2)

        target_actor_dict = dict(target_actor_params)
        target_critic_dict1 = dict(target_critic_params1)
        target_critic_dict2 = dict(target_critic_params2)

        for name in target_actor_dict:
            target_actor_dict[name] = tau*actor_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()

        for name in target_critic_dict1:
            target_critic_dict1[name] = tau*critic_dict1[name].clone() + (1-tau)*target_critic_dict1[name].clone()
            target_critic_dict2[name] = tau*critic_dict2[name].clone() + (1-tau)*target_critic_dict2[name].clone()

        self.target_actor.load_state_dict(target_actor_dict)
        self.target_critic1.load_state_dict(target_critic_dict1)
        self.target_critic2.load_state_dict(target_critic_dict2)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            # return T.tensor(np.random.normal(scale=self.noise, size=(self.action_dims,)))
            min, max = self.min_action[0], self.max_action[0]
            mu_prime =  (max-min)*T.rand((self.action_dims,)) + min
        else:
            state = T.tensor(observation, dtype = T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
            mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),dtype=T.float).to(self.actor.device)
            mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])

        self.time_step += 1
        ret =  mu_prime.cpu().detach().numpy()
        return ret


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
        target_actions = T.clamp(target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5), self.min_action[0], self.max_action[0])

        critic_values1_ = self.target_critic1.forward(states_t,target_actions)
        critic_values2_ = self.target_critic2.forward(states_t,target_actions)

        critic_values1_[donest] = 0.0
        critic_values2_[donest] = 0.0

        critic_values1_ = critic_values1_.view(-1)
        critic_values2_ = critic_values2_.view(-1)

        target = rewardst + self.gamma * T.min(critic_values1_,critic_values2_)
        target = target.view(self.batch_size, 1)

        # Calculation of Current Q
        critic_values1 = self.critic1.forward(statest,actionst)
        critic_values2 = self.critic2.forward(statest,actionst)

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()

        critic_loss1 = F.mse_loss(target, critic_values1)
        critic_loss2 = F.mse_loss(target, critic_values2)
        total_critic_loss = critic_loss1 + critic_loss2
        total_critic_loss.backward()

        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.learn_step_cntr += 1


        if self.learn_step_cntr % self.update_actor_interval == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = - self.critic1.forward(statest, self.actor.forward(statest))
            actor_loss = T.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters()


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()

        self.critic1.save_checkpoint()
        self.target_critic1.save_checkpoint()

        self.critic2.save_checkpoint()
        self.target_critic2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()

        self.critic1.load_checkpoint()
        self.target_critic1.load_checkpoint()

        self.critic2.load_checkpoint()
        self.target_critic2.load_checkpoint()
