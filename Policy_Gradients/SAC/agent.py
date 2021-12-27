import numpy as np
import torch as T
import torch.nn.functional as F

from utils import ReplayBuffer
from network import Critic, Actor, Value

class Agent():
    def __init__(self, env, alpha, beta, input_dims,
                    action_dims, name, chkptdir='models', tau = 5e-3,  gamma = 0.99, fc1_dims = 256, fc2_dims = 256,
                    max_buffer_size = 1000000, batch_size = 256, reward_scale = 2):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.scale = reward_scale

        self.input_dims = input_dims
        self.action_dims = action_dims

        self.memory = ReplayBuffer(max_buffer_size, input_dims, action_dims)

        self.actor = Actor(input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims, env.action_space.high, alpha)
        self.critic1 = Critic(input_dims, action_dims, name+'_1', chkptdir, fc1_dims, fc2_dims, beta)
        self.critic2 = Critic(input_dims, action_dims, name+'_2', chkptdir, fc1_dims, fc2_dims, beta)
        self.value_net = Value(input_dims, action_dims, name, chkptdir, fc1_dims, fc2_dims,beta)

        self.target_value_net = Value(input_dims, action_dims, name+'_targetvalue', chkptdir, fc1_dims, fc2_dims,beta)

        self.update_network_parameters(tau = 1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        value_params = self.value_net.named_parameters()
        target_value_params = self.target_value_net.named_parameters()

        value_dict = dict(value_params)
        target_value_dict = dict(target_value_params)

        for name in target_value_dict:
            target_value_dict[name] = tau*value_dict[name].clone() + (1-tau)*target_value_dict[name].clone()

        self.target_value_net.load_state_dict(target_value_dict)

    def choose_action(self, observation, load_chkpt):

        if load_chkpt:
            state = T.tensor([observation], dtype = T.float).to(self.actor.device)
            mu, _ = self.actor.forward(state)
            ret = mu.cpu().detach().numpy()[0]
        else:
            state = T.tensor([observation], dtype = T.float).to(self.actor.device)
            actions, _ = self.actor.sample_normal(state, reparameterize=False)

            ret =  actions.cpu().detach().numpy()[0]
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


        # Value loss
        value = self.value_net.forward(statest).view(-1)

        actions, log_probs = self.actor.sample_normal(statest, reparameterize = False)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic1.forward(statest, actions)
        q2_new_policy = self.critic2.forward(statest, actions)

        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

        self.value_net.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True) # https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
        self.value_net.optimizer.step()

        # Actor loss
        actions, log_probs = self.actor.sample_normal(statest, reparameterize = True)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic1.forward(statest, actions)
        q2_new_policy = self.critic2.forward(statest, actions)

        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Critic Loss
        value_ = self.target_value_net.forward(states_t).view(-1)
        value_[donest] = 0.0
        q_hat = self.scale * rewardst + self.gamma*value_

        # Action from replay buffer
        q1_old_policy = self.critic1.forward(statest, actionst).view(-1)
        q2_old_policy = self.critic2.forward(statest, actionst).view(-1)
        critic_loss1 = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_loss2 = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss = critic_loss1 + critic_loss2
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_network_parameters()


    def save_models(self):
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.value_net.save_checkpoint()
        self.target_value_net.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.value_net.load_checkpoint()
        self.target_value_net.load_checkpoint()
