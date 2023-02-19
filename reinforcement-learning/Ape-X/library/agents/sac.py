"""
Implement soft actor critic agent here
"""

import copy

import torch
from torch import nn

import library.pytorch_utils as ptu
from library.gym_utils import verify_continuous_action_space
from library.nn.functional import make_module_trainable
from library.nn.layers import LagrangeLayer
from library.nn.model import SquashedGaussianMLPActor, EnsembleMinQNet
from library.rl_functional import soft_update


class SACAgent(nn.Module):
    def __init__(self,
                 env,
                 policy_lr=3e-4,
                 num_q_ensembles=2,
                 mlp_hidden=256,
                 q_lr=3e-4,
                 alpha=1.0,
                 alpha_lr=1e-3,
                 tau=5e-3,
                 target_entropy_per_dim=None,
                 reward_scale=1.0,
                 device=ptu.get_accelerator()
                 ):
        super().__init__()
        self.obs_spec = env.observation_space
        self.act_spec = env.action_space
        verify_continuous_action_space(self.act_spec)
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.alpha_lr = alpha_lr
        self.act_dim = self.act_spec.shape[0]
        self.policy_net = SquashedGaussianMLPActor(env.observation_space.shape[0],
                                                   env.action_space.shape[0],
                                                   mlp_hidden=mlp_hidden)
        self.num_q_ensembles = num_q_ensembles
        self.reward_scale = reward_scale
        self.q_network = EnsembleMinQNet(env.observation_space.shape[0],
                                         env.action_space.shape[0],
                                         mlp_hidden=mlp_hidden,
                                         num_ensembles=num_q_ensembles)
        self.target_q_network = copy.deepcopy(self.q_network)
        make_module_trainable(self.target_q_network, trainable=False)

        self.alpha_net = LagrangeLayer(initial_value=alpha)

        if target_entropy_per_dim is None:
            target_entropy_per_dim = -1

        self.target_entropy = self.act_dim * target_entropy_per_dim

        self.tau = tau

        self.reset_optimizer()

        self.device = device
        self.to(device)

        self.logger = None

    def reset_optimizer(self):
        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=self.policy_lr)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=self.q_lr)
        self.alpha_optimizer = torch.optim.Adam(params=self.alpha_net.parameters(), lr=self.alpha_lr)

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        for i in range(self.num_q_ensembles):
            self.logger.log_tabular(f'Q{i + 1}Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)
        self.logger.log_tabular('TDError', average_only=True)

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)

    def compute_priority_torch(self, obs, act, rew, next_obs, done, gamma):
        with torch.no_grad():
            alpha = self.alpha_net()
            next_action, next_action_log_prob, _, _ = self.policy_net(next_obs, False)
            target_q_values = self.target_q_network(next_obs, next_action,
                                                    take_min=True) - alpha * next_action_log_prob
            q_target = rew * self.reward_scale + gamma * (1.0 - done) * target_q_values

            # q loss
            q_values = self.q_network(obs, act, take_min=False)  # (num_ensembles, None)
            q_values_loss = torch.abs(torch.unsqueeze(q_target, dim=0) - q_values)
            # (num_ensembles, None)
            q_values_loss = torch.mean(q_values_loss, dim=0)  # (None,)
        return q_values_loss

    def compute_priority(self, data):
        data = ptu.convert_dict_to_tensor(data, device=self.device)
        return ptu.to_numpy(self.compute_priority_torch(**data))

    def train_on_batch_torch(self, obs, act, next_obs, done, rew, gamma):
        """ Sample a mini-batch from replay buffer and update the network
        Args:
            obs: (batch_size, ob_dim)
            actions: (batch_size, action_dim)
            next_obs: (batch_size, ob_dim)
            done: (batch_size,)
            reward: (batch_size,)
        Returns: None
        """
        with torch.no_grad():
            alpha = self.alpha_net()
            next_action, next_action_log_prob, _, _ = self.policy_net(next_obs, False)
            target_q_values = self.target_q_network(next_obs, next_action,
                                                    take_min=True) - alpha * next_action_log_prob
            q_target = rew * self.reward_scale + gamma * (1.0 - done) * target_q_values

        # q loss
        q_values = self.q_network(obs, act, take_min=False)  # (num_ensembles, None)
        delta_q = torch.unsqueeze(q_target, dim=0) - q_values
        q_values_loss = 0.5 * torch.square(delta_q)
        # (num_ensembles, None)
        q_values_loss = torch.sum(q_values_loss, dim=0)  # (None,)
        # apply importance weights
        q_values_loss = torch.mean(q_values_loss)
        self.q_optimizer.zero_grad()
        q_values_loss.backward()
        self.q_optimizer.step()

        # policy loss
        action, log_prob, _, _ = self.policy_net(obs, False)
        q_values_pi_min = self.q_network(obs, action, take_min=True)
        policy_loss = torch.mean(log_prob * alpha - q_values_pi_min)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha = self.alpha_net()
        alpha_loss = -torch.mean(alpha * (log_prob.detach() + self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            td_error = torch.abs(delta_q)  # (num_ensembles, None)
            td_error = torch.mean(td_error)  # (None,)

        info = dict(
            LogPi=log_prob,
            Alpha=alpha,
            LossQ=q_values_loss,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
            TDError=td_error
        )

        for i in range(self.num_q_ensembles):
            info[f'Q{i + 1}Vals'] = q_values[i]

        return info

    def train_on_batch(self, data):
        data = ptu.convert_dict_to_tensor(data, device=self.device)
        info = self.train_on_batch_torch(**data)
        self.update_target()
        self.logger.store(**info)

    def act_batch_torch(self, obs, deterministic):
        with torch.no_grad():
            pi_final = self.policy_net.select_action(obs, deterministic)
            return pi_final

    def act_batch_explore(self, obs, global_steps):
        obs = torch.as_tensor(obs, device=self.device)
        return self.act_batch_torch(obs, deterministic=False).cpu().numpy()

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        return self.act_batch_torch(obs, deterministic=True).cpu().numpy()
