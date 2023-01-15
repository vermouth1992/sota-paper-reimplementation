import copy

import gym
import numpy as np
import torch.nn as nn
import torch.optim
from tqdm.auto import trange

import d4rl
from library import pytorch_utils as ptu
from library.gym_utils import verify_continuous_action_space, TransformObservationDtype
from library.infrastructure.logger import EpochLogger, setup_logger_kwargs
from library.infrastructure.seeder import Seeder
from library.infrastructure.tester import D4RLTester
from library.infrastructure.timer import StopWatch
from library.nn.layers import LagrangeLayer
from library.nn.model import SquashedGaussianMLPActor, EnsembleMinQNet
from library.replay_buffer import UniformReplayBuffer
from library.rl_functional import soft_update, hard_update

LOG_2 = 0.6931471805599453


class CQLContinuousAgent(nn.Module):
    def __init__(self,
                 env,
                 policy_mlp_hidden=128,
                 policy_lr=3e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 alpha=0.2,
                 alpha_lr=5e-3,
                 alpha_cql=0.2,
                 alpha_cql_lr=5e-3,
                 tau=5e-3,
                 gamma=0.99,
                 num_samples=10,
                 # cql related
                 cql_threshold=-1.,
                 temperature=1.,
                 min_q_weight=1.,
                 target_entropy=None,
                 device=None
                 ):
        super().__init__()
        self.obs_spec = env.observation_space
        self.act_spec = env.action_space
        self.num_samples = num_samples
        self.act_dim = self.act_spec.shape[0]
        self.num_ensembles = 2
        if len(self.obs_spec.shape) == 1:  # 1D observation
            self.obs_dim = self.obs_spec.shape[0]
            self.policy_net = SquashedGaussianMLPActor(self.obs_dim, self.act_dim, policy_mlp_hidden)
            self.q_network = EnsembleMinQNet(self.obs_dim, self.act_dim, q_mlp_hidden, num_ensembles=self.num_ensembles)
            self.target_q_network = copy.deepcopy(self.q_network)
        else:
            raise NotImplementedError
        hard_update(self.target_q_network, self.q_network)

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer = torch.optim.Adam(params=self.q_network.parameters(), lr=q_lr)

        self.log_alpha = LagrangeLayer(initial_value=alpha)
        self.log_cql = LagrangeLayer(initial_value=alpha_cql)
        self.alpha_optimizer = torch.optim.Adam(params=self.log_alpha.parameters(), lr=alpha_lr)
        self.cql_alpha_optimizer = torch.optim.Adam(params=self.log_cql.parameters(), lr=alpha_cql_lr)

        self.target_entropy = -self.act_dim if target_entropy is None else target_entropy
        self.cql_threshold = cql_threshold
        self.temperature = temperature
        self.min_q_weight = min_q_weight

        self.tau = tau
        self.gamma = gamma

        self.max_backup = True

        self.device = device
        self.to(self.device)

        self.logger = None

    def log_tabular(self):
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('LossAlpha', average_only=True)
        self.logger.log_tabular('AlphaCQL', average_only=True)
        self.logger.log_tabular('AlphaCQLLoss', average_only=True)
        self.logger.log_tabular('DeltaCQL', with_min_and_max=True)

    def update_target(self):
        soft_update(self.target_q_network, self.q_network, self.tau)

    def _compute_next_obs_q(self, next_obs, max_backup=True):
        """ Max backup """
        with torch.no_grad():
            batch_size = next_obs.shape[0]
            next_obs = torch.tile(next_obs, (self.num_samples, 1))
            actions = self.policy_net.select_action(obs=next_obs, deterministic=False)
            q_values = self.target_q_network(obs=next_obs, act=actions, take_min=True)
            q_values = torch.reshape(q_values, (self.num_samples, batch_size))  # (num_samples, None)
            if max_backup:
                q_values = torch.max(q_values, dim=0)[0]
            else:
                q_values = torch.mean(q_values, dim=0)
            return q_values

    def train_nets_behavior_cloning(self, obs, act):
        self.policy_optimizer.zero_grad(set_to_none=True)
        log_prob_data, _ = self.policy_net.compute_log_prob(obs=obs, act=act)
        policy_loss = torch.mean(-log_prob_data, dim=0)
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss

    def train_nets_cql_pytorch(self, obs, act, next_obs, rew, done):
        # update
        with torch.no_grad():
            alpha = self.log_alpha()
            alpha_cql = self.log_cql()
            batch_size = obs.shape[0]
            next_q_values = self._compute_next_obs_q(next_obs, max_backup=self.max_backup)

            q_target = rew + self.gamma * (1.0 - done) * next_q_values

        # q loss
        self.q_optimizer.zero_grad(set_to_none=True)
        q_values = self.q_network(obs=obs, act=act, take_min=False)
        mse_q_values_loss = 0.5 * torch.square(torch.unsqueeze(q_target, dim=0) - q_values)  # (num_ensembles, None)
        mse_q_values_loss = torch.mean(torch.sum(mse_q_values_loss, dim=0), dim=0)  # scalar

        # in-distribution q values is simply q_values
        in_distribution_q_values = q_values  # (num_ensembles, None)

        # max_a Q(s,a). Current Q values
        with torch.no_grad():
            obs_tile = torch.tile(obs, (self.num_samples, 1))
            next_obs_tile = torch.tile(next_obs, (self.num_samples, 1))

            # current actions
            actions, log_prob, _, _ = self.policy_net(obs=obs_tile,
                                                      deterministic=False)  # (num_samples * None, act_dim)

            # next_obs actions
            next_obs_actions, next_obs_log_prob, _, _ = self.policy_net(obs=next_obs_tile, deterministic=False)

            # random actions
            pi_random_actions = torch.rand(size=(self.num_samples * batch_size, self.act_dim),
                                           device=self.device) * 2. - 1.  # [-1., 1]
            log_prob_random = torch.ones_like(log_prob) * -LOG_2 * self.act_dim  # uniform from [-1, 1], prob=0.5

        obs_tile_tile = torch.cat((obs_tile, obs_tile, obs_tile), dim=0)
        act_tile_tile = torch.cat((actions, next_obs_actions, pi_random_actions), dim=0)
        log_prob_tile_tile = torch.cat((log_prob, next_obs_log_prob, log_prob_random), dim=0)

        # shape (3 * num_samples, None)
        log_prob_tile_tile = torch.unsqueeze(log_prob_tile_tile, dim=0)
        cql_q_values = self.q_network(obs=obs_tile_tile, act=act_tile_tile, take_min=False) - log_prob_tile_tile

        # (3 * num_samples, None)
        cql_q_values = torch.reshape(cql_q_values, shape=(self.num_ensembles, 3 * self.num_samples, batch_size))
        # (num_ensemble, None)
        cql_q_values = torch.logsumexp(cql_q_values / self.temperature, dim=1) * self.temperature
        cql_threshold = torch.mean(torch.sum(cql_q_values - in_distribution_q_values, dim=0), dim=0) * self.min_q_weight

        q_loss = mse_q_values_loss + alpha_cql * cql_threshold
        q_loss.backward()
        self.q_optimizer.step()

        # update alpha_cql
        self.cql_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_cql = self.log_cql()
        delta_cql = cql_threshold - self.cql_threshold
        alpha_cql_loss = -alpha_cql * delta_cql.detach()
        alpha_cql_loss.backward()
        self.cql_alpha_optimizer.step()

        # update policy
        self.policy_optimizer.zero_grad(set_to_none=True)
        action, log_prob, _, _ = self.policy_net(obs=obs, deterministic=False)
        q_values_pi_min = self.q_network(obs=obs, act=action, take_min=True)
        policy_loss = torch.mean(log_prob * alpha - q_values_pi_min, dim=0)
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha = self.log_alpha()
        alpha_loss = -torch.mean(alpha * (log_prob.detach() + self.target_entropy))
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        info = dict(
            Q1Vals=q_values[0],
            Q2Vals=q_values[1],
            LogPi=log_prob,
            Alpha=alpha,
            LossQ=mse_q_values_loss,
            LossAlpha=alpha_loss,
            LossPi=policy_loss,
            AlphaCQL=alpha_cql,
            AlphaCQLLoss=alpha_cql_loss,
            DeltaCQL=cql_threshold,
        )
        return info

    def train_on_batch(self, data):
        data = ptu.convert_dict_to_tensor(data, self.device)
        info = self.train_nets_cql_pytorch(**data)
        self.update_target()
        self.logger.store(**info)

    def act_batch_test(self, obs):
        obs = torch.as_tensor(obs).to(self.device)
        return self.act_batch_test_pytorch(obs).cpu().numpy()

    def act_batch_test_pytorch(self, obs):
        with torch.no_grad():
            batch_size = obs.shape[0]
            obs_tile = torch.tile(obs, (self.num_samples, 1))
            actions = self.policy_net.select_action(obs=obs_tile, deterministic=False)  # (num_samples * None, act_dim)
            q_values = self.q_network(obs=obs_tile, act=actions, take_min=True)  # (num_samples * None)
            q_values = torch.reshape(q_values, shape=(self.num_samples, batch_size))
            max_idx = torch.max(q_values, dim=0)[1]  # (None)
            max_idx = torch.tile(max_idx, (self.act_dim,))  # (None * act_dim,)
            actions = torch.reshape(actions, shape=(
                self.num_samples, batch_size * self.act_dim))  # (num_samples, None * act_dim)
            actions = actions.gather(0, max_idx.unsqueeze(0)).squeeze(0)  # (None * act_dim)
            actions = torch.reshape(actions, shape=(batch_size, self.act_dim))
            return actions


def run_d4rl_cql(env_name: str,
                 exp_name: str = None,
                 asynchronous=False,
                 # agent
                 policy_mlp_hidden=256,
                 policy_lr=1e-4,
                 q_mlp_hidden=256,
                 q_lr=3e-4,
                 alpha=0.02,
                 tau=5e-3,
                 gamma=0.99,
                 cql_threshold=-1,
                 temperature=1.,
                 min_q_weight=1.,
                 # runner args
                 epochs=250,
                 steps_per_epoch=4000,
                 num_test_episodes=30,
                 batch_size=256,
                 seed=1,
                 behavior_cloning_epochs=10,
                 logger_path: str = None
                 ):
    config = locals()

    # setup seed
    seeder = Seeder(seed=seed, backend='torch')
    seeder.setup_global_seed()

    # environment
    def env_fn():
        env = gym.make(env_name)
        env = TransformObservationDtype(env, dtype=np.float32)
        return env

    # agent
    env = env_fn()
    verify_continuous_action_space(env.action_space)

    dataset = d4rl.qlearning_dataset(env)
    dataset['obs'] = dataset.pop('observations').astype(np.float32)
    dataset['act'] = dataset.pop('actions').astype(np.float32)
    dataset['next_obs'] = dataset.pop('next_observations').astype(np.float32)
    dataset['rew'] = dataset.pop('rewards').astype(np.float32)
    dataset['done'] = dataset.pop('terminals').astype(np.float32)

    # normalized reward to [0, 1]
    max_reward = np.max(dataset['rew'])
    min_reward = np.min(dataset['rew'])
    dataset['rew'] = (dataset['rew'] - min_reward) / (max_reward - min_reward)

    agent = CQLContinuousAgent(env=env, policy_lr=policy_lr, policy_mlp_hidden=policy_mlp_hidden,
                               q_mlp_hidden=q_mlp_hidden, q_lr=q_lr, alpha=alpha,
                               tau=tau, gamma=gamma, cql_threshold=cql_threshold,
                               temperature=temperature, min_q_weight=min_q_weight,
                               device=ptu.get_accelerator())

    # setup logger
    if exp_name is None:
        exp_name = f'{env_name}_{agent.__class__.__name__}_test'
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name, data_dir=logger_path, seed=seed)
    logger = EpochLogger(**logger_kwargs, tensorboard=False)
    logger.save_config(config)

    timer = StopWatch()

    # replay buffer
    replay_buffer = UniformReplayBuffer.from_dataset(dataset=dataset, seed=seeder.generate_seed())

    # setup tester
    tester = D4RLTester(env_fn=env_fn, num_parallel_env=num_test_episodes,
                        asynchronous=asynchronous, seed=seeder.generate_seed())

    # register log_tabular ops
    timer.logger = logger
    agent.logger = logger
    tester.logger = logger

    logger.register(timer.log_tabular)
    logger.register(tester.log_tabular)
    logger.register(agent.log_tabular)

    timer.start()
    policy_updates = 0

    # fit via behavior cloning and compute the entropy
    from library.replay_buffer import create_dict_data_loader
    from tqdm.auto import tqdm

    data_loader = create_dict_data_loader(tensors=dataset,
                                          batch_size=batch_size,
                                          subset_keys={'obs', 'act'})

    for epoch in range(1, behavior_cloning_epochs + 1):
        for data in tqdm(data_loader, desc=f'Pretraining Epoch {epoch}/{behavior_cloning_epochs}'):
            data = ptu.convert_dict_to_tensor(data, device=agent.device)
            policy_loss = agent.train_nets_behavior_cloning(**data)

    # main training loop
    for epoch in range(1, epochs + 1):
        for t in trange(steps_per_epoch, desc=f'Epoch {epoch}/{epochs}'):
            # Update handling
            batch = replay_buffer.sample(batch_size)
            agent.train_on_batch(data=batch)
            policy_updates += 1

        tester.test_agent(get_action=lambda obs: agent.act_batch_test(obs),
                          name=agent.__class__.__name__,
                          num_test_episodes=num_test_episodes)
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('PolicyUpdates', policy_updates)
        logger.dump_tabular()


if __name__ == '__main__':
    from library.infrastructure.commandline_utils import run_func_as_main

    run_func_as_main(func=run_d4rl_cql)
