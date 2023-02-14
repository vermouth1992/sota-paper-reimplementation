import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as td
from torch import nn


def _compute_rank(tensor):
    return len(tensor.shape)


def apply_squash_log_prob(raw_log_prob, x):
    """ Compute the log probability after applying tanh on raw_actions

    Args:
        log_prob: (None,)
        raw_actions: (None, act_dim)

    Returns:
    """
    log_det_jacobian = 2. * (np.log(2.) - x - F.softplus(-2. * x))
    num_reduce_dim = _compute_rank(x) - _compute_rank(raw_log_prob)
    log_det_jacobian = torch.sum(log_det_jacobian, dim=list(range(-num_reduce_dim, 0)))
    log_prob = raw_log_prob - log_det_jacobian
    return log_prob


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, mlp_hidden):
        super(SquashedGaussianMLPActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 2 * act_dim)
        )
        self.act_dim = act_dim

    def pi_dist_layer(self, params):
        loc_params, scale_params = torch.split(params, params.shape[-1] // 2, dim=-1)
        scale_params = torch.clip(scale_params, min=-10., max=5.)
        scale_params = F.softplus(scale_params)
        distribution = td.Independent(base_distribution=td.Normal(loc=loc_params, scale=scale_params),
                                      reinterpreted_batch_ndims=1)
        return distribution

    def select_action(self, obs, deterministic: bool):
        params = self.net(obs)
        pi_distribution = self.pi_dist_layer(params)
        if deterministic:
            pi_action = pi_distribution.mean
        else:
            pi_action = pi_distribution.rsample()
        pi_action_final = torch.tanh(pi_action)
        return pi_action_final

    def compute_pi_distribution(self, obs):
        return self.pi_dist_layer(self.net(obs))

    def transform_raw_actions(self, raw_actions):
        return torch.tanh(raw_actions)

    def compute_raw_actions(self, actions):
        EPS = 1e-6
        actions = torch.clip(actions, min=-1. + EPS, max=1. - EPS)
        return torch.atanh(actions)

    def compute_log_prob(self, obs, act):
        params = self.net(obs)
        pi_distribution = self.pi_dist_layer(params)
        # compute actions
        pi_action = pi_distribution.rsample()
        raw_act = self.compute_raw_actions(act)
        # compute log probability
        log_prob = pi_distribution.log_prob(raw_act)
        log_prob = apply_squash_log_prob(log_prob, raw_act)
        log_prob_pi = pi_distribution.log_prob(pi_action)
        log_prob_pi = apply_squash_log_prob(log_prob_pi, pi_action)
        return log_prob, log_prob_pi

    def forward(self, obs, deterministic: bool):
        params = self.net(obs)
        pi_distribution = self.pi_dist_layer(params)
        if deterministic:
            pi_action = pi_distribution.mean
        else:
            pi_action = pi_distribution.rsample()
        logp_pi = pi_distribution.log_prob(pi_action)
        logp_pi = apply_squash_log_prob(logp_pi, pi_action)
        pi_action_final = torch.tanh(pi_action)
        return pi_action_final, logp_pi, pi_action, pi_distribution


from .layers import EnsembleLinear, SqueezeLayer


class EnsembleMinQNet(nn.Module):
    def __init__(self, obs_dim, act_dim, mlp_hidden, num_ensembles=2):
        super(EnsembleMinQNet, self).__init__()
        self.mlp_hidden = mlp_hidden
        self.num_ensembles = num_ensembles
        self.q_net = nn.Sequential(
            EnsembleLinear(num_ensembles, obs_dim + act_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            EnsembleLinear(num_ensembles, mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            EnsembleLinear(num_ensembles, mlp_hidden, 1),
            SqueezeLayer(dim=-1)
        )

    def forward(self, obs, act, take_min=True):
        inputs = torch.cat((obs, act), dim=-1)
        inputs = torch.unsqueeze(inputs, dim=0)  # (1, None, obs_dim + act_dim)
        inputs = inputs.repeat(self.num_ensembles, 1, 1)
        q = self.q_net(inputs)  # (num_ensembles, None)
        if not take_min:
            return q
        else:
            return torch.min(q, dim=0)[0]
