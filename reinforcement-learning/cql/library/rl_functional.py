import torch
from torch import nn


def soft_update(target: nn.Module, source: nn.Module, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data.to(target_param.data.device) * tau)


def hard_update(target: nn.Module, source: nn.Module):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data.to(target_param.data.device))
