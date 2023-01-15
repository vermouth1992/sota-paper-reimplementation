import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(x, beta=1.):
    assert x > 0, 'x must be positive'
    if x < 20:
        return np.log(np.exp(x * beta) - 1.) / beta
    else:
        return x


def clip_by_value_preserve_gradient(t, clip_value_min=None, clip_value_max=None):
    clip_t = torch.clip(t, min=clip_value_min, max=clip_value_max)
    return t + (clip_t - t).detach()


class EnsembleBatchNorm1d(nn.Module):
    def __init__(self, num_ensembles, num_features, **kwargs):
        super(EnsembleBatchNorm1d, self).__init__()
        self.num_ensembles = num_ensembles
        self.num_features = num_features
        self.batch_norm_layer = nn.BatchNorm1d(num_features=num_features * num_ensembles, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: shape (num_ensembles, None, num_features)
        Returns:
        """
        batch_size = input.shape[1]
        input = input.permute(1, 0, 2)  # (None, num_ensembles, num_features)
        input = input.reshape(batch_size, self.num_ensembles * self.num_features)
        output = self.batch_norm_layer(input)  # (None, num_ensembles, num_features)
        output = output.view(batch_size, self.num_ensembles, self.num_features)
        output = output.permute(1, 0, 2)  # (num_ensembles, None, num_features)
        return output


class EnsembleLinear(nn.Module):
    __constants__ = ['num_ensembles', 'in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, num_ensembles: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(EnsembleLinear, self).__init__()
        self.num_ensembles = num_ensembles
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_ensembles, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_ensembles, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan = self.in_features
        gain = nn.init.calculate_gain('leaky_relu', param=math.sqrt(5))
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.bmm(input, self.weight) + self.bias

    def extra_repr(self) -> str:
        return 'num_ensembles={}, in_features={}, out_features={}, bias={}'.format(
            self.num_ensembles, self.in_features, self.out_features, self.bias is not None
        )


class SqueezeLayer(nn.Module):
    def __init__(self, dim=-1):
        super(SqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.squeeze(inputs, dim=self.dim)


class LagrangeLayer(nn.Module):
    def __init__(self, initial_value=0., min_value=None, max_value=10000., enforce_positive_func=F.softplus):
        super(LagrangeLayer, self).__init__()
        self.log_alpha = nn.Parameter(data=torch.as_tensor(inverse_softplus(initial_value), dtype=torch.float32))
        self.min_value = min_value
        self.max_value = max_value
        self.enforce_positive_func = enforce_positive_func

    def forward(self):
        alpha = self.enforce_positive_func(self.log_alpha)
        return clip_by_value_preserve_gradient(alpha, clip_value_min=self.min_value, clip_value_max=self.max_value)


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super(LambdaLayer, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)
