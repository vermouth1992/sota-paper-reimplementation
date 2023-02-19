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

    def reset_parameters_linear(self, weight, bias):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

    def reset_parameters(self) -> None:
        for i in range(self.num_ensembles):
            bias = self.bias[i, 0] if self.bias is not None else None
            self.reset_parameters_linear(self.weight[i], bias)

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


if __name__ == '__main__':
    # verify EnsembleLinear vs. Linear output and gradients
    ensemble_linear = EnsembleLinear(num_ensembles=2, in_features=5, out_features=10)
    linear1 = nn.Linear(in_features=5, out_features=10)
    linear2 = nn.Linear(in_features=5, out_features=10)
    linear1.weight.data.copy_(ensemble_linear.weight.data[0].T)
    linear1.bias.data.copy_(ensemble_linear.bias.data[0, 0])
    linear2.weight.data.copy_(ensemble_linear.weight.data[1].T)
    linear2.bias.data.copy_(ensemble_linear.bias.data[1, 0])

    fake_data = torch.randn(100, 5)
    ensemble_output = ensemble_linear(torch.tile(torch.unsqueeze(fake_data, dim=0), dims=(2, 1, 1)))
    output1 = linear1(fake_data)
    output2 = linear2(fake_data)
    output = torch.stack((output1, output2), dim=0)
    torch.testing.assert_close(output, ensemble_output)

    ensemble_output.mean().backward()
    ((output1 + output2) / 2.).mean().backward()

    torch.testing.assert_close(linear1.weight.grad, ensemble_linear.weight.grad[0].T)
    torch.testing.assert_close(linear2.weight.grad, ensemble_linear.weight.grad[1].T)
    torch.testing.assert_close(linear1.bias.grad, ensemble_linear.bias.grad[0, 0])
    torch.testing.assert_close(linear2.bias.grad, ensemble_linear.bias.grad[1, 0])
