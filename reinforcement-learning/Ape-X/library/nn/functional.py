from torch import nn


def make_module_trainable(module: nn.Module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable
