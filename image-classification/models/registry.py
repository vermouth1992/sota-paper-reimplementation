from torch import nn

from typing import List

MODELS = {}


def register_model(task, **kwargs):
    if task not in MODELS:
        MODELS[task] = dict()

    def _inner_register_model(model_fn):
        MODELS[task][model_fn.__name__] = (model_fn, kwargs)
        return model_fn

    return _inner_register_model


def create_model(task, name, **kwargs) -> nn.Module:
    model_fn, default_kwargs = MODELS[task][name]
    for key in default_kwargs:
        if key in kwargs:
            default_kwargs[key] = kwargs[key]
    return model_fn(**default_kwargs)


def get_available_models(task) -> List[str]:
    return list(MODELS[task].keys())
