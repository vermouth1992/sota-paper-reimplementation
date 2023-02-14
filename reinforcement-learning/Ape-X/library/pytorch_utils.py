import numpy as np
import torch
from torch import nn


def convert_dict_to_tensor(data, device=None):
    tensor_data = {}
    for key, d in data.items():
        if not isinstance(d, np.ndarray):
            d = np.array(d)
        tensor_data[key] = torch.as_tensor(d).to(device, non_blocking=True)
    return tensor_data


def get_accelerator(id=None):
    if not torch.cuda.is_available():
        print('Cuda is not available. Using cpu.')
        return torch.device('cpu')
    else:
        if id is None:
            return torch.device('cuda')
        else:
            assert isinstance(id, int)
            total_device_count = torch.cuda.device_count()
            if id < 0 or id >= torch.cuda.device_count():
                print(f'id {id} exceeds total device count {total_device_count}')
                return torch.device('cuda')
            else:
                cuda_device = f'cuda:{id}'
                return torch.device(cuda_device)


def get_state_dict(module: nn.Module, device=None):
    state_dict = module.state_dict()
    for k, v in state_dict.items():
        if device is not None:
            state_dict[k] = v.to(device)
    return state_dict
