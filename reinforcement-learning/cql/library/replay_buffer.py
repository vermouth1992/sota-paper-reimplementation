import threading
from typing import Dict, Union

import gym
import numpy as np
from gym.utils import seeding


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def get_data_spec_from_dataset(dataset: Dict[str, np.ndarray], obj_keys=None):
    if obj_keys is None:
        obj_keys = set()
    data_spec = {}
    data_size = None
    for key, data in dataset.items():
        if key in obj_keys:
            print(f'Store key {key} as object')
            data_spec[key] = None
        else:
            data_spec[key] = gym.spaces.Space(shape=data.shape[1:], dtype=data.dtype)

        if data_size is None:
            data_size = data.shape[0]
        else:
            assert data_size == data.shape[0]
    return data_spec, data_size


def get_data_spec_from_env(env, memory_efficient=False):
    if memory_efficient:
        obs_spec = None
    else:
        obs_spec = env.observation_space

    act_spec = env.action_space

    data_spec = {
        'obs': obs_spec,
        'act': act_spec,
        'next_obs': obs_spec,
        'rew': gym.spaces.Space(shape=None, dtype=np.float32),
        'done': gym.spaces.Space(shape=None, dtype=np.float32),
        'gamma': gym.spaces.Space(shape=None, dtype=np.float32)
    }
    return data_spec


class PyDictStorage(object):
    def __init__(self, data_spec: Dict[str, Union[gym.spaces.Space, None]], capacity):
        self.data_spec = data_spec
        self.max_size = capacity
        self.storage = self._create_storage()
        self.reset()

    def is_full(self):
        return len(self) == self.capacity

    def is_empty(self):
        return len(self) <= 0

    def _create_storage(self):
        storage = {}
        self.np_key = []
        self.obj_key = []
        for key, item in self.data_spec.items():
            if isinstance(item, gym.spaces.Space):
                storage[key] = np.zeros(combined_shape(self.capacity, item.shape), dtype=item.dtype)
                self.np_key.append(key)
            elif item is None:
                print(f"Store key {key} as an object")
                storage[key] = np.zeros(self.capacity, dtype=object)
                self.obj_key.append(key)
            else:
                raise ValueError(f'Unknonw type item {type(item)}')
        return storage

    def reset(self):
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        data = {key: self.storage[key][item] for key in self.np_key}
        for key in self.obj_key:
            output = []
            for idx in item:
                output.append(self.storage[key][idx])
            data[key] = output
        return data

    @property
    def capacity(self):
        return self.max_size

    def get_available_indexes(self, batch_size):
        if self.ptr + batch_size > self.max_size:
            index = np.concatenate((np.arange(self.ptr, self.capacity),
                                    np.arange(batch_size - (self.capacity - self.ptr))), axis=0)
            # print('Reaches the end of the replay buffer')
        else:
            index = np.arange(self.ptr, self.ptr + batch_size)
        return index

    def add(self, data: Dict[str, np.ndarray], index=None):
        batch_size = len(data[self.np_key[0]])
        if index is None:
            index = self.get_available_indexes(batch_size)
        for key, item in data.items():
            if key in self.np_key:
                self.storage[key][index] = item
            elif key in self.obj_key:
                for i in range(batch_size):
                    self.storage[key][(self.ptr + i) % self.max_size] = item[i]
            else:
                raise ValueError(f'Unknown type {type(item)}')

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
        return index

    def get(self):
        return self[np.arange(len(self))]


class UniformReplayBuffer(object):
    def __init__(self, capacity, data_spec, seed=None):
        self.data_spec = data_spec
        self.storage = PyDictStorage(self.data_spec, capacity)
        self.set_seed(seed)

        self.lock = threading.Lock()

        self.storage.reset()

    def __len__(self):
        with self.lock:
            return len(self.storage)

    def add(self, data):
        with self.lock:
            self.storage.add(data)

    @property
    def capacity(self):
        return self.storage.capacity

    def is_full(self):
        with self.lock:
            return self.storage.is_full()

    def is_empty(self):
        with self.lock:
            return self.storage.is_empty()

    def set_seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)

    def sample(self, batch_size):
        assert not self.is_empty()
        with self.lock:
            idxs = self.np_random.integers(0, len(self.storage), size=batch_size)
            data = self.storage[idxs]
            for key in self.storage.obj_key:
                data[key] = np.array(data[key])
            return data

    @classmethod
    def from_env(cls, env, memory_efficient, **kwargs):
        data_spec = get_data_spec_from_env(env, memory_efficient=memory_efficient)
        return cls(data_spec=data_spec, **kwargs)

    @classmethod
    def from_dataset(cls, dataset: Dict[str, np.ndarray], obj_keys=None, **kwargs):
        # sanity check
        if obj_keys is None:
            obj_keys = set()
        data_spec, capacity = get_data_spec_from_dataset(dataset, obj_keys=obj_keys)
        replay_buffer = cls(data_spec=data_spec, capacity=capacity, **kwargs)
        replay_buffer.add(dataset)
        assert replay_buffer.is_full()
        return replay_buffer


from torch.utils import data
from typing import Dict
import torch


class DictDataset(data.Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor], subset_keys=None):
        self.tensors = tensors
        self.batch_size = None
        if subset_keys is None:
            self.subset_keys = tensors.keys()
        else:
            self.subset_keys = subset_keys

        for key, tensor in tensors.items():
            if self.batch_size is None:
                self.batch_size = tensor.shape[0]
            else:
                assert self.batch_size == tensor.shape[0]

        for key in self.subset_keys:
            assert key in tensors.keys()

    def __getitem__(self, item):
        return {key: self.tensors[key][item] for key in self.subset_keys}

    def __len__(self):
        return self.batch_size


def create_dict_data_loader(tensors: Dict[str, torch.Tensor], batch_size, subset_keys=None):
    dataset = DictDataset(tensors, subset_keys=subset_keys)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
