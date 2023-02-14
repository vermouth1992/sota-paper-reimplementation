import gym
import numpy as np


def verify_continuous_action_space(act_spec: gym.spaces.Box):
    assert np.max(act_spec.high) == np.min(act_spec.high), \
        f'Not all the values in high are the same. Got {act_spec.high}'
    assert np.max(act_spec.low) == np.min(act_spec.low), \
        f'Not all the values in low are the same. Got {act_spec.low}'
    assert act_spec.high[0] + act_spec.low[0] == 0., f'High is not equal to low'
    assert act_spec.high[0] == 1.0


from gym import spaces


class TransformObservationDtype(gym.wrappers.TransformObservation):
    def __init__(self, env: gym.Env, dtype):
        super(TransformObservationDtype, self).__init__(env, f=lambda x: x.astype(dtype))
        assert isinstance(env.observation_space, spaces.Box)
        self.observation_space = spaces.Box(low=self.env.observation_space.low,
                                            high=self.env.observation_space.high,
                                            shape=self.env.observation_space.shape,
                                            dtype=dtype)
