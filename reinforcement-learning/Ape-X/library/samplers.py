import gym
import numpy as np
import collections


class BatchSampler(object):
    def __init__(self, env, n_steps, gamma, seed):
        self.env = env
        self.seed = seed
        self.n_steps = n_steps
        self.gamma = gamma
        self.gamma_vector = gamma ** np.arange(self.n_steps)  # (n_steps,)
        self.oa_queue = collections.deque(maxlen=n_steps)
        self.rew_queue = collections.deque(maxlen=n_steps)
        self.logger = None
        self.reset()

    @property
    def total_env_steps(self):
        return self._global_env_step

    def reset(self):
        self._global_env_step = 0
        self.reset_episode(seed=self.seed)

    def reset_episode(self, seed=None):
        self.o, info = self.env.reset(seed=seed)
        self.ep_ret = 0.
        self.ep_len = 0

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self._global_env_step)

    def sample(self, collect_fn):
        while True:
            a = collect_fn(self.o)
            assert not np.any(np.isnan(a)), f'NAN action: {a}'
            # Step the env
            next_obs, r, terminate, truncate, infos = self.env.step(a)
            self.ep_ret += r
            self.ep_len += 1

            d = terminate or truncate

            true_d = terminate  # affect value function boostrap

            self.oa_queue.append((self.o, a))
            self.rew_queue.append(r)

            if len(self.oa_queue) >= self.n_steps:
                last_o, last_a = self.oa_queue.popleft()
                last_r = np.sum(np.array(self.rew_queue) * self.gamma_vector)
                self.rew_queue.popleft()

                yield dict(
                    obs=last_o,
                    act=last_a,
                    rew=last_r,
                    next_obs=next_obs,
                    done=true_d,
                    gamma=self.gamma ** self.n_steps
                )

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            self.o = next_obs

            # End of trajectory handling
            if d:
                # empty queues
                for steps in range(self.n_steps - 1, 0, -1):
                    last_o, last_a = self.oa_queue.popleft()
                    gamma_vector = self.gamma ** np.arange(steps)
                    last_r = np.sum(np.array(self.rew_queue) * gamma_vector)
                    self.rew_queue.popleft()

                    yield dict(
                        obs=last_o,
                        act=last_a,
                        rew=last_r,
                        next_obs=next_obs,
                        done=true_d,
                        gamma=self.gamma ** steps
                    )

                if self.logger is not None:
                    self.logger.store(EpRet=self.ep_ret, EpLen=self.ep_len)
                self.reset_episode()

            self._global_env_step += 1


if __name__ == '__main__':
    from typing import Optional


    class DummyEnv(gym.Env):
        def __init__(self, terminate_steps=10, max_episode_steps=20):
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.int32)
            self.action_space = gym.spaces.Discrete(n=2)
            self.terminate_steps = terminate_steps
            self.max_episode_steps = max_episode_steps

        def step(self, action):
            self.t += 1
            reward = 1.
            terminate = self.t >= self.terminate_steps
            truncated = self.t >= self.max_episode_steps
            return self._get_obs(), reward, terminate, truncated, {}

        def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict] = None,
        ):
            self.t = 0
            return self._get_obs(), {}

        def _get_obs(self):
            return self.t


    env = DummyEnv(terminate_steps=10, max_episode_steps=9)
    sampler = BatchSampler(env=env, n_steps=3, gamma=0.99, seed=1)
    total_steps = 0
    for samples in sampler.sample(collect_fn=lambda o: env.action_space.sample()):
        total_steps += 1
        print(samples)

        if total_steps >= 20:
            break
