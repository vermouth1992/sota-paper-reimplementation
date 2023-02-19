"""
Implement parallel SPO using ray.
The logging is done by the number of policy updates
"""

import os
import threading
import time
from typing import Dict

import gym
import numpy as np
import ray
from absl import flags, app
from ray.util import queue

import library.pytorch_utils as ptu
from library.infrastructure import logger as logx

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_cpus_replay', 1, 'Number of cpus used for ReplayManager')
flags.DEFINE_integer('update_after', 10000, 'Number of environment steps before update begins')
flags.DEFINE_integer('replay_capacity', 1000000, 'Capacity of the replay buffer')
flags.DEFINE_float('replay_alpha', 0.6, 'alpha of the Prioritized Replay Buffer')
flags.DEFINE_float('replay_beta', 0.4, 'beta of the Prioritized Replay Buffer')
flags.DEFINE_float('replay_eviction', None, 'eviction of the Prioritized Replay Buffer')


class ReplayManager(object):
    def __init__(self, data_spec, seed,
                 capacity,
                 alpha,
                 beta,
                 eviction,
                 update_after
                 ):
        from library.replay_buffer import PrioritizedReplayBuffer
        self.replay_buffer = PrioritizedReplayBuffer(data_spec=data_spec,
                                                     capacity=capacity,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     eviction=eviction,
                                                     seed=seed)
        self.update_after = update_after
        self._global_env_step = 0
        self.start_time = None

    def sample(self, batch_size):
        data = self.replay_buffer.sample(batch_size)
        return data

    def get_stats(self):
        return {
            'Samples/s': self._global_env_step / (time.time() - self.start_time),
            'TotalEnvInteracts': self._global_env_step
        }

    def update_priorities(self, transaction_id, priorities):
        self.replay_buffer.update_priorities(transaction_id, priorities)

    def ready(self):
        return len(self.replay_buffer) >= self.update_after, len(self.replay_buffer)

    def add(self, data: Dict[str, np.ndarray], priority=None):
        if self.start_time is None:
            self.start_time = time.time()
        self.replay_buffer.add(data, priority)
        self._global_env_step += data[list(data.keys())[0]].shape[0]

    def data_spec(self):
        return self.replay_buffer.data_spec


class Dataset(object):
    def __init__(self, replay_buffer_actor, batch_size):
        self.replay_buffer_actor = replay_buffer_actor
        self.batch_size = batch_size

    def __call__(self):
        while True:
            yield ray.get(self.replay_buffer_actor.sample.remote(batch_size=self.batch_size))


import tensorflow as tf


def _convert_gym_space_to_tf_tensorspec(space: gym.spaces.Space, batch_size):
    if space.shape is None:
        shape = ()
    else:
        shape = space.shape
    return tf.TensorSpec(shape=(batch_size,) + tuple(shape), dtype=space.dtype)


def _set_torch_threads_tf32(num_threads, allow_tf32=True):
    import torch
    torch.set_num_threads(num_threads)
    import torch.backends.cuda
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32


flags.DEFINE_integer('num_cpus_learner', default=1, help='Number of CPUs per learner')
flags.DEFINE_boolean('use_gpu_learner', default=True, help='Whether use GPU for the learner')
flags.DEFINE_integer('weight_push_freq', default=10, help='Frequency of learner pushes its weights')
flags.DEFINE_integer('batch_size', default=256, help='Batch size for training')
flags.DEFINE_integer('prefetch', default=5, help='Batch prefetch from ReplayManager')
flags.DEFINE_integer('logging_freq', default=5000, help='Number of policy updates per logging')


class Learner(object):
    def __init__(self,
                 make_agent_fn,
                 replay_manager,
                 testing_queue: queue.Queue,
                 num_cpus_learner,
                 prefetch,
                 batch_size,
                 weight_push_freq,
                 logging_freq
                 ):
        import tensorflow as tf
        import tensorflow_datasets as tfds

        _set_torch_threads_tf32(num_cpus_learner, True)

        self.agent = make_agent_fn()
        self.replay_manager = replay_manager
        data_spec = ray.get(self.replay_manager.data_spec.remote())
        output_signature = {key: _convert_gym_space_to_tf_tensorspec(space, batch_size)
                            for key, space in data_spec.items()}
        output_signature['weights'] = tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
        output_signature = (tf.TensorSpec(shape=(), dtype=tf.int32), output_signature)
        print(output_signature)
        self.dataset = tf.data.Dataset.from_generator(Dataset(self.replay_manager, batch_size),
                                                      output_signature=output_signature
                                                      ).prefetch(prefetch)
        self.dataset = tfds.as_numpy(self.dataset)
        self.batch_size = batch_size
        self.weight_push_freq = weight_push_freq
        self.policy_updates = 0

        from library.infrastructure.logger import EpochLogger
        self.logger = EpochLogger()
        self.agent.set_logger(self.logger)

        self.store_weights()

        self.testing_queue = testing_queue
        self.logging_freq = logging_freq

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = ptu.get_state_dict(self.agent, device='cpu')
        self.weights_id = ray.put(state_dict)

    def get_stats(self):
        stats = self.logger.get_epoch_dict()
        self.logger.clear_epoch_dict()
        return stats

    def run(self):
        background_thread = threading.Thread(target=self.train, daemon=True)
        background_thread.start()

    def train(self):
        start = time.time()
        for transaction_id, data in self.dataset:
            info = self.agent.train_on_batch(data)
            td_error = ptu.to_numpy(info['TDError'])
            self.replay_manager.update_priorities.remote(transaction_id, td_error)
            self.policy_updates += 1

            # push weights
            if self.policy_updates % self.weight_push_freq:
                self.store_weights()

            # testing
            if self.policy_updates % self.logging_freq == 0:
                # perform testing and logging
                try:
                    self.testing_queue.put((self.policy_updates,
                                            self.policy_updates / (time.time() - start)),
                                           block=False)
                except queue.Full:
                    logx.log('Testing queue is full. Skip this epoch', color='red')


from library.samplers import BatchSampler

flags.DEFINE_integer('num_cpus_actor', default=1, help='Number of CPUs per actor')
flags.DEFINE_integer('n_steps', default=1, help='N-steps')
flags.DEFINE_float('gamma', default=0.99, help='Discount factor')
flags.DEFINE_integer('weight_update_freq', default=100, help='Weight update frequency')
flags.DEFINE_integer('local_capacity', default=1000, help='Number of environment steps before adding data')


class Actor(object):
    def __init__(self, make_agent_fn, env_train_fn, learner, replay_manager, seed,
                 num_cpus_actor,
                 weight_update_freq,
                 local_capacity,
                 n_steps,
                 gamma,
                 ):
        _set_torch_threads_tf32(num_cpus_actor, True)

        self.agent = make_agent_fn()
        self.agent.eval()
        self.learner = learner
        self.replay_manager = replay_manager
        self.weight_update_freq = weight_update_freq
        self.local_capacity = local_capacity
        self.sampler = BatchSampler(env=env_train_fn(), n_steps=n_steps, gamma=gamma, seed=seed)

        from library.infrastructure.logger import EpochLogger
        from library.replay_buffer.replay_buffer import PyDictStorage
        self.logger = EpochLogger()
        self.sampler.set_logger(self.logger)
        data_spec = ray.get(self.replay_manager.data_spec.remote())
        self.local_storage = PyDictStorage(data_spec=data_spec, capacity=local_capacity)

    def get_stats(self):
        stats = self.logger.get_epoch_dict()
        self.logger.clear_epoch_dict()
        return stats

    def run(self):
        background_thread = threading.Thread(target=self.train, daemon=True)
        background_thread.start()

    def add_local_to_global(self, data, priority):
        # pick a random replay manager
        replay_manager = self.replay_manager
        replay_manager.add.remote(data, priority)

    def train(self):
        self.update_weights()
        self.sampler.reset()
        local_steps = 0
        collect_fn = lambda o: self.agent.act_batch_explore(np.expand_dims(o, axis=0), None)[0]
        for data in self.sampler.sample(collect_fn=collect_fn):
            data = {key: np.expand_dims(val, axis=0) for key, val in data.items()}
            self.local_storage.add(data=data)

            if self.local_storage.is_full():
                data = self.local_storage.get()
                priority = self.agent.compute_priority(data)
                self.add_local_to_global(data, priority)
                self.local_storage.reset()

            if local_steps % self.weight_update_freq == 0:
                self.update_weights()

    def update_weights(self):
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.agent.load_state_dict(weights)


flags.DEFINE_integer('total_num_policy_updates', default=1000000, help='Total number of policy updates')
flags.DEFINE_integer('num_test_episodes', default=30, help='Number of test episodes')
flags.DEFINE_string('logger_path', default=None, help='Logger path')
flags.DEFINE_integer('seed', default=100, help='Global seed')
flags.DEFINE_string('exp_name', default=None, help='Name of the experience')


class Logger(object):
    def __init__(self,
                 receive_queue: queue.Queue,
                 actors,
                 learner,
                 replay_manager,
                 make_test_agent_fn,
                 env_fn_test,
                 tester_seed,
                 config):
        _set_torch_threads_tf32(num_threads=FLAGS.num_test_episodes, allow_tf32=True)

        from library.infrastructure.tester import Tester
        # modules
        self.receive_queue = receive_queue
        self.logging_freq = FLAGS.logging_freq
        self.actors = actors
        self.learner = learner
        self.replay_manager = replay_manager

        # create tester
        self.test_agent = make_test_agent_fn()
        self.tester = Tester(env_fn=env_fn_test, num_parallel_env=10, asynchronous=False, seed=tester_seed)
        self.num_test_episodes = FLAGS.num_test_episodes
        self.total_num_policy_updates = FLAGS.total_num_policy_updates

        from library.infrastructure.logger import EpochLogger, setup_logger_kwargs

        self.logger = EpochLogger(**setup_logger_kwargs(exp_name=FLAGS.exp_name,
                                                        seed=FLAGS.seed,
                                                        data_dir=FLAGS.logger_path))
        self.logger.save_config(config)

        self.tester.set_logger(self.logger)
        self.test_agent.set_logger(self.logger)

        self.logger.register(self.tester.log_tabular)
        self.logger.register(self.test_agent.log_tabular)

    def update_weights(self):
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.test_agent.load_state_dict(weights)

    def log(self):
        while True:
            num_policy_updates, training_throughput = self.receive_queue.get()
            self.update_weights()
            self.tester.test_agent(get_action=self.test_agent.act_batch_test,
                                   name=self.test_agent.__class__.__name__,
                                   num_test_episodes=self.num_test_episodes,
                                   max_episode_length=None,
                                   timeout=None,
                                   verbose=False)

            # actor stats. EpRet, EpLen
            for actor in self.actors:
                stats = ray.get(actor.get_stats.remote())
                self.logger.store(**stats)

            # learner states
            learner_stats = ray.get(self.learner.get_stats.remote())
            self.logger.store(**learner_stats)

            # replay_stats
            replay_stats = ray.get(self.replay_manager.get_stats.remote())
            total_env_interactions = replay_stats['TotalEnvInteracts']
            sampling_throughput = replay_stats['Samples/s']
            estimated_finish_time = (self.total_num_policy_updates - num_policy_updates) / training_throughput

            self.logger.log_tabular('Epoch', num_policy_updates // self.logging_freq)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('TotalEnvInteracts', total_env_interactions)
            self.logger.log_tabular('Samples/s', sampling_throughput)
            self.logger.log_tabular('PolicyUpdates', num_policy_updates)
            self.logger.log_tabular('GradientSteps/s', training_throughput)
            self.logger.log_tabular('Remaining Time (h)', estimated_finish_time / 3600)
            self.logger.dump_tabular()

            if num_policy_updates >= self.total_num_policy_updates:
                break


flags.DEFINE_string('env_name', default=None, required=True, help='Environment name')
flags.DEFINE_integer('num_actors', default=16, help='Number of actors')
flags.DEFINE_float('sac_target_entropy_per_dim', default=0.2, help='Target entropy per dim of SAC')

from library.agents.sac import SACAgent


def run_apex(argv):
    config = {flag.name: flag.value for flag in FLAGS.get_key_flags_for_module('__main__')}
    print(f'Total number of cpus {os.cpu_count()}')

    from library.infrastructure.seeder import Seeder
    seeder = Seeder(seed=FLAGS.seed)

    ReplayManager_remote = ray.remote(num_cpus=FLAGS.num_cpus_replay)(ReplayManager)
    Learner_remote = ray.remote(num_cpus=FLAGS.num_cpus_learner, num_gpus=1 if FLAGS.use_gpu_learner else 0)(Learner)
    Actor_remote = ray.remote(num_cpus=FLAGS.num_cpus_actor, num_gpus=0)(Actor)

    ray.init()

    from library.gym_utils import TransformObservationDtype

    env_name = FLAGS.env_name

    if FLAGS.exp_name is None:
        FLAGS.set_default('exp_name', f'apex_continuous_{env_name}')

    def env_fn_train():
        env = gym.make(env_name)
        env = TransformObservationDtype(env, dtype=np.float32)
        env = gym.wrappers.RescaleAction(env, min_action=-1., max_action=1.)
        return env

    env_fn_test = env_fn_train

    dummy_env = env_fn_train()

    # create replay manager
    from library.replay_buffer.replay_buffer import get_data_spec_from_env
    replay_manager = ReplayManager_remote.remote(data_spec=get_data_spec_from_env(dummy_env),
                                                 seed=seeder.generate_seed(),
                                                 capacity=FLAGS.replay_capacity,
                                                 alpha=FLAGS.replay_alpha,
                                                 beta=FLAGS.replay_beta,
                                                 eviction=FLAGS.replay_eviction,
                                                 update_after=FLAGS.update_after
                                                 )

    # make queues
    testing_queue = queue.Queue(maxsize=3)

    target_entropy_per_dim = FLAGS.sac_target_entropy_per_dim
    make_agent_fn = lambda: SACAgent(dummy_env, target_entropy_per_dim=target_entropy_per_dim)

    # create learners
    learner = Learner_remote.remote(make_agent_fn=make_agent_fn,
                                    replay_manager=replay_manager,
                                    testing_queue=testing_queue,
                                    num_cpus_learner=FLAGS.num_cpus_learner,
                                    prefetch=FLAGS.prefetch,
                                    batch_size=FLAGS.batch_size,
                                    weight_push_freq=FLAGS.weight_push_freq,
                                    logging_freq=FLAGS.logging_freq
                                    )

    # create actors
    actors = [Actor_remote.remote(
        make_agent_fn=make_agent_fn,
        env_train_fn=env_fn_train,
        learner=learner,
        replay_manager=replay_manager,
        seed=seeder.generate_seed(),
        num_cpus_actor=FLAGS.num_cpus_actor,
        weight_update_freq=FLAGS.weight_update_freq,
        local_capacity=FLAGS.local_capacity,
        n_steps=FLAGS.n_steps,
        gamma=FLAGS.gamma
    ) for _ in range(FLAGS.num_actors)]

    logger = Logger(
        receive_queue=testing_queue,
        actors=actors,
        learner=learner,
        make_test_agent_fn=make_agent_fn,
        env_fn_test=env_fn_test,
        tester_seed=seeder.generate_seed(),
        config=config,
        replay_manager=replay_manager
    )

    # start to run actors
    for actor in actors:
        actor.run.remote()

    # wait for replay buffer warmup
    while True:
        ready, replay_size = ray.get(replay_manager.ready.remote())
        print(f'Replay buffer size: {replay_size}')

        if ready:
            # start learner prefetches
            learner.run.remote()
            break
        else:
            time.sleep(0.1)

    # start global training
    logx.log('Start training', color='green')
    logger.log()

    ray.shutdown()


if __name__ == '__main__':
    app.run(run_apex)
