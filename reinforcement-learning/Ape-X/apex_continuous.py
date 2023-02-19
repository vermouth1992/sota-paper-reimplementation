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


@ray.remote(num_cpus=FLAGS.num_cpus_replay)
class ReplayManager(object):
    def __init__(self,
                 data_spec,
                 capacity=FLAGS.replay_capacity,
                 alpha=FLAGS.replay_alpha,
                 beta=FLAGS.replay_beta,
                 eviction=FLAGS.replay_eviction,
                 update_after=FLAGS.update_after):
        from library.replay_buffer import PrioritizedReplayBuffer
        self.replay_buffer = PrioritizedReplayBuffer(data_spec=data_spec,
                                                     capacity=capacity,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     eviction=eviction,
                                                     seed=None)
        self.update_after = update_after

    def sample(self, batch_size):
        data = self.replay_buffer.sample(batch_size)
        return data

    def update_priorities(self, transaction_id, priorities):
        self.replay_buffer.update_priorities(transaction_id, priorities)

    def ready(self):
        return len(self.replay_buffer) >= self.update_after, len(self.replay_buffer)

    def add(self, data: Dict[str, np.ndarray], priority=None):
        self.replay_buffer.add(data, priority)

    def data_spec(self):
        return self.replay_buffer.data_spec


class Dataset(object):
    def __init__(self, replay_buffer_actor):
        self.replay_buffer_actor = replay_buffer_actor

    def __call__(self, batch_size):
        while True:
            yield ray.get(self.replay_buffer_actor.sample.remote(batch_size=batch_size))


import tensorflow as tf


def _convert_gym_space_to_tf_tensorspec(space: gym.spaces.Space):
    return tf.TensorSpec(shape=space.shape, dtype=space.dtype)


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
flags.DEFINE_integer('logging_freq', default=1000, help='Number of ')


@ray.remote(num_cpus=FLAGS.num_cpus_learner, num_gpus=1 if FLAGS.use_gpu_learner else 0)
class Learner(object):
    def __init__(self,
                 make_agent_fn,
                 replay_manager,
                 testing_queue: queue.Queue,
                 weight_push_freq=FLAGS.weight_push_freq,
                 batch_size=FLAGS.batch_size,
                 prefetch=FLAGS.prefetch,
                 num_threads=FLAGS.num_cpus_learner,
                 logging_freq=FLAGS.logging_freq,
                 ):
        import tensorflow as tf
        import tensorflow_datasets as tfds

        _set_torch_threads_tf32(num_threads, True)

        self.agent = make_agent_fn()
        self.replay_manager = replay_manager
        data_spec = self.replay_manager.data_spec.remote()
        output_signature = {key: _convert_gym_space_to_tf_tensorspec(space) for key, space in data_spec.items()}
        print(output_signature)
        self.dataset = tf.data.Dataset.from_generator(Dataset(self.replay_manager),
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


flags.DEFINE_integer('num_cpus_actor', default=1, help='Number of CPUs per actor')


@ray.remote(num_cpus=FLAGS.num_cpus_actor, num_gpus=0)
class Actor(object):
    def __init__(self,
                 make_agent_fn,
                 learner,
                 replay_manager,
                 weight_update_freq=100,
                 num_threads=1,
                 ):
        _set_torch_threads_tf32(num_threads, True)

        self.agent = make_agent_fn()
        self.agent.eval()
        self.learner = learner
        self.replay_manager = replay_manager
        self.weight_update_freq = weight_update_freq
        self.sampler = make_sampler_fn()
        self.local_buffer = make_local_buffer_fn()

        from rlutils.logx import EpochLogger
        self.logger = EpochLogger()
        self.sampler.set_logger(self.logger)

        self.current_data_index = 0
        self.current_weight_index = 0
        self.num_learners = len(self.weight_server_lst)

        assert len(self.weight_server_lst) == len(self.replay_manager_lst)

    def get_stats(self):
        stats = self.logger.get_epoch_dict()
        self.logger.clear_epoch_dict()
        return stats

    def run(self):
        background_thread = threading.Thread(target=self.train, daemon=True)
        background_thread.start()

    def add_local_to_global(self, data, priority):
        # pick a random replay manager
        replay_manager = self.replay_manager_lst[self.current_data_index]
        replay_manager.add.remote(data, priority)

        self.current_data_index = (self.current_data_index + 1) % self.num_learners

    def get_weights(self):
        weight_server = self.weight_server_lst[self.current_weight_index]
        weights_id = ray.get(weight_server.get_weights.remote())
        weights = ray.get(weights_id)
        self.current_weight_index = (self.current_weight_index + 1) % self.num_learners
        return weights

    def train(self):
        self.update_weights()
        self.sampler.reset()
        local_steps = 0
        while True:
            self.sampler.sample(num_steps=1, collect_fn=lambda o: self.agent.act_batch_explore(o, None),
                                replay_buffer=self.local_buffer)

            local_steps += 1
            if self.local_buffer.is_full():
                data = self.local_buffer.storage.get()
                priority = self.agent.compute_priority(data)
                self.add_local_to_global(data, priority)
                self.local_buffer.reset()

            if local_steps % self.weight_update_freq == 0:
                self.update_weights()

    def update_weights(self):
        weights = self.get_weights()
        self.agent.load_state_dict(weights)


class Logger(object):
    def __init__(self,
                 receive_queue: queue.Queue,
                 logging_freq,
                 total_num_policy_updates,
                 actors,
                 local_learners,
                 replay_manager_lst,
                 num_cpus_tester,
                 make_test_agent_fn,
                 make_tester_fn,
                 num_test_episodes,
                 exp_name,
                 seed,
                 logger_path,
                 config):
        _set_torch_threads_tf32(num_threads=num_cpus_tester, allow_tf32=True)
        # modules
        self.receive_queue = receive_queue
        self.logging_freq = logging_freq
        self.actors = actors
        self.local_learners = local_learners
        self.replay_manager_lst = replay_manager_lst

        # create tester
        self.test_agent = make_test_agent_fn()
        self.tester = make_tester_fn()
        self.num_test_episodes = num_test_episodes
        self.total_num_policy_updates = total_num_policy_updates

        from rlutils.logx import EpochLogger, setup_logger_kwargs

        self.logger = EpochLogger(**setup_logger_kwargs(exp_name=exp_name, seed=seed, data_dir=logger_path))
        self.logger.save_config(config)

        self.tester.set_logger(self.logger)
        self.test_agent.set_logger(self.logger)

    def log(self):
        while True:
            weights, num_policy_updates, training_throughput = self.receive_queue.get()
            self.test_agent.load_state_dict(weights)
            self.tester.test_agent(get_action=self.test_agent.act_batch_test,
                                   name=self.test_agent.__class__.__name__,
                                   num_test_episodes=self.num_test_episodes,
                                   max_episode_length=None,
                                   timeout=None,
                                   verbose=False)
            # actor stats
            for actor in self.actors:
                stats = ray.get(actor.get_stats.remote())
                self.logger.store(**stats)

            # learner states
            stats_lst = [dict(ray.get(learner.get_stats.remote())) for learner in self.local_learners]

            # replay stats
            replay_stats_lst = [dict(ray.get(replay_manager.get_stats.remote()))
                                for replay_manager in self.replay_manager_lst]

            # analyze stats
            prefetch_rates_lst = []
            for stats in stats_lst:
                prefetch_rates_lst.append(stats.pop('PrefetchRate'))
                self.logger.store(**stats)

            # trick to use the agent log_tabular
            self.test_agent.policy_updates = num_policy_updates
            prefetch_rates = np.mean(prefetch_rates_lst)

            total_env_interactions = 0
            sampling_throughput = []
            for replay_stats in replay_stats_lst:
                total_env_interactions += replay_stats.pop('TotalEnvInteracts')
                sampling_throughput.append(replay_stats.pop('Samples/s'))
            sampling_throughput = np.sum(sampling_throughput)

            self.logger.log_tabular('Epoch', num_policy_updates // self.logging_freq)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('TotalEnvInteracts', total_env_interactions)
            self.logger.log_tabular('GradientSteps/s', training_throughput)
            self.logger.log_tabular('Samples/s', sampling_throughput)
            self.logger.log_tabular('PrefetchRate', prefetch_rates)
            self.logger.dump_tabular()

            if num_policy_updates >= self.total_num_policy_updates:
                break


flags.DEFINE_string('env_name', default='Humanoid-v4', required=True, help='Environment name')


def run_apex(argv):
    if config is None:
        config = locals()

    print(f'Total number of cpus {os.cpu_count()}')

    ray.init()

    # create replay manager

    from library.gym_utils import TransformObservationDtype

    def env_fn_train():
        env = gym.make(FLAGS.env_name)
        env = TransformObservationDtype(env, dtype=np.float32)
        env = gym.wrappers.RescaleAction(env, min_action=-1., max_action=1.)
        return env

    env_fn_test = env_fn_train

    # create replay buffer
    from library.replay_buffer.replay_buffer import get_data_spec_from_env
    replay_manager = ReplayManager.remote(data_spec=get_data_spec_from_env(env_fn_train()))

    # make queues
    testing_queue = queue.Queue(maxsize=3)

    # create learners
    learner_args = dict(
        actor_critic=actor_critic,
        batch_size=batch_size // num_learners,
        num_threads=num_cpus_per_learner,
        weight_push_freq=weight_push_freq,
    )

    local_learners = []
    for i in range(num_learners):
        if i == num_learners - 1:
            # the last one is the main leaner
            local_learners.append(Learner_remote.remote(
                replay_manager=replay_manager_lst[i],
                make_agent_fn=make_local_learner_fn_lst[i],
                receive_queue=None,
                push_queue=None,
                learner_push_queue=learner_push_queue,
                learner_receive_queues=learner_receive_queues,
                testing_queue=testing_queue,
                sync_freq=sync_freq,
                target_update_freq=target_update_freq,
                logging_freq=logging_freq,
                main_learner=True,
                **learner_args
            ))
        else:
            local_learners.append(Learner_remote.remote(
                replay_manager=replay_manager_lst[i],
                make_agent_fn=make_local_learner_fn_lst[i],
                receive_queue=learner_receive_queues[i],
                push_queue=learner_push_queue,
                **learner_args))

    # allocate learners to actors
    weight_server_lst_lst = [[] for _ in range(num_actors)]
    replay_manager_lst_lst = [[] for _ in range(num_actors)]

    if num_actors >= num_learners:
        for i in range(num_actors):
            weight_server_lst_lst[i].append(local_learners[i % num_learners])
            replay_manager_lst_lst[i].append(replay_manager_lst[i % num_learners])

    else:
        for i in range(num_learners):
            weight_server_lst_lst[i % num_actors].append(local_learners[i])
            replay_manager_lst_lst[i % num_actors].append(replay_manager_lst[i])

    # create actors
    actors = [Actor_remote.remote(
        make_agent_fn=make_actor_fn_lst[i],
        make_sampler_fn=make_sampler_fn,
        make_local_buffer_fn=make_local_buffer_fn,
        weight_server_lst=weight_server_lst_lst[i],
        replay_manager_lst=replay_manager_lst_lst[i],
        weight_update_freq=weight_update_freq,
        num_threads=num_cpus_per_actor) for i in range(num_actors)]

    logger = Logger(
        receive_queue=testing_queue,
        logging_freq=logging_freq,
        total_num_policy_updates=total_num_policy_updates,
        actors=actors,
        local_learners=local_learners,
        replay_manager_lst=replay_manager_lst,
        num_cpus_tester=num_cpus_tester,
        make_test_agent_fn=make_test_agent_fn,
        make_tester_fn=make_tester_fn,
        num_test_episodes=num_test_episodes,
        exp_name=exp_name,
        seed=seed,
        logger_path=logger_path,
        config=config
    )

    # start to run actors
    for actor in actors:
        actor.run.remote()

    # wait for replay buffer warmup
    ready_lst = np.array([False for _ in range(num_replay_managers)], dtype=bool)

    while True:
        for i in range(num_replay_managers):
            if not ready_lst[i]:
                ready_lst[i], replay_size = ray.get(replay_manager_lst[i].ready.remote())
                print(f'Replay buffer {i} size: {replay_size}')

        ready = np.all(ready_lst)
        if ready:
            # start learner prefetches
            for learner in local_learners:
                learner.run.remote()
            break
        else:
            time.sleep(1)

    # start global training
    logx.log('Start training', color='green')
    logger.log()

    ray.shutdown()


if __name__ == '__main__':
    app.run(run_apex)
