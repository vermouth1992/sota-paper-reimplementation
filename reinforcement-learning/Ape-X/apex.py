"""
Implement parallel SPO using ray.
The logging is done by the number of policy updates
"""

import threading
from typing import Dict
import numpy as np
import collections
import ray
import time

import library
from ray.util import queue
from typing import List

import library.pytorch_utils as ptu

from rlutils import logx

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_cpus_replay', 1, 'Number of cpus used for ReplayManager')
flags.DEFINE_integer('update_after', 10000, 'Number of environment steps before update begins')


@ray.remote(num_cpus=FLAGS.num_cpus_replay)
class ReplayManager(object):
    def __init__(self, make_replay_fn,
                 update_after=FLAGS.update_after):
        self.replay_buffer = make_replay_fn()
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


class Dataset(object):
    def __init__(self, replay_buffer_actor):
        self.replay_buffer_actor = replay_buffer_actor

    def __call__(self, batch_size):
        while True:
            yield ray.get(self.replay_buffer_actor.sample.remote(batch_size=batch_size))


flags.DEFINE_integer('num_cpus_learner', default=1, help='Number of CPUs per learner')
flags.DEFINE_boolean('use_gpu_learner', default=True, help='Whether use GPU for the learner')
flags.DEFINE_integer('weight_push_freq', default=10, help='Frequency of learner pushes its weights')
flags.DEFINE_integer('batch_size', default=256, help='Batch size for training')
flags.DEFINE_integer('prefetch', default=5, help='Batch prefetch from ReplayManager')


@ray.remote(num_cpus=FLAGS.num_cpus_learner, num_gpus=1 if FLAGS.use_gpu_learner else 0)
class Learner(object):
    def __init__(self,
                 make_agent_fn,
                 replay_manager,
                 actor_critic,
                 weight_push_freq=FLAGS.weight_push_freq,
                 batch_size=FLAGS.batch_size,
                 prefetch=FLAGS.prefetch,
                 num_threads=FLAGS.num_cpus_learner,
                 testing_queue: queue.Queue = None,
                 logging_freq=None,
                 ):
        import torch
        import tensorflow as tf
        import tensorflow_datasets as tfds

        torch.set_num_threads(num_threads)

        self.agent = make_agent_fn()
        self.actor_critic = actor_critic
        self.replay_manager = replay_manager
        self.dataset = tf.data.Dataset.from_generator(Dataset(self.replay_manager),
                                                      output_signature=dict(

                                                      )
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
            self.replay_manager.update_priorities.remote(transaction_id,
                                                         info['TDError'].cpu().numpy())
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





class Actor(object):
    def __init__(self,
                 make_agent_fn,
                 make_sampler_fn,
                 make_local_buffer_fn,
                 weight_server_lst,
                 replay_manager_lst,
                 weight_update_freq=100,
                 num_threads=1,
                 ):
        import torch
        torch.set_num_threads(num_threads)

        self.agent = make_agent_fn()
        self.agent.eval()
        self.weight_server_lst = weight_server_lst
        self.replay_manager_lst = replay_manager_lst
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
        import torch
        torch.set_num_threads(num_cpus_tester)
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


def run_spo(make_local_learner_fn_lst,  # used for each learner
            make_test_agent_fn,
            make_sampler_fn,
            make_local_buffer_fn,
            make_actor_fn_lst,
            make_replay_fn,
            make_tester_fn,
            exp_name,
            config=None,
            # actor args
            num_cpus_per_actor=1,
            num_actors=4,
            weight_update_freq=20,
            # learner args
            num_learners=1,
            num_cpus_per_learner=1,
            num_gpus_per_learner=1,
            batch_size=256,
            weight_push_freq=10,
            update_after=10000,
            sync_freq=10,
            target_update_freq=100,
            actor_critic=False,
            averaging_device=None,
            # logging
            total_num_policy_updates=1000000,
            logging_freq=10000,
            num_test_episodes=30,
            num_cpus_tester=4,
            num_gpus_tester=0,
            logger_path: str = None,
            seed=1,
            ):
    # argument checking
    assert batch_size % num_learners == 0
    assert num_actors % num_learners == 0 or num_learners % num_actors == 0

    if config is None:
        config = locals()

    import os
    import torch
    torch.set_num_threads(num_cpus_per_learner)

    total_cpus = os.cpu_count()

    print(f'Total number of CPUs: {total_cpus}, Actor usage: {num_actors * num_cpus_per_actor}, '
          f'Learner usage: {num_cpus_per_learner * num_learners}, Tester usage: {num_cpus_tester}')

    Actor_remote = ray.remote(num_cpus=num_cpus_per_actor)(Actor)
    Learner_remote = ray.remote(num_cpus=num_cpus_per_learner, num_gpus=num_gpus_per_learner)(Learner)
    ReplayManager_remote = ray.remote(num_cpus=1)(ReplayManager)

    ray.init()

    # create replay manager
    num_replay_managers = num_learners
    replay_manager_lst = [ReplayManager_remote.remote(make_replay_fn=make_replay_fn,
                                                      update_after=update_after // num_replay_managers)
                          for _ in range(num_replay_managers)]

    # make queues
    learner_push_queue = queue.Queue() if num_learners > 1 else None
    learner_receive_queues = [queue.Queue() for _ in range(num_learners - 1)]
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
