import sys
import logging

import gym
import h5py

from es_distributed import tf_util
from es_distributed.policies import policies
from es_distributed.atari_wrappers import wrap_deepmind
from .common import RunningStat, SharedNoiseTable

log = logging.getLogger(__name__)

def build_env(env_id):
    gym.undo_logger_setup()
    env = gym.make(env_id)

    if env_id.endswith('NoFrameskip-v4'):
        env = wrap_deepmind(env)
    return env

def build_policy(policy_type, *args, **kwargs):
    return getattr(policies, policy_type)(*args, **kwargs)

def load_policy(policy_type, file):
    return getattr(policies, policy_type).Load(file)

def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(
        config=tf.ConfigProto(
            inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
        )
    )


class Algo(object):

    @property
    def name(self):
        raise NotImplemented()

    def save(self, file, *args, **kwargs):
        assert self.policy is not None, "policy not initialised"
        file.attrs['algo.type'] = self.name
        self.policy.save(file)

    def load(self, config, file):
        self.policy = load_policy(config.policy.type, file)

    def setup_job(self, config, noise=None, snapshot=None):
        log.debug("Building env")
        self.env = build_env(config.task.env_id)
        self.session = make_session(single_threaded=False)
        self.noise = noise if noise else SharedNoiseTable()

        if snapshot:
            log.debug("Loading policy from snapshot")
            with h5py.File(snapshot, 'r') as f:
                self.load(config, f)
        else:
            log.debug("Building new policy")
            self.policy = build_policy(
                config.policy.type,
                self.env.observation_space, self.env.action_space, **config.policy.args)
            tf_util.initialize()

        if self.policy.needs_ob_stat:
            self.ob_stat = RunningStat(
                self.env.observation_space.shape,
                eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
            )

    def setup_iteration(self, master_client, timestep_limit):
        raise NotImplemented()

    def process_result(self, result):
        raise NotImplemented()

    def process_iteration(self, config, iteration_results):
        raise NotImplemented()