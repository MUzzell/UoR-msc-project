from collections import namedtuple
import time
import logging

import numpy as np
import gym

from es_distributed import tf_util
from es_distributed.policies import policies
from es_distributed.config import Result
from es_distributed.optimizers import SGD, Adam

from .common import SharedNoiseTable, RunningStat
from . import algo

log = logging.getLogger(__name__)

ESTask = namedtuple('ESTask', [
    'params', 'ob_mean', 'ob_std', 'ref_batch', 'timestep_limit'])

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

def get_ref_batch(env, batch_size=32):
    ref_batch = []
    ob = env.reset()
    while len(ref_batch) < batch_size:
        ob, rew, done, info = env.step(env.action_space.sample())
        ref_batch.append(ob)
        if done:
            ob = env.reset()
    return ref_batch

def rollout_and_update_ob_stat(policy, env, timestep_limit, rs, task_ob_stat, calc_obstat_prob):
    if policy.needs_ob_stat and calc_obstat_prob != 0 and rs.rand() < calc_obstat_prob:
        rollout_rews, rollout_len, obs, rollout_nov = policy.rollout(
            env, timestep_limit=timestep_limit, save_obs=True, random_stream=rs)
        task_ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
    else:
        rollout_rews, rollout_len, rollout_nov = policy.rollout(
            env, timestep_limit=timestep_limit, random_stream=rs)

    return rollout_rews, rollout_len, rollout_nov

class ES(algo.Algo):

    @property
    def name(self):
        return "es"

    def __init__(self):
        self.rs = np.random.RandomState()
        self.ob_count = 0

    def setup_job(self, config, noise=None, snapshot=None):
        super(ES, self).setup_job(config, noise, snapshot)

        self.theta = self.policy.get_trainable_flat()

        self.optimizer = {
            'sgd': SGD,
            'adam': Adam
        }[config.algo['optimizer']['type']](
            self.theta, **config.algo['optimizer']['args']
        )

        if self.policy.needs_ob_stat:
            self.ob_stat = RunningStat(
                self.env.observation_space.shape,
                eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
            )

        if self.policy.needs_ref_batch:
            self.ref_batch = get_ref_batch(self.env, batch_size=128)
            self.policy.set_ref_batch(self.ref_batch)

    def setup_iteration(self, master_client, timestep_limit):

        self.theta = self.policy.get_trainable_flat()
        assert self.theta.dtype == np.float32

        log.debug("declaring task")
        return master_client.declare_task(ESTask(
            params=self.theta,
            ob_mean=self.ob_stat.mean if self.policy.needs_ob_stat else None,
            ob_std=self.ob_stat.std if self.policy.needs_ob_stat else None,
            ref_batch=self.ref_batch if self.policy.needs_ref_batch else None,
            timestep_limit=timestep_limit
        ))

    def process_result(self, result):
        if self.policy.needs_ob_stat and result.ob_count > 0:
            self.ob_stat.increment(
                result.ob_sum, result.ob_sumsq, result.ob_count
            )
        self.ob_count += result.ob_count

    def process_iteration(self, config, iteration_results):
        noise_inds_n = np.concatenate([r.noise_inds_n for r in iteration_results.results])
        returns_n2 = np.concatenate([r.returns_n2 for r in iteration_results.results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in iteration_results.results])
        action_rews = [r.action_mean for r in iteration_results.results]
        signreturns_n2 = np.concatenate([r.signreturns_n2 for r in iteration_results.results])

        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
        # Process returns
        if config.algo['return_proc_mode'] == 'centered_rank':
            proc_returns_n2 = compute_centered_ranks(returns_n2)
        elif config.algo['return_proc_mode'] == 'sign':
            proc_returns_n2 = signreturns_n2
        elif config.algo['return_proc_mode'] == 'centered_sign_rank':
            proc_returns_n2 = compute_centered_ranks(signreturns_n2)
        else:
            raise NotImplementedError(config.return_proc_mode)

        # Compute and take step
        g, count = batched_weighted_sum(
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            (self.noise.get(idx, self.policy.num_params) for idx in noise_inds_n),
            batch_size=500
        )
        g /= returns_n2.size
        assert g.shape == (self.policy.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)

        update_ratio, self.theta = self.optimizer.update(-g + config.algo['l2coeff'] * self.theta)

        #updating policy
        self.policy.set_trainable_flat(self.theta)

        # Update ob stat (we're never running the policy in the master, but we might be snapshotting the policy)
        if self.policy.needs_ob_stat:
            self.policy.set_ob_stat(self.ob_stat.mean, self.ob_stat.std)

        return returns_n2, lengths_n2, action_rews

    def run_episode(self, config,task_id, task_data):
        if self.policy.needs_ob_stat:
            self.policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        if self.policy.needs_ref_batch:
            self.policy.set_ref_batch(task_data.ref_batch)

        if self.rs.rand() < config.algo['eval_prob']:
            self.policy.set_trainable_flat(task_data.params)
            eval_rews, eval_length, _ = self.policy.rollout(
                self.env, timestep_limit=task_data.timestep_limit)
            eval_return = eval_rews.sum()
            log.debug("Eval job, Reward: %f, TS: %f, Action Reward: %f",
                      eval_return, eval_length, eval_return / eval_length)

            return Result(
                worker_id=1,
                noise_inds_n=None,
                returns_n2=None,
                signreturns_n2=None,
                lengths_n2=None,
                action_mean=None,
                eval_return=eval_return,
                eval_length=eval_length,
                ob_sum=None,
                ob_sumsq=None,
                ob_count=None
            )

        task_tstart = time.time()

        noise_inds, returns, signreturns, lengths = [], [], [], []
        task_ob_stat = RunningStat(self.env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

        while not noise_inds or time.time() - task_tstart < .2:
            noise_idx = self.noise.sample_index(self.rs, self.policy.num_params)
            v = config.algo['noise_stdev'] * self.noise.get(noise_idx, self.policy.num_params)

            self.policy.set_trainable_flat(task_data.params + v)
            rews_pos, len_pos, nov_vec_pos = rollout_and_update_ob_stat(
                self.policy, self.env, task_data.timestep_limit, self.rs,
                task_ob_stat, config.algo['calc_obstat_prob'])

            self.policy.set_trainable_flat(task_data.params - v)
            rews_neg, len_neg, nov_vec_neg = rollout_and_update_ob_stat(
                self.policy, self.env, task_data.timestep_limit, self.rs,
                task_ob_stat, config.algo['calc_obstat_prob'])

            signreturns.append([np.sign(rews_pos).sum(), np.sign(rews_neg).sum()])
            noise_inds.append(noise_idx)
            returns.append([rews_pos.sum(), rews_neg.sum()])
            lengths.append([len_pos, len_neg])

        return Result(
            worker_id=1,
            noise_inds_n=np.array(noise_inds),
            returns_n2=np.array(returns, dtype=np.float32),
            signreturns_n2=np.array(signreturns, dtype=np.float32),
            lengths_n2=np.array(lengths, dtype=np.int32),
            action_mean=rews_pos.mean(),
            eval_return=None,
            eval_length=None,
            ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
            ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
            ob_count=task_ob_stat.count
        )