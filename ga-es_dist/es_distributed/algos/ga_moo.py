from collections import namedtuple
import time

import logging

import numpy as np
import pickle

from es_distributed import tf_util
from es_distributed.policies import policies
from es_distributed.config import Result

from .common import SharedNoiseTable, RunningStat
from . import algo

log = logging.getLogger(__name__)

GAMooTask = namedtuple('GAMooTask', [
    'params', 'population', 'ob_mean', 'ob_std', 'timestep_limit',
    'num_eps'
])


def rollout_and_update_ob_stat(policy, env, num_eps, timestep_limit, rs, task_ob_stat, calc_obstat_prob):

    save_obs = policy.needs_ob_stat and calc_obstat_prob != 0 and rs.rand() < calc_obstat_prob

    rewards = []
    lens = []

    for _ in range(num_eps):
        if save_obs:
            rollout_rews, rollout_len, obs = policy.rollout(
                env, timestep_limit=timestep_limit,
                save_obs=True, random_stream=rs)
            task_ob_stat.increment(obs.sum(axis=0),
                                   np.square(obs).sum(axis=0), len(obs))
        else:
            rollout_rews, rollout_len = policy.rollout(
                env, timestep_limit=timestep_limit,
                save_obs=False, random_stream=rs)

        rewards += [rollout_rews.sum()]
        lens += [rollout_len]

    return rewards, lens


class GAMoo(algo.Algo):

    @property
    def name(self):
        return "ga_moo"

    def __init__(self):
        self.population = []
        self.max_avg = 0
        self.max_std = 0
        self.ob_count = 0
        self.population_score = np.array([])
        self.rs = np.random.RandomState()

    def save(self, file, *args, **kwargs):
        super(GAMoo, self).save(file)
        file.attrs['algo.population'] = np.void(pickle.dumps(self.population, protocol=-1))
        file.attrs['algo.max_avg'] = self.max_avg
        file.attrs['algo.max_std'] = self.max_std

    def load(self, config, file):
        super(GAMoo, self).load(config, file)
        # don't need to get theta as it s collected from the policy each time
        self.population = pickle.loads(file.attrs['algo.population']) if 'algo.population' in file.attrs else []
        self.max_avg = pickle.loads(file.attrs['algo.max_avg']) if 'algo.max_avg' in file.attrs else 0
        self.max_std = pickle.loads(file.attrs['algo.max_avg']) if 'algo.max_avg' in file.attrs else 0

    def setup_job(self, config, noise=None, snapshot=None):
        super(GAMoo, self).setup_job(config, noise, snapshot)
        self.population_size = config.algo['population_size']
        self.num_elites = config.algo['num_elites']
        self.num_eps = config.task.episodes_target

    def setup_iteration(self, master_client, timestep_limit):

        self.theta = self.policy.get_trainable_flat()
        assert self.theta.dtype == np.float32

        log.debug("declaring task")
        return master_client.declare_task(GAMooTask(
            params=self.theta,
            population=self.population,
            ob_mean=self.ob_stat.mean if self.policy.needs_ob_stat else None,
            ob_std=self.ob_stat.std if self.policy.needs_ob_stat else None,
            timestep_limit=timestep_limit,
            num_eps=self.num_eps
        ))

    def process_result(self, result):
        if self.policy.needs_ob_stat and result.ob_count > 0:
            self.ob_stat.increment(
                result.ob_sum, result.ob_sumsq, result.ob_count
            )
        self.ob_count += result.ob_count

    def fitness(self, avg, std):
        avg = avg / self.max_avg if self.max_avg > 0 else 0
        std = std / self.max_std if self.max_std > 0 else 0

        return (1 - avg) + std

    def process_iteration(self, config, iteration_results):
        #import pdb; pdb.set_trace()
        noise_inds_n = list(self.population[:self.num_elites])
        returns_n2 = list(self.population_score[:self.num_elites])
        action_rews = []
        fitness_n2 = []

        self.max_avg = np.max(
            [self.max_avg] +
            [np.mean(r.returns_n2) for r in iteration_results.results])
        self.max_std = np.max(
            [self.max_std] +
            [np.std(r.returns_n2) for r in iteration_results.results])

        if iteration_results.eval_returns:
            elite_avg = np.mean(iteration_results.eval_returns)
            elite_std = np.std(iteration_results.eval_returns)

            fitness_n2.append(self.fitness(elite_avg, elite_std))

        for r in iteration_results.results:

            avg = np.mean(r.returns_n2)
            std = np.std(r.returns_n2)
            fitness_n2.append(self.fitness(avg, std))

            noise_inds_n.extend(r.noise_inds_n)
            returns_n2.extend(r.returns_n2)
            action_rews.append(r.action_mean)

        noise_inds_n = np.array(noise_inds_n)
        returns_n2 = np.array(returns_n2)
        lengths_n2 = np.array([r.lengths_n2 for r in iteration_results.results])
        fitness_n2 = np.array(fitness_n2)

        idx = [x[0] for x in sorted(
               enumerate(fitness_n2),
               key=lambda y: y[1])][:self.population_size]

        '''
        idx = np.argpartition(
            fitness_n2,
            (-self.population_size, -1)
        )[-1:-self.population_size-1:-1][::-1]
        '''

        self.population = noise_inds_n[idx]
        self.population_score = fitness_n2[idx]

        assert len(self.population) == self.population_size, "%d != %d" % (len(self.population), self.population_size)
        assert np.min(fitness_n2) == self.population_score[0], "%d != %d" % (np.min(fitness_n2), self.population_score[0])

        self.policy.set_trainable_flat(
            self.noise.get(self.population[0][0], self.policy.num_params)
        )
        self.policy.reinitialize()
        v = self.policy.get_trainable_flat()

        for seed in self.population[0][1:]:
            v += config.algo['noise_stdev'] * self.noise.get(seed, self.policy.num_params)
        self.policy.set_trainable_flat(v)

        log.info("Max Avg: %.3f, Max Std: %.3f", self.max_avg, self.max_std)

        return returns_n2, lengths_n2, action_rews

    def run_episode(self, config, task_id, task_data):
        if self.policy.needs_ob_stat:
            self.policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        if self.rs.rand() < config.algo['eval_prob']:
            self.policy.set_trainable_flat(task_data.params)
            eval_rews, eval_length = self.policy.rollout(self.env)
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
            if len(task_data.population) > 0:
                seeds = list(
                    task_data.population[self.rs.randint(len(task_data.population))]) + [self.noise.sample_index(self.rs, self.policy.num_params)]
            else:
                seeds = [self.noise.sample_index(self.rs, self.policy.num_params)]

            v = self.noise.get(seeds[0], self.policy.num_params)

            self.policy.set_trainable_flat(v)
            self.policy.reinitialize()
            v = self.policy.get_trainable_flat()

            for seed in seeds[1:]:
                v += config.algo['noise_stdev'] * self.noise.get(seed, self.policy.num_params)
            self.policy.set_trainable_flat(v)

            rews_pos, len_pos, = rollout_and_update_ob_stat(
                self.policy, self.env, task_data.num_eps,
                task_data.timestep_limit, self.rs, task_ob_stat,
                config.algo['calc_obstat_prob'])

            noise_inds.append(seeds)
            returns.extend(rews_pos)
            signreturns.extend(np.sign(rews_pos))
            lengths.extend(len_pos)

        #import pdb; pdb.set_trace()
        log.debug("Result: {} (avg), timesteps: {}".format(np.mean(returns), sum(lengths)))

        return Result(
            worker_id=1,
            noise_inds_n=noise_inds,
            returns_n2=np.array(returns, dtype=np.float32),
            signreturns_n2=np.array(signreturns, dtype=np.float32),
            lengths_n2=np.array(lengths, dtype=np.int32),
            action_mean=np.mean(returns),
            eval_return=None,
            eval_length=None,
            ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
            ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
            ob_count=task_ob_stat.count
        )
