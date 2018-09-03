import time
import argparse
import os
import numpy as np

import logging

import h5py

from . import log as log_config
from . import dist
from . import config
from .util import get_algo_from_exp

from collections import namedtuple

log = logging.getLogger(__name__)
data_log = logging.getLogger('data')

IterationResults = namedtuple('IterationResults', [
    'results', 'eval_returns', 'eval_lengths', 'results_skipped',
    'episodes_popped', 'timesteps_popped'
])


def _save_snapshot(conf, data_dir, algo, task_id, iteration_results):
    '''
    '''
    '''
    if len(snapshots) == snapshot_count:
        old_file = snapshots.pop()
        if os.path.exists(old_file):
            os.remove(old_file)

    assert not os.path.exists(filename)
    snapshots.insert(0, filename)
    '''

    snapshot_file = config.get_next_snapshot(data_dir, conf.task.snapshot_count, task_id)

    with h5py.File(snapshot_file, 'w', libver='latest') as f:
        algo.save(f, iteration_results)
        #f.attrs['IterMax'] = np.max(iteration_results.results)
        #f.attrs['IterAvg'] = np.mean(iteration_results.results)
        #f.attrs['IterStd'] = np.std(iteration_results.results)
        #f.attrs['EvalMax'] = np.max(iteration_results.eval_returns)
        #f.attrs['EvalAvg'] = np.mean(iteration_results.eval_returns)
        #f.attrs['EvalStd'] = np.std(iteration_results.eval_returns)
        f.attrs['Episodes'] = iteration_results.episodes_popped
        f.attrs['Timesteps'] = iteration_results.timesteps_popped
    log.info('Saved snapshot %s', snapshot_file)


def _log_results(task_id, time_fin, total_time, iter_time, eps, ts, ep_count, ts_count,
                 rew_max, rew_mean, rew_std, action_reward, algo, action):
    log.info(
        """
Iteration %s results (%s)
    Finished:           %s
    Duration:           %s
    Episodes:           %d
    Timesteps:          %d

    Time so far:        %s
    Episodes so far:    %d
    Timesteps so far:   %d

    Reward max          %f
    Reward mean         %f
    Reward std          %f
    Avg action reward   %f

    Action              %s
        """, task_id, algo, time.asctime(time.localtime(time_fin)), time_fin - total_time, ep_count,
        ts_count, total_time, eps, ts, rew_max, rew_mean, rew_std, action_reward, action)

    data_log.info(",".join(
        [str(x) for x in (task_id, time_fin, total_time, iter_time, eps, ts, ep_count, ts_count,
        rew_max, rew_mean, rew_std, action_reward, algo, action)]))


def _adjust_timestep_limit(tslimit, incr_tslimit_threshold, tslimit_incr_ratio):
    old_tslimit = tslimit
    tslimit = int(tslimit_incr_ratio * tslimit)
    log.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))
    return tslimit


def _loop_job(master_client, algo, conf, data_dir):
    '''
    '''

    task_id, episodes_so_far, timesteps_so_far, ts_limit, ts_th, ts_inc_ratio, tstart, best_score = config.load_session(conf, data_dir)

    log.debug("Declaring experient to redis")
    master_client.declare_experiment(conf, task_id)

    snapshot_file = config.get_latest_snapshot(data_dir)

    algo.setup_job(conf, snapshot=snapshot_file)

    log.debug("Starting Experiment")

    while (timesteps_so_far < conf.task.timesteps_target):
        step_tstart = time.time()
        task_id = algo.setup_iteration(master_client, ts_limit)

        iteration_results = _loop_iteration(master_client, task_id, algo, conf)
        episodes_so_far += iteration_results.episodes_popped
        timesteps_so_far += iteration_results.timesteps_popped

        rewards, timesteps, action_rewards = algo.process_iteration(
            conf, iteration_results)

        step_tend = time.time()

        #       iter_id, time,total_time,iter_time,
        #       ep_count,ts_count,
        #       ts_ep,res_ep,
        #       rew_max,rew_mean,rew_std
        #       algo,action
        _log_results(task_id, step_tend, step_tend - tstart, step_tend - step_tstart,
                     episodes_so_far, timesteps_so_far,
                     iteration_results.episodes_popped, iteration_results.timesteps_popped,
                     rewards.max(), rewards.mean(), rewards.std(),
                     sum(action_rewards) / len(action_rewards), algo.name,"c")

        config.save_session(data_dir,
                            task_id=task_id, episodes_so_far=int(episodes_so_far),
                            timesteps_so_far=int(timesteps_so_far),
                            ts_limit=ts_limit, tstart=tstart,
                            best_score=best_score, algo=algo.name)

        _save_snapshot(conf, data_dir, algo, task_id, iteration_results)

        if ts_th and (timesteps == ts_limit).mean() >= ts_th:
            ts_limit = _adjust_timestep_limit(ts_limit, ts_th, ts_inc_ratio)

    log.info("Shutting down job")


def _loop_iteration(master_client, task_id, algo, conf):

    results = []
    eval_rets = []
    eval_lens = []

    num_eval_episodes = 0
    num_eval_timesteps = 0

    num_episodes_popped = 0
    num_results_skipped = 0
    num_timesteps_popped = 0

    while (num_episodes_popped < conf.task.episodes_per_batch):

        result_task_id, result = master_client.pop_result()

        if result_task_id != task_id:
            num_results_skipped += 1
            continue

        if result.eval_length is not None:
            num_eval_episodes += 1
            num_eval_timesteps += result.eval_length

            eval_rets.append(result.eval_return)
            eval_lens.append(result.eval_length)
            continue

        results.append(result)
        num_episodes_popped += result.lengths_n2.size
        num_timesteps_popped += result.lengths_n2.sum()

        algo.process_result(result)

        log.debug("%d: %d / %d (%d), %d",
                  task_id,
                  num_episodes_popped, conf.task.episodes_per_batch,
                  int(num_episodes_popped * 100 / conf.task.episodes_per_batch),
                  num_timesteps_popped)

    return IterationResults(results, eval_rets, eval_lens, num_results_skipped,
                            num_eval_episodes + num_episodes_popped,
                            num_eval_timesteps + num_timesteps_popped)


def start(redis_config, conf, data_dir):
    '''
    '''
    master_client = dist.MasterClient(redis_config)
    algo = get_algo_from_exp(conf)

    try:
        _loop_job(master_client, algo, conf, data_dir)
    finally:
        stop()


def stop():
    '''
    '''
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_file",
        help="Experiment configuration file")
    parser.add_argument(
        "-m", "--master_socket_path", default="/tmp/es_redis_master.sock",
        help="Socket path for master redis server (ignored if host is given)")
    parser.add_argument(
        "-r", "--redis_host",
        help="master redis host, overrides master_socket_path")
    parser.add_argument(
        "-p", "--redis_port", default=6379, type=int,
        help="master redis port")
    parser.add_argument(
        "-b", "--debug", default=False, action="store_true",
        help="Enable debug logging")
    parser.add_argument(
        "-s", "--snapshot_file", default=None)
    parser.add_argument(
        "-d", "--data_dir", default=None)

    args = parser.parse_args()

    if args.redis_host:
        redis_config = {'host': args.redis_host, 'port': args.redis_port}
    else:
        redis_config = {'unix_socket_path': args.master_socket_path}

    if args.snapshot_file:
        assert os.path.exists(args.snapshot_file), "Could not find snapshot {0}".format(args.snapshot_file)

    conf = config.build_config(args.exp_file)

    data_dir = config.set_data_dir(args.data_dir)

    log_config.config_debug_logging(args.debug)
    log_config.config_file_logging(data_dir, "master")
    log_config.config_stream_logging()

    try:
        start(redis_config, conf, data_dir)
    except KeyboardInterrupt:
        pass
    finally:
        stop()