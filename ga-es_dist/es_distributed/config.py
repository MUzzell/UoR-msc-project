import json
import logging
import os
import time
import re
import tempfile
from collections import namedtuple

log = logging.getLogger(__name__)

Task = namedtuple('Task', [
    'params', 'ob_mean', 'ob_std', 'ref_batch', 'timestep_limit'
])
Result = namedtuple('Result', [
    'worker_id',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'action_mean','eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count'
])
MasterConfig = namedtuple('MasterConfig', [
    'env_id', 'snapshot_count', 'episodes_per_batch',
    'episodes_target', 'timesteps_target'
])
Policy = namedtuple('Policy', [
    'args', 'type', 'handler'
])

Config = namedtuple('Config', [
    'task', 'algo', 'policy'
])


def set_data_dir(data_dir):

    if not data_dir:
        return tempfile.mkdtemp()

    return data_dir


def build_config(exp_file):
    assert os.path.exists(exp_file), "Exp file does not exist"
    with open(exp_file, 'r') as f:
        exp = json.loads(f.read())
        master = MasterConfig(**exp['task'])
        policy = Policy(**exp['policy'])
        return Config(master, exp['algo'], policy)


def __load_session_from_file(session_file):

    with open(session_file) as f:
        session_data = json.loads(f.read())
        task_id = session_data['task_id']
        episodes_so_far = session_data['episodes_so_far']
        timesteps_so_far = session_data['timesteps_so_far']
        ts_limit = session_data['ts_limit']
        tstart = session_data['tstart']
        best_score = session_data['best_score']

    return (task_id, episodes_so_far, timesteps_so_far,
            ts_limit, tstart, best_score)


def load_session(conf, data_dir):

    task_id = 0
    episodes_so_far = 0
    timesteps_so_far = 0
    ts_limit, ts_th, ts_inc_ratio = _load_timestep_limit(conf)
    tstart = time.time()
    best_score = 0

    session_file = os.path.join(data_dir, 'session.json')
    if os.path.exists(session_file):
        (task_id, episodes_so_far, timesteps_so_far,
         ts_limit, tstart, best_score) = __load_session_from_file(session_file)

    return task_id, episodes_so_far, timesteps_so_far, ts_limit, ts_th, ts_inc_ratio, tstart, best_score


def save_session(data_dir, *args, **kwargs):
    #import pdb; pdb.set_trace()
    session_file = os.path.join(data_dir, 'session.json')
    with open(session_file, 'w') as f:
        f.write(json.dumps(kwargs))


def get_latest_snapshot(data_dir):

    snapshot_files = _find_snapshots(data_dir)

    if not snapshot_files:
        return os.path.join(data_dir, snapshot_files[-1])


def get_next_snapshot(data_dir, snapshot_count, task_id):

    _delete_oldest_snapshot(data_dir, snapshot_count)

    return os.path.join(data_dir, "snapshot_{}.h5".format(task_id))


def _find_snapshots(data_dir):

    return [f for f in os.listdir(data_dir) if re.match("snapshot_\d+.h5", f)]


def _delete_oldest_snapshot(data_dir, snapshot_count):

    snapshot_files = _find_snapshots(data_dir)

    if len(snapshot_files) >= snapshot_count:
        os.remove(os.path.join(data_dir, snapshot_files[0]))


def _load_timestep_limit(conf):

    tslimit_cutoff = conf.algo["episode_cutoff_mode"]

    if isinstance(tslimit_cutoff, int):
        log.info("Starting with timestep limit fixed to %d", tslimit_cutoff)
        return tslimit_cutoff, None, None
    elif tslimit_cutoff.startswith('adaptive:'):
        _, args = tslimit_cutoff.split(':')
        ts_limit, ts_th, ts_inc_ratio = args.split(',')
        ts_limit = int(ts_limit)
        ts_th = float(ts_th)
        ts_inc_ratio = int(ts_limit)
        log.info(
            'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}'.format(
            ts_limit, ts_th * 100, ts_inc_ratio))
        return ts_limit, ts_th, ts_inc_ratio
    elif tslimit_cutoff == 'env_default':
        log.info("Starting with no timestep limit")
        return None, None, None
    else:
        raise NotImplementedError(conf.episode_cutoff_mode)