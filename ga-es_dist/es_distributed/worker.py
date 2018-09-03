import time
import os
import argparse
import logging

from . import log as log_config
from . import dist
from . import algos
from .util import get_algo_from_exp
from collections import namedtuple

log = logging.getLogger(__name__)

IterationResults = namedtuple('IterationResults', [
    'results', 'eval_returns', 'eval_lengths', 'results_skipped',
    'episodes_popped', 'timesteps_popped'
])


def _start_workers(num_workers, master_redis_cfg, relay_redis_cfg):
    for _ in range(num_workers):
        if os.fork() == 0:
            try:
                log.debug("Starting worker")
                start(master_redis_cfg, relay_redis_cfg)
            except KeyboardInterrupt:
                log.info("Worker stopped by user")

def _loop_job(worker_client):
    '''
    '''
    current_env_id = None
    algo = None

    while True:
        exp = worker_client.get_experiment()

        if current_env_id != exp.task.env_id:
            algo = get_algo_from_exp(exp)
            algo.setup_job(exp)
            current_env_id = exp.task.env_id

        task_id, task_data = worker_client.get_current_task()

        result = algo.run_episode(exp, task_id, task_data)

        worker_client.push_result(task_id, result)



def start(master_redis_config, worker_redis_config):
    '''
    '''
    worker_client = dist.WorkerClient(master_redis_config, worker_redis_config)

    try:
        _loop_job(worker_client)
    finally:
        stop()


def stop():
    '''
    '''
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "redis_host",
        help="master redis host")
    parser.add_argument(
        "-r", "--relay_socket_path", default="/tmp/es_redis_relay.sock",
        help="Socket path for relay redis server")
    parser.add_argument(
        "-p", "--redis_port", default=6379, type=int,
        help="master redis port")
    parser.add_argument(
        "-n", "--num_workers", default=None, type=int,
        help="Number of worker threads to spawn")
    parser.add_argument(
        "-d", "--debug", default=False, action="store_true",
        help="Enable debug logging")

    args = parser.parse_args()

    log_config.config_debug_logging(args.debug)
    log_config.config_stream_logging()

    master_redis_cfg = {'host': args.redis_host, 'port': args.redis_port}
    relay_redis_cfg = {'unix_socket_path': args.relay_socket_path}
    num_workers = args.num_workers if args.num_workers else os.cpu_count() -1

    if os.fork() == 0:
        try:
            log.debug("Starting RelayClient")
            dist.RelayClient(master_redis_cfg, relay_redis_cfg).run()
        except KeyboardInterrupt:
            log.info("RelayClient stopped by user")
    else:
        _start_workers(num_workers, master_redis_cfg, relay_redis_cfg)

    os.wait()