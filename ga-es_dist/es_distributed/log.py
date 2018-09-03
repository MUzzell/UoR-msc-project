import errno
import logging
import sys
import os

def mkdir_p(path):
    path = os.path.expanduser(path)
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def config_debug_logging(debug=False):
    logging.getLogger().setLevel(
        logging.DEBUG if debug else logging.INFO)

def config_stream_logging():
    logging.getLogger().addHandler(
        logging.StreamHandler(stream=sys.stdout)
    )
def config_file_logging(log_dir, log_name):
    mkdir_p(log_dir)
    logging.getLogger().addHandler(logging.FileHandler(
        "{0}/{1}".format(log_dir, log_name)
    ))

    data_logger = logging.getLogger("data")
    data_file_formatter = logging.FileHandler(
        "{0}/{1}".format(log_dir, "data"))
    data_file_formatter.setFormatter(
        logging.Formatter("%(message)s"))
    data_logger.handlers = []
    data_logger.addHandler(data_file_formatter)
