import os.path
import time
import logging
import logging.handlers


def get_new_log_dir(root='./logs', postfix='', prefix=''):
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    os.makedirs(log_dir)
    return log_dir


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger



