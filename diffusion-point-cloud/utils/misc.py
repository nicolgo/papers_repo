import os.path
import random
import time

import numpy.random
import torch
import logging

THOUSAND = 1000
MILLION = 1000000

class BlackHole(object):
    def __setattr__(self, key, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, item):
        return self


class CheckpointManager(object):
    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir,exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger - logger

        for f in os.listdir(self.save_dir):
            if f[:4] != 'ckpt':
                continue
        _, score, it = f.split('_')
        it = it.split('.')[0]
        self.ckpts.append({
            'score': float(score),
            'file': f,
            'iteration': int(it),
        })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None





def str_list(argstr):
    return list(argstr.split(','))


def seed_all(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


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
