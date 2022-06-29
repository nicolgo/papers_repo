import argparse

import torch.utils.tensorboard
import torch
from torch.utils.data import dataset, DataLoader

from utils.misc import *
from utils.dataset import *
from models.vae_gaussian import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=64)

# Model arguments
parser.add_argument('--latent_dim', type=int, default=256)

args = parser.parse_args()

## Initialize the logger
log_dir = get_new_log_dir('./logs_gen', prefix='GEN_', postfix='')
logger = get_logger('train', log_dir)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)

logger.info('Loading dataset...')
train_dataset = ShapeNetData(path=args.dataset_path, categories=['airplane'], split='train', scale_mode=args.scale_mode)
val_dataset = ShapeNetData(path=args.dataset_path, categories=['airplane'], split='val', scale_mode=args.scale_mode)
train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=0))

logger.info('create model')
model = GaussianVAE(args)
model.eval()
batch = next(train_iter)
x = batch['point_cloud']
y = model(x)
writer.add_graph(model, x)
writer.flush()
if __name__ == '__main__':
    pass
