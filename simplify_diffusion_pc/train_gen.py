import argparse

import torch.utils.tensorboard
import torch
from torch.utils.data import dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from utils.misc import *
from utils.dataset import *
from models.vae_gaussian import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--val_batch_size', type=int, default=64)

# Model arguments
parser.add_argument('--latent_dim', type=int, default=256)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200 * THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=400 * THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=30 * THOUSAND)
parser.add_argument('--test_size', type=int, default=400)

args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Initialize the logger
log_dir = get_new_log_dir('./logs_gen', prefix='GEN_', postfix='')
logger = get_logger('train', log_dir)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)

logger.info('Loading dataset...')
train_dataset = ShapeNetData(path=args.dataset_path, categories=['airplane'], split='train', scale_mode=args.scale_mode)
val_dataset = ShapeNetData(path=args.dataset_path, categories=['airplane'], split='val', scale_mode=args.scale_mode)
train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=0))

logger.info('create model')
model = GaussianVAE(args).to(args.device)
# model.eval()
# x = (next(train_iter))['point_cloud']
# writer.add_graph(model, x)
# writer.flush()
logger.info(repr(model))

# optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_linear_scheduler(optimizer, start_epoch=args.sched_start_epoch, end_epoch=args.sched_end_epoch,
                                 start_lr=args.lr, end_lr=args.end_lr)


def train(iteration_id):
    x = (next(train_iter))['point_cloud'].to(args.device)
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    # Forward Process
    model(x)
    # Loss
    loss = model.get_loss()
    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    # record training info
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % ( iteration_id, loss.item(), orig_grad_norm))
    pass


def validate_inspect(iteration_id):
    pass


def test(iteration_id):
    ref_pcs = []
    for i, data in enumerate(val_dataset):
        if i >= args.test_size:
            break
        ref_pcs.append(data['point_cloud'].unsqueeze(0))
    ref_pcs = torch.cat(ref_pcs, dim=0)

    gen_pcs = []

    pass


if __name__ == '__main__':
    logger.info('Start training...')
    try:
        i = 1
        while i <= args.max_iters:
            train(i)
            if i % args.val_freq == 0:
                validate_inspect(i)
            if i % args.test_freq == 0:
                test(i)
            i += 1
    except KeyboardInterrupt:
        logger.info('Terminating...')
