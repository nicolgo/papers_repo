import argparse
import math

import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.misc import *
from utils.dataset import *
from utils.data import *
from models.vae_flow import *
from models.vae_gaussian import *
from models.flow import *
from models.diffusion import *


def get_parameters():
    parser = argparse.ArgumentParser()
    # Dataset and loaders
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
    parser.add_argument('--categories', type=str_list, default=['airplane'])
    parser.add_argument('--scale_mode', type=str, default='shape_unit')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=64)

    # Model arguments
    parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
    parser.add_argument('--latent_dim', type=int, default=256)  # pointnet encoder output
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)

    # Training
    parser.add_argument('--seed', type=int, default=2020)  # reproduce same result
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./logs_gen')
    parser.add_argument('--tag', type=str, default=None)  # log tag
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_iters', type=int, default=float('inf'))
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--test_freq', type=int, default=30 * THOUSAND)
    parser.add_argument('--test_size', type=int, default=400)

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--end_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--sched_start_epoch', type=int, default=200 * THOUSAND)
    parser.add_argument('--sched_end_epoch', type=int, default=400 * THOUSAND)

    return parser.parse_args()


def setting_logger(args):
    if args.logging:
        log_dir = get_new_log_dir(args.log_root, prefix='GEN_',
                                  postfix='_' + args.tag if args.tag is not None else '')
        logger = get_logger(name='train', log_dir=log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        ckpt_mgr = CheckpointManager(log_dir)
    else:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_mgr = BlackHole()
    return logger, writer, ckpt_mgr


def train(model, args, optimizer, scheduler, train_iter, it, writer):
    # Load data
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    kl_weight = args.kl_weight
    loss = model.get_loss(x, kl_weight=kl_weight, writer=writer, it=it)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        it, loss.item(), orig_grad_norm, kl_weight
    ))

    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/kl_weight', kl_weight, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)


def validate_inspect(args, model, it, writter, logger):
    z = torch.randn([args.num_samples, args.latent_dim]).to(args.device)
    x = model.sample(z, args.sample_num_points, flexibility=args.flexibility)
    writer.add_mesh('val/pointcloud', x, global_step=it)
    writer.flush()
    logger.info(['[Inspect] Generating samples...'])


def test(it):
    ref_pcs = []
    for i, data in enumerate(val_dataset):
        if i >= args.test_size:
            break
        ref_pcs.append(data['pointcloud'].unsqueeze(0))
    ref_pcs = torch.cat(ref_pcs, dim=0)

    gen_pcs = []
    for i in tqdm(range(0,math.ceil(args.test_size/args.val_batch_size)),'Generate'):
        with torch.no_grad():
            z = torch.randn([args.val_batch_size,args.latent_dim]).to(args.device)
            x = model.sample(z,args.sample_num_points, flexibility=args.flexibility)
            gen_pcs.append(x.detach().cpu())
    gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]

    with torch.no_grad():
        results = compute_all_metrics(gen_pcs.to(args.device),ref_pcs.to(args.device),args.val_batch_size)


if __name__ == '__main__':
    # get parameters
    args = get_parameters()
    seed_all(args.seed)

    # setting log
    logger, writer, ckpt_mgr = setting_logger(args)

    logger.info(args)
    logger.info('Loading datasets...')
    # get dataset
    train_dataset = ShapeNetCore(path=args.dataset_path, cates=args.categories, split='train',
                                 scale_mode=args.scale_mode)
    val_dataset = ShapeNetCore(path=args.dataset_path, cates=args.categories, split='val',
                               scale_mode=args.scale_mode)
    train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=0))

    # Model
    logger.info('Building model...')
    if args.model == 'gaussian':
        model = GaussianVAE(args).to(args.device)
    elif args.model == 'flow':
        model = FlowVAE(args).to(args.device)
    logger.info(repr(model))

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_scheduler(optimizer, start_epoch=args.sched_start_epoch, end_epoch=args.sched_end_epoch,
                                     start_lr=args.lr, end_lr=args.endlr)

    logger.info('Start training...')
    try:
        it = 1
        while it <= args.max_iters:
            # train(it)
            train(model, args, optimizer, scheduler, train_iter, it, writer)
            if it % args.val_freq == 0 or it == args.max_iters:
                validate_inspect(it)
                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
            if it % args.test_freq == 0 or it == args.max_iters:
                test(it)
            it += 1
    except KeyboardInterrupt:
        logger.info('Terminating...')
