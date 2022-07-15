import argparse
import math

from tqdm.auto import tqdm
import torch.utils.tensorboard
import torch
from torch.utils.data import dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from utils.misc import *
from utils.dataset import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.vae_diffusion import *
from evaluation import *

# Global directory Path
BACKUP_PATH = "\\\\COMPDrive\Student1\\21042139g\\COMProfile\\Documents\\Backup"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=64)

# Model arguments
parser.add_argument('--model_type', type=str, default='flow', choices=['flow', 'gaussian', 'pure'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--num_samples', type=int, default=6)
parser.add_argument('--sample_num_points', type=int, default=2048)
# Flow Model arguments
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)

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

# Resume the training Process from the backup directory
parser.add_argument('--backup_dir', type=str, default="train_507_102")
parser.add_argument('--resume', type=str, default=None)  # pass the check point file absolute path
parser.add_argument('--resume_step', type=int, default=1)  # save the resume step

args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_backup_path = os.path.join(BACKUP_PATH, args.backup_dir)
if os.path.exists(my_backup_path):
    args.backup_dir = args.backup_dir + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    my_backup_path = os.path.join(BACKUP_PATH, args.backup_dir)

seed_all(2022)  # give a value to get a reproducible result

if args.resume is None:
    # Initialize the logger
    log_dir = get_new_log_dir('./logs_gen', prefix='GEN_', postfix='')
else:
    log_dir = os.path.dirname(args.resume)

logger = get_logger('train', log_dir)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)
ckpt_mgr = CheckpointManager(log_dir)

# for resume, no need override the parameters, just load the model
ckpt = None
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = get_model_by_type(ckpt['args'].model_type, args)
    model.load_state_dict(ckpt['state_dict'])
    args.resume_step = ckpt['args'].resume_step if (args.resume_step == 1) else args.resume_step
    args.model_type = ckpt['args'].model_type  # update the value for next resume
else:
    logger.info('create model')
    model = get_model_by_type(args.model_type, args)

logger.info('Loading dataset...')
train_dataset = ShapeNetData(path=args.dataset_path, categories=['airplane'], split='train', scale_mode=args.scale_mode)
val_dataset = ShapeNetData(path=args.dataset_path, categories=['airplane'], split='val', scale_mode=args.scale_mode)
train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=0))

# save and visualize the model
model.eval()
x = (next(train_iter))['point_cloud'].to(args.device)
writer.add_graph(model, x)
writer.flush()
logger.info(repr(model))

# optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_linear_scheduler(optimizer, start_epoch=args.sched_start_epoch, end_epoch=args.sched_end_epoch,
                                 start_lr=args.lr, end_lr=args.end_lr)


def train(iteration_id):
    x0 = (next(train_iter))['point_cloud'].to(args.device)
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    # Forward and Loss
    model(x0)
    loss = model.get_loss(writer=writer, iteration_id=iteration_id)
    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    # record training info
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (iteration_id, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, iteration_id)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iteration_id)
    writer.add_scalar('train/grad_norm', orig_grad_norm, iteration_id)
    writer.flush()


def validate_inspect(iteration_id):
    z_context = torch.randn([args.num_samples, args.latent_dim]).to(args.device)
    gen_pcl = model.sample(z_context, args.sample_num_points)

    logger.info('[Inspect] Generating samples...')
    writer.add_mesh('val/point_cloud', gen_pcl, global_step=iteration_id)
    writer.flush()


def test(iteration_id):
    ref_pcs = []
    for i, data in enumerate(val_dataset):
        if i >= args.test_size:
            break
        ref_pcs.append(data['point_cloud'].unsqueeze(0))
    ref_pcs = torch.cat(ref_pcs, dim=0)

    gen_pcs = []
    for temp in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
        with torch.no_grad():
            z_context = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
            gen_pcl = model.sample(z_context, args.sample_num_points)
            gen_pcs.append(gen_pcl.detach().cpu())
    gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]

    # evaluation
    with torch.no_grad():
        results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.val_batch_size)
        results = {k: v.item() for k, v in results.items()}
        jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
        results['jsd'] = jsd

    # record information
    logger.info('[Test] Coverage  | CD %.6f | EMD n/a' % (results['lgan_cov-CD'],))
    logger.info('[Test] MinMatDis | CD %.6f | EMD n/a' % (results['lgan_mmd-CD'],))
    logger.info('[Test] 1NN-Accur | CD %.6f | EMD n/a' % (results['1-NN-CD-acc'],))
    logger.info('[Test] JsnShnDis | %.6f ' % (results['jsd']))

    writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], global_step=iteration_id)
    writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], global_step=iteration_id)
    writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], global_step=iteration_id)
    writer.add_scalar('test/JSD', results['jsd'], global_step=iteration_id)
    writer.flush()


def backup_current_training_files(i, log_dir):
    try:
        backup_training_files(log_dir, my_backup_path)
        logger.info(f"backup successfully at {i} step!")
    except Exception as e:
        logger.warning(f"failed to backup training files at {i} step!")
        logger.warning('An exception occurred: {}'.format(e))
        pass


if __name__ == '__main__':
    logger.info('Start training...')
    if args.resume is None:
        i = 1
    else:
        i = args.resume_step + 1

    try:
        while i <= args.max_iters:
            train(i)
            if i % args.val_freq == 0:
                validate_inspect(i)
                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                # save args & model & optimizer & scheduler
                args.resume_step = i
                ckpt_mgr.save(model, args, score=0, others=opt_states, step=i)
                # backup intermediate files
                backup_current_training_files(i, log_dir)
            if i % args.test_freq == 0:
                test(i)
            i += 1
    except KeyboardInterrupt:
        logger.info('Terminating...')
