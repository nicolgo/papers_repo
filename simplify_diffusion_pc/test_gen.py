# Arguments
import argparse
import math
import os.path
import time
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from utils.misc import *
from utils.dataset import *
from models.vae_gaussian import *
from models.vae_flow import *
from evaluation import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/airplane_90000.pt')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=64)
# Sampling
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
args = parser.parse_args()

# Initialize logger
log_test_dir = os.path.join('./results', 'GEN_Test_%s_%d' % ('_'.join(['airplane']), int(time.time())))
if not os.path.exists(log_test_dir):
    os.makedirs(log_test_dir)
logger = get_logger('test', log_test_dir)

logger.info('Loading model...')
ckpt = torch.load(args.ckpt)
if ckpt['args'].model_type == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model_type == 'flow':
    model = FlowVAE(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])
logger.info(repr(model))

logger.info('Loading test dataset...')
test_dataset = ShapeNetData(path=args.dataset_path, categories=['airplane'], split='test', scale_mode=args.normalize)
ref_pcs = []
for i, data in enumerate(test_dataset):
    ref_pcs.append(data['point_cloud'].unsqueeze(0))
ref_pcs = torch.cat(ref_pcs, dim=0)  # (N, 2048, 3)

logger.info('Generating the point clouds...')
gen_pcs = []
for i in tqdm(range(0, math.ceil(len(test_dataset) / args.batch_size))):
    with torch.no_grad():
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        x = model.sample(z, ckpt['args'].sample_num_points)
        gen_pcs.append(x.detach().cpu())
gen_pcs = torch.cat(gen_pcs, dim=0)[:len(test_dataset)]  # make the size same as the size of test dataset
if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)

# logger.info('Saving the generated point clouds...')
# np.save(os.path.join(log_test_dir, 'out.npy'), gen_pcs.numpy())


logger.info('Evaluate the quality of generated point clouds.')
with torch.no_grad():
    results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.batch_size)
    results = {k: v.item() for k, v in results.items()}
    jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
