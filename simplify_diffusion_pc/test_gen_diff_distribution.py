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
from models.model_factory import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/gaussian_986000.pt')
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
model = (get_model_by_type(ckpt['args'].model_type, ckpt['args'])).to(args.device)
model.load_state_dict(ckpt['state_dict'])
logger.info(repr(model))

logger.info('Loading test dataset...')
test_dataset = ShapeNetData(path=args.dataset_path, categories=['airplane'], split='test', scale_mode=args.normalize)
ref_pcs = []
for i, data in enumerate(test_dataset):
    ref_pcs.append(data['point_cloud'].unsqueeze(0))
ref_pcs = torch.cat(ref_pcs, dim=0)  # (N, 2048, 3)


def create_diff_normal_distributions():
    diff_mean = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0)
    diff_var = (0.005, 0.010, 0.020, 0.040, 0.100, 0.200, 0.400, 1.000, 2.000, 4.000)
    diff_mean_list = [(0, 1)] + [(mean, 1.00) for mean in diff_mean] + [(-mean, 1.00) for mean in diff_mean]
    diff_var_list = [(0, 1)] + [(0, 1 - var) for var in diff_var] + [(0, 1 + var) for var in diff_var]
    # diff_mean_var_list = diff_mean_list + diff_var_list
    return diff_mean_list, diff_var_list


diff_mean_dis, diff_var_dis = create_diff_normal_distributions()
all_res = dict()
for (diff_mean, diff_var) in diff_mean_dis:
    logger.info(f'Generating the point clouds for ({diff_mean},{diff_var})')
    gen_pcs = []
    for i in tqdm(range(0, math.ceil(len(test_dataset) / args.batch_size))):
        with torch.no_grad():
            # z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
            z = torch.normal(mean=diff_mean, std=diff_var, size=[args.batch_size, ckpt['args'].latent_dim]).to(
                args.device)
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

    # all_res[(diff_mean, diff_var)] = [results['lgan_mmd'], results['lgan_cov'], results['1-NN-CD-acc'], results['jsd']]
    all_res[(diff_mean, diff_var)] = results
    logger.info(all_res)

## generate the final results