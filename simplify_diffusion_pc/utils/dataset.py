import random
from copy import copy
import h5py
import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
from torch.utils.data import DataLoader

RANDOM_SEED = 2020


def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    vis.run()
    vis.destroy_window()


synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetData(Dataset):
    def __init__(self, path, categories, split, scale_mode, transform=None):
        self.split_types = ('train', 'val', 'test')
        self.path = path
        if 'all' in categories:
            categories = cate_to_synsetid
        self.category_ids = [cate_to_synsetid[s] for s in categories]
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.point_clouds = []
        self.statistics = None  # mean and std

        self.get_statistics()
        self.load()

    def get_statistics(self):
        """
            traversal all dataset to get mean and std
        :return: mean and std
        """
        with h5py.File(self.path, 'r') as f:
            point_clouds = []
            for category_id in self.category_ids:
                for split in self.split_types:
                    point_clouds.append(torch.from_numpy(f[category_id][split][...]))
        all_points = torch.cat(point_clouds, dim=0)
        B, N, _ = all_points.size()
        mean = all_points.view(B * N, -1).mean(dim=0)
        std = all_points.view(-1).std(dim=0)

        self.statistics = {'mean': mean, 'std': std}
        return self.statistics

    def load(self):
        def _enumerate_point_clouds(f):
            for tmp_id in self.category_ids:
                category = synsetid_to_cate[tmp_id]
                for j, pc in enumerate(f[tmp_id][self.split]):
                    yield torch.from_numpy(pc), j, category

        with h5py.File(self.path, mode='r') as f:
            for pc, pc_id, category_name in _enumerate_point_clouds(f):
                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1)
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

                pc = (pc - shift) / scale

                self.point_clouds.append({
                    'point_cloud': pc,
                    'category': category_name,
                    'id': pc_id,
                    # 'shift': shift,
                    # 'scale': scale
                })
        self.point_clouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(RANDOM_SEED).shuffle(self.point_clouds)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        data = {k: v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.point_clouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def analysis_point_cloud(data_set, scale_mode=None):
    my_pcl = []
    for i, item in enumerate(data_set):
        my_pcl.append(item['point_cloud'].unsqueeze(dim=0))
    my_pcl = torch.cat(my_pcl, dim=0)
    xyz_max = torch.amax(my_pcl, dim=(0, 1))
    xyz_min = torch.amin(my_pcl, dim=(0, 1))
    print(f"for {scale_mode}, xyz_max is {xyz_max}, xyz_min is {xyz_min}")
    return xyz_max, xyz_min


def get_different_normalized_pcl(scale_mode):
    data_set = ShapeNetData(path='../data/shapenet.hdf5', categories=['airplane'], split='train',
                            scale_mode=scale_mode)
    analysis_point_cloud(data_set, scale_mode)
    cloud_iter = get_data_iterator(DataLoader(data_set, batch_size=2, num_workers=0))
    batch = next(cloud_iter)
    x = batch['point_cloud']
    return x


if __name__ == "__main__":
    x = get_different_normalized_pcl('shape_unit')
    x2 = get_different_normalized_pcl('original')
    x3 = get_different_normalized_pcl('shape_bbox')

    cloud_list = []
    pcl_list = []
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(x[0])
    cloud_list.append(pcl)

    pcl2 = o3d.geometry.PointCloud()
    pcl2.points = o3d.utility.Vector3dVector(x2[0])
    cloud_list.append(pcl2)

    # pcl3 = o3d.geometry.PointCloud()
    # pcl3.points = o3d.utility.Vector3dVector(x3[0])
    # cloud_list.append(pcl3)

    o3d.visualization.draw_geometries(cloud_list)
    pass
    # o3d.visualization.draw_geometries([pcl])
    # custom_draw_geometry(pcl)
