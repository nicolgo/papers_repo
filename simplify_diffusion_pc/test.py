from utils.misc import *
from torch.utils.data import dataset, DataLoader
from utils.dataset import *
from models.vae_gaussian import *
from models.diffusion import *

if __name__ == "__main__":
    train_dataset = ShapeNetData(path='./data/shapenet.hdf5', categories=['airplane'], split='train',
                                 scale_mode='shape_unit')
    train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=1, num_workers=0))
    batch = next(train_iter)
    x = batch['point_cloud']

    diffusion_model = DiffusionPoint()
    x_0_t = diffusion_model.diffusion_process(x)

    # visualize point cloud
    pcl = o3d.geometry.PointCloud()
    for i in range(0, 100, 10):
        pcl.points = o3d.utility.Vector3dVector(x_0_t[i])
        o3d.visualization.draw_geometries([pcl])
