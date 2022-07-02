from utils.misc import *
from torch.utils.data import dataset, DataLoader
from utils.dataset import *
from models.vae_gaussian import *
from models.diffusion import *
import time


def show_diffusion_process(x):
    diffusion_model = DiffusionPoint()
    x_0_t = diffusion_model.diffusion_process(x)

    # visualize the diffusion process of point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcl = o3d.geometry.PointCloud()
    for i in range(0, 100, 1):
        pcl.points = o3d.utility.Vector3dVector(x_0_t[i])
        if i == 0:
            vis.add_geometry(pcl)
        else:
            vis.update_geometry(pcl)
        time.sleep(0.1)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()


if __name__ == "__main__":
    ts = np.random.choice(np.arange(1, 100 + 1), 5)
    print(ts.tolist())
    train_dataset = ShapeNetData(path='./data/shapenet.hdf5', categories=['airplane'], split='train',
                                 scale_mode='shape_unit')
    train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=1, num_workers=0))
    batch = next(train_iter)
    x = batch['point_cloud']

    show_diffusion_process(x)
