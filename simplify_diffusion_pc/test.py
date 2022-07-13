from utils.misc import *
from torch.utils.data import dataset, DataLoader
from utils.dataset import *
from models.vae_gaussian import *
from models.diffusion import *
import time


def show_diffusion_process(x_0_t):
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


def show_point_cloud():
    train_dataset = ShapeNetData(path='./data/shapenet.hdf5', categories=['airplane'], split='train',
                                 scale_mode='shape_unit')
    train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=1, num_workers=0))
    batch = next(train_iter)
    x = batch['point_cloud']
    diffusion_model = DiffusionPoint()
    x_0_t = diffusion_model.diffusion_process(x)
    temp = x_0_t[1]
    show_diffusion_process(x_0_t)

    # reverse diffusion process
    x_t_0 = diffusion_model.reverse_with_x0_xt(x)
    show_list = []
    for key, value in x_t_0.items():
        show_list.append(x_t_0[key].squeeze(dim=0))
    show_diffusion_process(show_list)


if __name__ == "__main__":
    # show_point_cloud()
    BACKUP_PATH = "\\\\COMPDrive\Student1\\21042139g\\COMProfile\\Documents\\Backup"
    backup_name = "train_501_102"
    backup_path = os.path.join(BACKUP_PATH, backup_name)
    backup_training_files(source_path="D:\papers_repo\simplify_diffusion_pc\logs_gen\GEN_2022_07_13__15_58_25",
                          target_path=BACKUP_PATH)
