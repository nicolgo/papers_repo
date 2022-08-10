from utils.misc import *
from torch.utils.data import dataset, DataLoader
from utils.dataset import *
from models.vae_gaussian import *
from models.diffusion import *
from models.model_factory import *
from utils.tsne import *
from utils.visualization_pcd import *
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


def show_latent_space():
    train_dataset = ShapeNetData(path='./data/shapenet.hdf5', categories=['airplane'], split='train',
                                 scale_mode='shape_unit')
    train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=128, num_workers=0))
    # load model
    ckpt = torch.load('./pretrained/flow_1983000.pt')
    model = (get_model_by_type(ckpt['args'].model_type, ckpt['args']))
    model.load_state_dict(ckpt['state_dict'])
    # w = torch.randn([128, ckpt['args'].latent_dim])
    # z_context = model.flow(w, reverse=True).view(w.size(0), -1)
    z_total = []
    for i in range(10):
        batch = next(train_iter)
        x = batch['point_cloud']
        with torch.no_grad():
            z_mean, z_log_var = model.encoder(x)
            z_context = reparameterize_gaussian(z_mean, z_log_var).view(x.size(0), -1)
            if i == 0:
                z_total = z_context
            else:
                z_total = torch.cat([z_total, z_context])

    Y = tsne(z_total.numpy(), 2, 50, 20.0)
    # pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.scatter(Y[:, 0], Y[:, 1], 20)
    pylab.show()


if __name__ == "__main__":
    # show_point_cloud()
    # only_leave_latest_file("D:\GEN_2022_07_04__18_50_41")
    show_latent_space()
    # train_dataset = ShapeNetData(path='./data/shapenet.hdf5', categories=['airplane'], split='train',
    #                              scale_mode='shape_unit')
    # train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=128, num_workers=0))
    # batch = next(train_iter)
    # x = batch['point_cloud']
    # pcd = x.numpy()
    # show_pcds_with_plot(pcd)
