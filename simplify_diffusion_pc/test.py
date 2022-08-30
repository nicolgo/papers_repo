import numpy as np
import torch

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
    random_distribu = [(0, 1)]
    z_all = z_total
    for mean, std in random_distribu:
        z_all = torch.cat((z_all, torch.normal(mean=mean, std=std, size=z_total.size())), dim=0)

    labels = np.zeros(z_total.size(0))
    for i in range(len(random_distribu)):
        labels = np.concatenate((labels, np.ones(z_total.size(0)) * (i + 1)), axis=0)
    Y = tsne(z_all.numpy(), 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    # pylab.scatter(Y[:, 0], Y[:, 1], 20)
    pylab.show()


def show_diff_scale_pcl():
    diff_scale_modes = ('shape_unit', 'shape_half', 'shape_bbox', 'original')
    pcl_sets = []
    for diff_scale_mode in diff_scale_modes:
        train_dataset = ShapeNetData(path='./data/shapenet.hdf5', categories=['airplane'], split='train',
                                     scale_mode=diff_scale_mode)
        train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=1, num_workers=0))
        batch = next(train_iter)
        x = batch['point_cloud']
        pcl_sets.append(x[0].numpy())

    fig = plt.figure()
    for i in range(1, 5):
        numpy_pcd = pcl_sets[i - 1]
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.set_title(diff_scale_modes[i - 1])
        # ax.set_axis_off()
        ax.scatter(numpy_pcd[:, 0], numpy_pcd[:, 1], numpy_pcd[:, 2])
    plt.show()


def show_sample_process():
    # load model
    ckpt = torch.load('./pretrained/no_z_1072000.pt')
    ckpt['args'].latent_dim = 0
    model = (get_model_by_type(ckpt['args'].model_type, ckpt['args'])).to('cuda')
    model.load_state_dict(ckpt['state_dict'])

    z = torch.randn([1, 256]).to('cuda')
    if ckpt['args'].model_type == 'flow':
        z = model.flow(z, reverse=True).view(z.size(0), -1)
    z = None
    samples = model.diffusion.reverse_sample(2048, None, batch_size=1, device=torch.device('cuda'), ret_traj=True)

    fig = plt.figure()
    for i in range(1, 101):
        value = samples[i - 1].cpu().numpy()
        numpy_pcd = value[0]
        ax = fig.add_subplot(10, 10, i, projection='3d')
        # ax.set_axis_off()
        ax.scatter(numpy_pcd[:, 0], numpy_pcd[:, 1], numpy_pcd[:, 2])
    plt.show()
    pass


def sample_one_res(check_point_file, noise, batch_size, device):
    ckpt_flow = torch.load('./pretrained/z_dims/' + check_point_file)
    model_flow = (get_model_by_type(ckpt_flow['args'].model_type, ckpt_flow['args'])).to('cuda')
    model_flow.load_state_dict(ckpt_flow['state_dict'])
    with torch.no_grad():
        z = torch.randn([batch_size, ckpt_flow['args'].latent_dim]).to(device)
        z = model_flow.flow(z, reverse=True).view(z.size(0), -1)
        sample_flow = model_flow.diffusion.reverse_sample(2048, z, batch_size=1, device=device, noise=noise)
    return sample_flow[0].cpu().numpy()


def create_pcl(x_numpy, color):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(x_numpy)
    pcl.paint_uniform_color(color)
    return pcl


def compare_sample_z_dim_8421():
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn([batch_size, 2048, 3]).to(device)
    file_name = ('dim1_548000.pt', 'dim2_545000.pt', 'dim4_548000.pt', 'dim8_547000.pt')
    colors = ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    pcl_list = []
    for file, color in zip(file_name, colors):
        x = sample_one_res(file, noise, batch_size, device)
        pcl_list.append(create_pcl(x, color))
    o3d.visualization.draw_geometries(pcl_list)


def compare_sample_with_z_and_no_z():
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn([batch_size, 2048, 3]).to(device)

    ckpt = torch.load('./pretrained/no_z_1072000.pt')
    ckpt['args'].latent_dim = 0
    model = (get_model_by_type(ckpt['args'].model_type, ckpt['args'])).to(device)
    model.load_state_dict(ckpt['state_dict'])
    with torch.no_grad():
        sample = model.diffusion.reverse_sample(2048, None, batch_size=batch_size, device=device, noise=noise)

    ckpt_flow = torch.load('./pretrained/flow_1287000.pt')
    model_flow = (get_model_by_type(ckpt_flow['args'].model_type, ckpt_flow['args'])).to('cuda')
    model_flow.load_state_dict(ckpt_flow['state_dict'])
    with torch.no_grad():
        z = torch.randn([batch_size, ckpt_flow['args'].latent_dim]).to(device)
        z = model_flow.flow(z, reverse=True).view(z.size(0), -1)
        sample_flow = model_flow.diffusion.reverse_sample(2048, z, batch_size=batch_size, device=device, noise=noise)

    pcl_list = []
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(sample[0].cpu().numpy())
    pcl.paint_uniform_color([1, 0, 0])
    pcl_list.append(pcl)

    pcl2 = o3d.geometry.PointCloud()
    pcl2.points = o3d.utility.Vector3dVector(sample_flow[0].cpu().numpy())
    pcl2.paint_uniform_color([0, 0, 1])
    pcl_list.append(pcl2)

    o3d.visualization.draw_geometries(pcl_list)


def convert_pcl_to_image(category, num=20):
    train_dataset = ShapeNetData(path='./data/shapenet.hdf5', categories=[category], split='train',
                                 scale_mode='shape_unit')
    train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=num, num_workers=0))
    batch = next(train_iter)
    x = batch['point_cloud']
    os.makedirs(f'./images/{category}', exist_ok=True)
    param = o3d.io.read_pinhole_camera_parameters('position.json')
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcl = o3d.geometry.PointCloud()
    for i in range(num):
        pcl.points = o3d.utility.Vector3dVector(x[i])
        if i == 0:
            vis.add_geometry(pcl)
        else:
            vis.update_geometry(pcl)
        # time.sleep(0.1)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)
        # param = ctr.convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters('position.json',param)
        vis.capture_screen_image(f"./images/{category}/{category}_{i}.jpg", do_render=True)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()


def save_all_pcd_to_images():
    categories = list(synsetid_to_cate.values())
    for category in categories:
        convert_pcl_to_image(category, 20)


def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None

    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]
    # see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# TODO: calculate the kl divergence to measure data distribution
def calculation_kl_xt_and_standard():
    batch_size = 8
    for scale_mode in ('shape_unit', 'shape_half', 'shape_bbox', 'original'):
        train_dataset = ShapeNetData(path='./data/shapenet.hdf5', categories=['airplane'], split='train',
                                     scale_mode=scale_mode)
        train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=batch_size, num_workers=0))
        batch = next(train_iter)
        x = batch['point_cloud']
        diffusion_model = DiffusionPoint()
        mean_xt_on_x0 = diffusion_model.alpha_bar_sqrt[100] * x
        log_var1 = torch.log((1 - diffusion_model.alpha_bar[100]) * torch.ones(x.size()))
        mean_standard = torch.zeros(x.size())
        log_var_standard = torch.zeros(x.size())
        kl = normal_kl(mean_xt_on_x0, log_var1, mean_standard, log_var_standard)
        kl = mean_flat(kl) / np.log(2.0)
        print(f'The kl of {scale_mode} is {kl}')
    # loss = torch.nn.functional.kl_div(torch.log(standard_pts), xt_on_x0)
    # print(loss)


if __name__ == "__main__":
    # show_point_cloud()
    # only_leave_latest_file("D:\GEN_2022_07_04__18_50_41")
    # show_latent_space()
    # show_diff_scale_pcl()
    # show_sample_process()
    # calculation_kl_xt_and_standard()
    compare_sample_with_z_and_no_z()
    # compare_sample_z_dim_8421()
    # train_dataset = ShapeNetData(path='./data/shapenet.hdf5', categories=['airplane'], split='train',
    #                              scale_mode='shape_unit')
    # train_iter = get_data_iterator(DataLoader(train_dataset, batch_size=128, num_workers=0))
    # batch = next(train_iter)
    # x = batch['point_cloud']
    # pcd = x.numpy()
    # show_pcds_with_plot(pcd)
