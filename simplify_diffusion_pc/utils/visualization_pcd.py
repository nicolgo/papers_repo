import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def show_pcd_with_plot(numpy_pcd, is_downsample=False):
    """
    numpy_pcd: (N,3)
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(numpy_pcd[:, 0], numpy_pcd[:, 1], numpy_pcd[:, 2])
    plt.show()


def down_sample_pcd(numpy_pcd):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(numpy_pcd)
    # downpcd = pcd.voxel_down_sample(voxel_size=0.005)
    downpcd = pcd.uniform_down_sample(4)
    return np.asarray(downpcd.points)


def show_pcds_with_plot(numpy_pcds):
    fig = plt.figure()
    for i in range(1, 129):
        numpy_pcd = down_sample_pcd(numpy_pcds[i - 1])
        ax = fig.add_subplot(12, 12, i, projection='3d')
        ax.set_axis_off()
        ax.scatter(numpy_pcd[:, 0], numpy_pcd[:, 1], numpy_pcd[:, 2])
    plt.show()
