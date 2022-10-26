import imageio


def save_as_gif(images, file_name):
    images = images.cpu().permute(1, 2, 3, 0)
    image_list = []
    for i in range(images.shape[0]):
        image_list.append(images[i])
    imageio.mimwrite(file_name, image_list, fps=4)
