import os
import numpy as np
import matplotlib.pyplot as plt

from utils import read_image, show_images, read_labels, DATASET_PATH, split_train_val


def demo_show():
    filenames = sorted(os.listdir(os.path.join(DATASET_PATH, 'Origin')))
    image_names = list(map(lambda s: s.split('.')[0], filenames))

    for image_name in image_names:
        org = read_image(os.path.join(DATASET_PATH, 'Origin', image_name + '.png'))
        exp = read_image(os.path.join(DATASET_PATH, 'Expert', image_name + '_expert.png'))
        smp1 = read_image(os.path.join(DATASET_PATH, 'sample_1', image_name + '_s1.png'))
        smp2 = read_image(os.path.join(DATASET_PATH, 'sample_2', image_name + '_s2.png'))
        smp3 = read_image(os.path.join(DATASET_PATH, 'sample_3', image_name + '_s3.png'))

        images_to_show = [
            org,
            exp,
            # smp1,
            # smp2,
            smp3
        ]
        show_images(images_to_show, image_name)


def plot_label_hist(dataset_path, cls):
    train_image_names, _ = split_train_val(dataset_path, n_val=0, seed=0)
    labels = np.array(read_labels(train_image_names))

    if cls == -1:
        plt.hist(labels.ravel(), bins=[1, 2, 3, 4, 5, 6], rwidth=0.9)
    else:
        plt.hist(labels[:, cls - 1], bins=[1, 2, 3, 4, 5, 6], rwidth=0.9)
    plt.title('score distribution')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    demo_show()
    # plot_label_hist(-1)
