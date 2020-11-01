import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import os

import matplotlib
matplotlib.use('TkAgg')


def read_image(filepath):
    """
    Read image by the given path
    """
    try:
        return cv2.imread(filepath)[..., ::-1].astype(np.uint8)
    except Exception as e:
        raise Exception(f'No such image file "{filepath}"')


def read_binary_mask(filepath):
    """
    Read binary mask by the given path
    """
    mask_image = read_image(filepath)
    mask = (mask_image[:, :, 0] / 255).astype(np.bool)
    return mask


def show_images(images, title=''):
    """
    Display several images at once
    """
    fig, axes = plt.subplots(ncols=len(images))
    if len(images) == 1:
        axes = [axes]
    for i in range(len(images)):
        axes[i].imshow(images[i])

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.title(title)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
    plt.close(fig)


def read_csv(filepath, field_names):
    """
    Read data from CSV file
    """
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, field_names)

        header = True
        for it in reader:
            if header:
                header = False
                continue
            data.append(it)

    return data


def read_labels(dataset_path, image_names):
    """
    Read ground truth label scores for required images from the CSV file
    """
    data = read_csv(os.path.join(dataset_path, 'OpenPart.csv'), ['Case', 'Sample 1', 'Sample 2', 'Sample 3'])

    labels = []
    for image_name in image_names:
        for it in data:
            if it['Case'].split('.')[0] == image_name:
                labels.append(tuple(map(int, (it['Sample 1'], it['Sample 2'], it['Sample 3']))))
    return labels


def compute_l1_errors(labels, preds):
    """
    Compute mean per-sample-type L1 errors and total L1 error
    """
    assert len(labels) == len(preds)

    preds = np.round(preds)
    errors = np.abs(labels - preds).mean(0)

    return errors


class seeded_random:
    """
    Creates a context manager with fixed numpy random state defined by the seed
    """
    def __init__(self, seed):
        """
        :param seed: seed to use within the context; if None, random state is the same as outside of the context
        """
        self.seed = seed

    def __enter__(self):
        if self.seed is None:
            return
        self.after_seed = np.random.randint(1 << 30)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed is None:
            return
        np.random.seed(self.after_seed)


def split_train_val(dataset_path, n_val, seed):
    """
    Randomly split data into train and validation parts according to the given seed
    """

    data = read_csv(os.path.join(dataset_path, 'OpenPart.csv'), ['Case', 'Sample 1', 'Sample 2', 'Sample 3'])
    image_names = list(map(lambda it: it['Case'].split('.')[0], data))

    with seeded_random(seed):
        prm = np.random.permutation(len(image_names))

    val_image_names = []
    train_image_names = []
    n_train = len(image_names) - n_val
    for i in range(n_train):
        train_image_names.append(image_names[prm[i]])
    for i in range(n_train, n_train + n_val):
        val_image_names.append(image_names[prm[i]])

    return train_image_names, val_image_names


def write_predictions(path, image_names, predictions):
    fieldnames = ['Case', 'Sample 1', 'Sample 2', 'Sample 3']
    with open('NibbersSubmission.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        for image_name, pred in zip(image_names, predictions):
            writer.writerow(
                {'Case': image_name + '.png',
                 'Sample 1': '{:.2f}'.format(pred[0]),
                 'Sample 2': '{:.2f}'.format(pred[1]),
                 'Sample 3': '{:.2f}'.format(pred[2])
            })
