import os
import pickle
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


def demo_plot_feature_importances():
    """
    Plot dumped feature improtances for demonstration in presentation
    """
    with open('f_imps.pkl', 'rb') as f:
        feature_importances = pickle.load(f)
    feature_names = ['emb', 'tn', 'fp', 'fn', 'tp', 'precision', 'recall', 'iou', 'fpr', 'fnr', 'f1', 'a_exp', 'a_smp',
                     'fp_edt', 'fn_edt', 's1_ind', 's2_ind', 's3_ind']

    f = np.array(feature_importances).mean(0).tolist()
    f = sorted(zip(f, feature_names), key=lambda it: -it[0])
    f, feature_names = tuple(zip(*f))

    plt.bar(feature_names, f)
    plt.xticks(rotation=-45)
    plt.title('Feature importances')
    plt.grid()
    plt.show()


def demo_plot_results():
    # 10
    # train mean: 0.2968 [0.2544 0.3912 0.2448]
    # train std: 0.07777163150318159 [0.04215021 0.03963029 0.03721505]
    # val mean: 0.5693333333333334 [0.504 0.696 0.508]
    # val std: 0.21846332008422426 [0.1843475  0.21813757 0.1937421 ]
    #
    # 20
    # train mean: 0.3081481481481481 [0.26755556 0.40711111 0.24977778]
    # train std: 0.08846814001086588 [0.05047136 0.06432019 0.04412328]
    # val mean: 0.5982222222222222 [0.55466667 0.71733333 0.52266667]
    # val std: 0.17246735243031064 [0.1516956  0.17514946 0.11727271]
    #
    # 30
    # train mean: 0.3076666666666667 [0.253 0.414 0.256]
    # train std: 0.09407207638590505 [0.04760252 0.06894926 0.05063596]
    # val mean: 0.6346666666666666 [0.596 0.768 0.54 ]
    # val std: 0.1762523443500508 [0.14207041 0.1821428  0.1077033 ]
    val_sizes = np.array([10, 20, 30])
    l1_errors_total = [0.569, 0.598, 0.634]
    l1_errors = np.array([[0.504, 0.696, 0.508],
                          [0.555, 0.717, 0.523],
                          [0.596, 0.768, 0.54]]).T

    plt.bar(val_sizes - 1.5, l1_errors_total, label='total')
    plt.bar(val_sizes - 0.5, l1_errors[0], label='sample_1')
    plt.bar(val_sizes + 0.5, l1_errors[1], label='sample_2')
    plt.bar(val_sizes + 1.5, l1_errors[2], label='sample_3')
    plt.xticks(val_sizes, val_sizes)
    plt.title('Mean L1 errors for 20 evaluations')
    plt.xlabel('Validation set size')
    plt.ylabel('Mean L1 error')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # demo_show()
    # plot_label_hist(-1)
    # demo_plot_feature_importances()
    demo_plot_results()