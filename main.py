import os
import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
from tqdm import tqdm
import edt
import argparse

from utils import read_labels, compute_l1_errors, show_images, read_binary_mask, split_train_val, \
    read_csv, write_predictions


def baseline_model(train_image_names, val_image_names, **kwargs):
    """
    Trivial baseline model that predicts mean per-sample-type scores
    """
    train_labels = np.array(read_labels(train_image_names))

    mean_label = np.mean(train_labels, 0)

    val_preds = np.repeat(mean_label[np.newaxis, :], len(val_image_names), 0)
    train_preds = np.repeat(mean_label[np.newaxis, :], len(train_image_names), 0)

    return train_preds, val_preds


def main_model(train_image_names, val_image_names, regressor_type=None, verbose=False,
               feature_importances=None, **kwargs):
    def extract_features_from_df(df):
        """
        Extract features from data frame and assemble them into a 2D matrix
        :param df: input data frame
        :return: data matrix with the shape [n_samples, n_features]
        """
        X_list = []
        for i in range(1, 3 + 1):
            X_list.append([
                df[f'emb{i}'].to_numpy()[np.newaxis, :],
                # np.array(df[f'emb{i}'].values.tolist()).T,
                np.array(df[f'cm{i}'].values.tolist()).T,
                df[f'pr{i}'].to_numpy()[np.newaxis, :],
                df[f'rc{i}'].to_numpy()[np.newaxis, :],
                df[f'iou{i}'].to_numpy()[np.newaxis, :],
                df[f'fpr{i}'].to_numpy()[np.newaxis, :],
                df[f'fnr{i}'].to_numpy()[np.newaxis, :],
                df[f'f1{i}'].to_numpy()[np.newaxis, :],
                df[f'a_exp{i}'].to_numpy()[np.newaxis, :],
                df[f'a_smp{i}'].to_numpy()[np.newaxis, :],
                df[f'fp_edt{i}'].to_numpy()[np.newaxis, :],
                df[f'fn_edt{i}'].to_numpy()[np.newaxis, :],
                # df[f'i_prf{i}'].to_numpy()[np.newaxis, :],
                # df[f'i_suf{i}'].to_numpy()[np.newaxis, :],
            ])

        X = None
        for x_l in X_list:
            x = np.vstack(x_l)[:, np.newaxis, :]
            X = x if X is None else np.append(X, x, 1)
        X = X.astype(np.float32).transpose([2, 1, 0])       # [n_samples, 3, n_features]

        # add one-hot encoded sample-type feature
        X = np.append(X, np.zeros((*X.shape[:2], X.shape[1])), -1)
        for i in range(X.shape[1]):
            X[:, i, -X.shape[1] + i] = 1

        X = X.reshape((-1, X.shape[-1]))    # [n_samples * 3, n_features + 3]

        return X

    with open('vectors.pickle', 'rb') as f:
        vectors = pickle.load(f)

    # process each image an compute its features
    data = []
    for image_name in train_image_names + val_image_names:
        # read expert binary mask
        mask_exp = read_binary_mask(os.path.join(kwargs['dataset_path'], 'Expert', f'{image_name}_expert.png'))
        # compute Euclidian Distance Transform (EDT) of the expert mask
        # EDT is a mapping where each empty pixel stores a distance value to the closest non-empty pixel
        # Note: if expert mask is empty, set its EDT at every point equal to 512, as if
        #   for every empty pixel its closest non-empty pixel is far away
        mask_exp_edt = edt.edt(np.logical_not(mask_exp), parallel=4) if mask_exp.any() else \
            np.ones_like(mask_exp) * 1024 / 2

        conf_mats = []
        ious = []
        precisions = []
        recalls = []
        fprs = []
        fnrs = []
        f1s = []
        expert_mask_area = []
        sample_mask_area = []
        image_prefs = []
        image_sufs = []
        fp_edts = []
        fn_edts = []
        embs = []
        for i in range(3):
            # read sample binary mask
            mask_s = read_binary_mask(os.path.join(kwargs['dataset_path'], f'sample_{i + 1}', f'{image_name}_s{i + 1}.png'))
            # combine the expert and sample masks and compute their flattened confusion matrix
            comb_mask = mask_exp * 2 + mask_s
            conf_mat = np.bincount(comb_mask.ravel(), minlength=4)

            # compute basic metrics from confusion matrix (we add 1 to denominator for stability)
            tn, fp, fn, tp = conf_mat       # True Positive, False Positive, False Negative, True Positive
            iou = tp / (tp + fp + fn + 1)   # IntersectionOverUnion
            precision = tp / (tp + fp + 1)  # precision
            recall = tp / (tp + fn + 1)     # recall
            fpr = fp / (fp + tn + 1)        # False Positive Rate
            fnr = fn / (fn + tp + 1)        # False Negative Rate
            f1 = tp / (tp+(fp + fn)/2+1)    # F1-score
            area_e = fn + tp                # Expert mask nonzero area
            area_s = fp + tn                # Sample mask nonzero area

            # compute sample mask EDT in the same fashion as for the expert mask
            mask_s_edt = edt.edt(np.logical_not(mask_s), parallel=4) if mask_s.any() else \
                np.ones_like(mask_s) * 1024 / 2
            # here for all non-zero mask pixels we compute distance to the closest GT positive pixel and take the mean
            #   - for TP pixels EDT values are 0, and for FP they are positive, hence the name
            #   - the less this metric, the more similar are the expert and sample masks
            fp_edt = (mask_exp_edt * mask_s).mean()
            # similarly for FN pixels
            fn_edt = (mask_s_edt * mask_exp).mean()

            conf_mats.append(conf_mat)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            fprs.append(fpr)
            fnrs.append(fnr)
            f1s.append(f1)
            fp_edts.append(fp_edt)
            fn_edts.append(fn_edt)
            expert_mask_area.append(area_e)
            sample_mask_area.append(area_s)

            # add dot product between auto-encoder latent vectors as additional feature
            embs.append(np.dot(vectors[image_name][0].ravel(), vectors[image_name][i].ravel()))

            # just in case ¯\_(ツ)_/¯
            image_prefs.append(int(image_name.split('_')[0]))
            image_sufs.append(int(image_name.split('_')[1]))

        row = [image_name]
        for i in range(3):
            row += [conf_mats[i], ious[i], precisions[i], recalls[i], fprs[i], fnrs[i], f1s[i],
                    expert_mask_area[i], sample_mask_area[i],
                    fp_edts[i], fn_edts[i], embs[i],
                    image_prefs[i], image_sufs[i]]
        data.append(row)

    # build a data frame out of computed values and split it into train and validation
    df = pd.DataFrame(data, columns=[
        'image_name',
        'cm1', 'iou1', 'pr1', 'rc1', 'fpr1', 'fnr1', 'f11', 'a_exp1', 'a_smp1', 'fp_edt1', 'fn_edt1', 'emb1', 'i_prf1', 'i_suf1',
        'cm2', 'iou2', 'pr2', 'rc2', 'fpr2', 'fnr2', 'f12', 'a_exp2', 'a_smp2', 'fp_edt2', 'fn_edt2', 'emb2', 'i_prf2', 'i_suf2',
        'cm3', 'iou3', 'pr3', 'rc3', 'fpr3', 'fnr3', 'f13', 'a_exp3', 'a_smp3', 'fp_edt3', 'fn_edt3', 'emb3', 'i_prf3', 'i_suf3',
    ])
    train_df = df[:len(train_image_names)]
    val_df = df[len(train_image_names):]

    # retrieve training labels
    Y = np.array(read_labels(kwargs['dataset_path'], train_image_names)).ravel()

    # extract features from the train data frame and normalize them
    X_t = extract_features_from_df(train_df)
    X_t_mean, X_t_std = X_t.mean(0), X_t.std(0)
    X_t = (X_t - X_t_mean) / X_t_std

    # extract features from the validation data frame and normalize thme
    X_v = extract_features_from_df(val_df)
    X_v = (X_v - X_t_mean) / X_t_std

    # create the selected regressor
    if regressor_type == 'MLP':
        reg = MLPRegressor(hidden_layer_sizes=(8, ),
                           random_state=0,
                           solver='sgd',
                           alpha=1e-3,
                           learning_rate='invscaling',
                           learning_rate_init=5e-3,
                           power_t=0.2,
                           max_iter=2000,
                           verbose=verbose,
                           )
    elif regressor_type == 'RF':
        reg = RandomForestRegressor(random_state=0,
                                    verbose=verbose,
                                    n_jobs=6,
                                    n_estimators=200,
                                    criterion='mae',
                                    # max_depth=3,
                                    bootstrap=True,
                                    max_samples=0.5,
                                    )
    elif regressor_type == 'BGG':
        reg = BaggingRegressor(random_state=0,
                               verbose=verbose,
                               n_jobs=4,
                               n_estimators=50,
                               max_samples=0.75
                               )
    elif regressor_type == 'AB':
        reg = AdaBoostRegressor(random_state=0,
                                loss='linear',
                                learning_rate=1e-2,
                                n_estimators=100
                                )
    else:
        raise Exception('Unknown regressor type')

    # fit the regressor
    reg = reg.fit(X_t, Y)

    # make predictions for validation and train datasets
    val_preds = reg.predict(X_v).reshape((-1, 3))
    train_preds = reg.predict(X_t).reshape((-1, 3))

    if hasattr(reg, 'feature_importances_') and feature_importances is not None:
        feature_importances.append(reg.feature_importances_.tolist())

    train_preds = np.round(train_preds)
    val_preds = np.round(val_preds)

    return train_preds, val_preds


def eval_model(model, train_image_names, val_image_names, **kwargs):
    """
    Evaluate model on a splitted dataset and compute train and validation L1 errors
    """
    train_preds, val_preds = model(train_image_names, val_image_names, **kwargs)

    train_labels = np.array(read_labels(train_image_names))
    train_errors = compute_l1_errors(train_labels, train_preds)

    val_labels = np.array(read_labels(val_image_names))
    val_errors = compute_l1_errors(val_labels, val_preds)

    return train_errors, val_errors


def multi_eval_model(dataset_path, model, val_size, n_repeats, **kwargs):
    """
    Perform several repeated experiments on different train-validation data splits and average resulting L1 errors
    :param model: model to evaluate
    :param val_size: validation set size
    :param n_repeats: number of evaluations to perform
    :return:
    """
    np.random.seed(0)

    total_train_errors, total_val_errors = [], []
    for _ in tqdm(range(n_repeats)):
        train_image_names, val_image_names = split_train_val(dataset_path, val_size, np.random.randint(1 << 30))
        train_errors, val_errors = eval_model(model, train_image_names, val_image_names,
                                              verbose=n_repeats == 1, **kwargs)
        total_train_errors.append(train_errors)
        total_val_errors.append(val_errors)
    total_train_errors, total_val_errors = np.array(total_train_errors), np.array(total_val_errors)

    # compute means and standard deviations for the errors
    mean_train_errors, mean_val_errors = total_train_errors.mean(0), total_val_errors.mean(0)
    std_train_errors, std_val_errors = total_train_errors.std(0), total_val_errors.std(0)
    total_train_error, total_val_error = mean_train_errors.mean(), mean_val_errors.mean()
    total_train_error_std, total_val_error_std = total_train_errors.std(), total_val_errors.std()

    print('train mean:', total_train_error, mean_train_errors)
    print('train std:', total_train_error_std, std_train_errors)

    print('val mean:', total_val_error, mean_val_errors)
    print('val std:', total_val_error_std, std_val_errors)


def make_final_prediction(**kwargs):
    """
    Create submission file
    """

    # read test image names
    test_image_names = []
    for r in read_csv(os.path.join(kwargs['dataset_path'], 'SecretPart_dummy.csv'), ['Case', 'Sample 1', 'Sample 2', 'Sample 3']):
        test_image_names.append(r['Case'].split('.')[0])

    # get the whole train dataset
    train_image_names, _ = split_train_val(kwargs['dataset_path'], 0, 0)

    # train and evaluate the model
    train_predictions, test_predictions = main_model(train_image_names, test_image_names, verbose=True, **kwargs)

    # compute train errors
    train_errors = compute_l1_errors(read_labels(kwargs['dataset_path'], train_image_names), train_predictions)
    print('train error', train_errors.mean(), train_errors)

    # write predictions
    write_predictions(kwargs['prediction_path'], test_image_names, test_predictions)


if __name__ == '__main__':
    # feature_importances = []
    # multi_eval_model(main_model, val_size=20, n_repeats=25, regressor_type='RF',
    #                  feature_importances=feature_importances)
    # with open('f_imps.pkl', 'wb') as f:
    #     pickle.dump(feature_importances, f)
    # make_final_prediction(regressor_type='RF')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset_path', type=str)

    args = parser.parse_args()
    make_final_prediction(dataset_path=args.dataset_path, prediction_path='./', regressor_type='RF')
