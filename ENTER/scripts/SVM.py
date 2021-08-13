import csv
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import itertools
from scipy import stats, random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def svm_model(case, index):
    # Data initialization
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    feat_case = case
    set = int(feat_case.split('_')[2][0])
    directory = 'svm_' + case.replace('.', '')
    LABELS = ['Inactive', 'Active', 'Walking', 'Driving']
    if not os.path.exists(directory):
        os.makedirs(directory)
    case = "svm_" + case

    if set == 2:
        columns = ['id', 'user', 'timestamp', 'acc_xs_mean', 'acc_ys_mean', 'acc_zs_mean', 'acc_xs_var',
                   'acc_ys_var', 'acc_zs_var', 'acc_xs_mad', 'acc_ys_mad', 'acc_zs_mad', 'acc_xs_max',
                   'acc_ys_max', 'acc_zs_max', 'acc_xs_min', 'acc_ys_min', 'acc_zs_min', 'acc_xs_iqr',
                   'acc_ys_iqr', 'acc_zs_iqr', 'gyro_xs_mean', 'gyro_ys_mean', 'gyro_zs_mean', 'gyro_xs_var',
                   'gyro_ys_var', 'gyro_zs_var', 'gyro_xs_mad', 'gyro_ys_mad', 'gyro_zs_mad', 'gyro_xs_max',
                   'gyro_ys_max', 'gyro_zs_max', 'gyro_xs_min', 'gyro_ys_min', 'gyro_zs_min', 'gyro_xs_iqr',
                   'gyro_ys_iqr', 'gyro_zs_iqr', 'magn_xs_mean', 'magn_ys_mean', 'magn_zs_mean', 'magn_xs_var',
                   'magn_ys_var', 'magn_zs_var', 'magn_xs_mad', 'magn_ys_mad', 'magn_zs_mad', 'magn_xs_max',
                   'magn_ys_max', 'magn_zs_max', 'magn_xs_min', 'magn_ys_min', 'magn_zs_min', 'magn_xs_iqr',
                   'magn_ys_iqr', 'magn_zs_iqr', 'gps_lat_mean', 'gps_long_mean', 'gps_alt_mean',
                   'gps_speed_mean', 'gps_bearing_mean', 'gps_accuracy_mean', 'gps_lat_var', 'gps_long_var',
                   'gps_alt_var', 'gps_speed_var', 'gps_bearing_var', 'gps_accuracy_var', 'gps_lat_mad',
                   'gps_long_mad', 'gps_alt_mad', 'gps_speed_mad', 'gps_bearing_mad', 'gps_accuracy_mad',
                   'gps_lat_max', 'gps_long_max', 'gps_alt_max', 'gps_speed_max', 'gps_bearing_max',
                   'gps_accuracy_max', 'gps_lat_min', 'gps_long_min', 'gps_alt_min', 'gps_speed_min',
                   'gps_bearing_min', 'gps_accuracy_min', 'gps_lat_iqr', 'gps_long_iqr', 'gps_alt_iqr',
                   'gps_speed_iqr', 'gps_bearing_iqr', 'gps_accuracy_iqr', 'activity_id', 'activity']
    else:
        if set == 1:
            columns = ['id', 'user', 'timestamp', 'acc_xs_mean', 'acc_ys_mean', 'acc_zs_mean', 'acc_xs_var',
                       'acc_ys_var', 'acc_zs_var', 'acc_xs_mad', 'acc_ys_mad', 'acc_zs_mad', 'acc_xs_max',
                       'acc_ys_max', 'acc_zs_max', 'acc_xs_min', 'acc_ys_min', 'acc_zs_min', 'acc_xs_iqr',
                       'acc_ys_iqr', 'acc_zs_iqr', 'magn_xs_mean', 'magn_ys_mean', 'magn_zs_mean',
                       'magn_xs_var', 'magn_ys_var', 'magn_zs_var', 'magn_xs_mad', 'magn_ys_mad', 'magn_zs_mad',
                       'magn_xs_max', 'magn_ys_max', 'magn_zs_max', 'magn_xs_min', 'magn_ys_min', 'magn_zs_min',
                       'magn_xs_iqr', 'magn_ys_iqr', 'magn_zs_iqr', 'gps_lat_mean', 'gps_long_mean',
                       'gps_alt_mean', 'gps_speed_mean', 'gps_bearing_mean', 'gps_accuracy_mean', 'gps_lat_var',
                       'gps_long_var', 'gps_alt_var', 'gps_speed_var', 'gps_bearing_var', 'gps_accuracy_var',
                       'gps_lat_mad', 'gps_long_mad', 'gps_alt_mad', 'gps_speed_mad', 'gps_bearing_mad',
                       'gps_accuracy_mad', 'gps_lat_max', 'gps_long_max', 'gps_alt_max', 'gps_speed_max',
                       'gps_bearing_max', 'gps_accuracy_max', 'gps_lat_min', 'gps_long_min', 'gps_alt_min',
                       'gps_speed_min', 'gps_bearing_min', 'gps_accuracy_min', 'gps_lat_iqr', 'gps_long_iqr',
                       'gps_alt_iqr', 'gps_speed_iqr', 'gps_bearing_iqr', 'gps_accuracy_iqr', 'activity_id',
                       'activity']
        else:
            if set == 0:
                columns = ['id', 'user', 'timestamp', 'acc_xs_mean', 'acc_ys_mean', 'acc_zs_mean', 'acc_xs_var',
                           'acc_ys_var', 'acc_zs_var', 'acc_xs_mad', 'acc_ys_mad', 'acc_zs_mad', 'acc_xs_max',
                           'acc_ys_max', 'acc_zs_max', 'acc_xs_min', 'acc_ys_min', 'acc_zs_min', 'acc_xs_iqr',
                           'acc_ys_iqr', 'acc_zs_iqr', 'gps_lat_mean', 'gps_long_mean', 'gps_alt_mean',
                           'gps_speed_mean', 'gps_bearing_mean', 'gps_accuracy_mean', 'gps_lat_var',
                           'gps_long_var', 'gps_alt_var', 'gps_speed_var', 'gps_bearing_var',
                           'gps_accuracy_var', 'gps_lat_mad', 'gps_long_mad', 'gps_alt_mad', 'gps_speed_mad',
                           'gps_bearing_mad', 'gps_accuracy_mad', 'gps_lat_max', 'gps_long_max', 'gps_alt_max',
                           'gps_speed_max', 'gps_bearing_max', 'gps_accuracy_max', 'gps_lat_min',
                           'gps_long_min', 'gps_alt_min', 'gps_speed_min', 'gps_bearing_min',
                           'gps_accuracy_min', 'gps_lat_iqr', 'gps_long_iqr', 'gps_alt_iqr', 'gps_speed_iqr',
                           'gps_bearing_iqr', 'gps_accuracy_iqr', 'activity_id', 'activity']

    df = pd.read_csv('./sensoringData_feature_prepared_' + feat_case + '.csv', header=0, names=columns)
    df.head()

    # Data gathering
    segments = []
    labels = []
    for i in range(0, len(df), 1):
        label = stats.mode(df['activity'][i])[0][0]

        acc_xs_mean = df['acc_xs_mean'].values[i]
        acc_ys_mean = df['acc_ys_mean'].values[i]
        acc_zs_mean = df['acc_zs_mean'].values[i]
        acc_xs_var = df['acc_xs_var'].values[i]
        acc_ys_var = df['acc_ys_var'].values[i]
        acc_zs_var = df['acc_zs_var'].values[i]
        acc_xs_mad = df['acc_xs_mad'].values[i]
        acc_ys_mad = df['acc_ys_mad'].values[i]
        acc_zs_mad = df['acc_zs_mad'].values[i]
        acc_xs_max = df['acc_xs_max'].values[i]
        acc_ys_max = df['acc_ys_max'].values[i]
        acc_zs_max = df['acc_zs_max'].values[i]
        acc_xs_min = df['acc_xs_min'].values[i]
        acc_ys_min = df['acc_ys_min'].values[i]
        acc_zs_min = df['acc_zs_min'].values[i]
        acc_xs_iqr = df['acc_xs_iqr'].values[i]
        acc_ys_iqr = df['acc_ys_iqr'].values[i]
        acc_zs_iqr = df['acc_zs_iqr'].values[i]

        if set != 0 and set != 1:
            gyro_xs_mean = df['gyro_xs_mean'].values[i]
            gyro_ys_mean = df['gyro_ys_mean'].values[i]
            gyro_zs_mean = df['gyro_zs_mean'].values[i]
            gyro_xs_var = df['gyro_xs_var'].values[i]
            gyro_ys_var = df['gyro_ys_var'].values[i]
            gyro_zs_var = df['gyro_zs_var'].values[i]
            gyro_xs_mad = df['gyro_xs_mad'].values[i]
            gyro_ys_mad = df['gyro_ys_mad'].values[i]
            gyro_zs_mad = df['gyro_zs_mad'].values[i]
            gyro_xs_max = df['gyro_xs_max'].values[i]
            gyro_ys_max = df['gyro_ys_max'].values[i]
            gyro_zs_max = df['gyro_zs_max'].values[i]
            gyro_xs_min = df['gyro_xs_min'].values[i]
            gyro_ys_min = df['gyro_ys_min'].values[i]
            gyro_zs_min = df['gyro_zs_min'].values[i]
            gyro_xs_iqr = df['gyro_xs_iqr'].values[i]
            gyro_ys_iqr = df['gyro_ys_iqr'].values[i]
            gyro_zs_iqr = df['gyro_zs_iqr'].values[i]

        if set != 0:
            magn_xs_mean = df['magn_xs_mean'].values[i]
            magn_ys_mean = df['magn_ys_mean'].values[i]
            magn_zs_mean = df['magn_zs_mean'].values[i]
            magn_xs_var = df['magn_xs_var'].values[i]
            magn_ys_var = df['magn_ys_var'].values[i]
            magn_zs_var = df['magn_zs_var'].values[i]
            magn_xs_mad = df['magn_xs_mad'].values[i]
            magn_ys_mad = df['magn_ys_mad'].values[i]
            magn_zs_mad = df['magn_zs_mad'].values[i]
            magn_xs_max = df['magn_xs_max'].values[i]
            magn_ys_max = df['magn_ys_max'].values[i]
            magn_zs_max = df['magn_zs_max'].values[i]
            magn_xs_min = df['magn_xs_min'].values[i]
            magn_ys_min = df['magn_ys_min'].values[i]
            magn_zs_min = df['magn_zs_min'].values[i]
            magn_xs_iqr = df['magn_xs_iqr'].values[i]
            magn_ys_iqr = df['magn_ys_iqr'].values[i]
            magn_zs_iqr = df['magn_zs_iqr'].values[i]

        gps_lat_mean = df['gps_lat_mean'].values[i]
        gps_long_mean = df['gps_long_mean'].values[i]
        gps_alt_mean = df['gps_alt_mean'].values[i]
        gps_speed_mean = df['gps_speed_mean'].values[i]
        gps_bearing_mean = df['gps_bearing_mean'].values[i]
        gps_accuracy_mean = df['gps_accuracy_mean'].values[i]
        gps_lat_var = df['gps_lat_var'].values[i]
        gps_long_var = df['gps_long_var'].values[i]
        gps_alt_var = df['gps_alt_var'].values[i]
        gps_speed_var = df['gps_speed_var'].values[i]
        gps_bearing_var = df['gps_bearing_var'].values[i]
        gps_accuracy_var = df['gps_accuracy_var'].values[i]
        gps_lat_mad = df['gps_lat_mad'].values[i]
        gps_long_mad = df['gps_long_mad'].values[i]
        gps_alt_mad = df['gps_alt_mad'].values[i]
        gps_speed_mad = df['gps_speed_mad'].values[i]
        gps_bearing_mad = df['gps_bearing_mad'].values[i]
        gps_accuracy_mad = df['gps_accuracy_mad'].values[i]
        gps_lat_max = df['gps_lat_max'].values[i]
        gps_long_max = df['gps_long_max'].values[i]
        gps_alt_max = df['gps_alt_max'].values[i]
        gps_speed_max = df['gps_speed_max'].values[i]
        gps_bearing_max = df['gps_bearing_max'].values[i]
        gps_accuracy_max = df['gps_accuracy_max'].values[i]
        gps_lat_min = df['gps_lat_min'].values[i]
        gps_long_min = df['gps_long_min'].values[i]
        gps_alt_min = df['gps_alt_min'].values[i]
        gps_speed_min = df['gps_speed_min'].values[i]
        gps_bearing_min = df['gps_bearing_min'].values[i]
        gps_accuracy_min = df['gps_accuracy_min'].values[i]
        gps_lat_iqr = df['gps_lat_iqr'].values[i]
        gps_long_iqr = df['gps_long_iqr'].values[i]
        gps_alt_iqr = df['gps_alt_iqr'].values[i]
        gps_speed_iqr = df['gps_speed_iqr'].values[i]
        gps_bearing_iqr = df['gps_bearing_iqr'].values[i]
        gps_accuracy_iqr = df['gps_accuracy_iqr'].values[i]

        if set == 2:
            segments.append(
                [acc_xs_mean, acc_ys_mean, acc_zs_mean, acc_xs_var, acc_ys_var, acc_zs_var, acc_xs_mad,
                 acc_ys_mad, acc_zs_mad, acc_xs_max, acc_ys_max, acc_zs_max, acc_xs_min, acc_ys_min, acc_zs_min,
                 acc_xs_iqr, acc_ys_iqr, acc_zs_iqr, gyro_xs_mean, gyro_ys_mean, gyro_zs_mean, gyro_xs_var,
                 gyro_ys_var, gyro_zs_var, gyro_xs_mad, gyro_ys_mad, gyro_zs_mad, gyro_xs_max, gyro_ys_max,
                 gyro_zs_max, gyro_xs_min, gyro_ys_min, gyro_zs_min, gyro_xs_iqr, gyro_ys_iqr, gyro_zs_iqr,
                 magn_xs_mean, magn_ys_mean, magn_zs_mean, magn_xs_var, magn_ys_var, magn_zs_var, magn_xs_mad,
                 magn_ys_mad, magn_zs_mad, magn_xs_max, magn_ys_max, magn_zs_max, magn_xs_min, magn_ys_min,
                 magn_zs_min, magn_xs_iqr, magn_ys_iqr, magn_zs_iqr, gps_lat_mean, gps_long_mean, gps_alt_mean,
                 gps_speed_mean, gps_bearing_mean, gps_accuracy_mean, gps_lat_var, gps_long_var, gps_alt_var,
                 gps_speed_var, gps_bearing_var, gps_accuracy_var, gps_lat_mad, gps_long_mad, gps_alt_mad,
                 gps_speed_mad, gps_bearing_mad, gps_accuracy_mad, gps_lat_max, gps_long_max, gps_alt_max,
                 gps_speed_max, gps_bearing_max, gps_accuracy_max, gps_lat_min, gps_long_min, gps_alt_min,
                 gps_speed_min, gps_bearing_min, gps_accuracy_min, gps_lat_iqr, gps_long_iqr, gps_alt_iqr,
                 gps_speed_iqr, gps_bearing_iqr, gps_accuracy_iqr])
        else:
            if set == 1:
                segments.append(
                    [acc_xs_mean, acc_ys_mean, acc_zs_mean, acc_xs_var, acc_ys_var, acc_zs_var, acc_xs_mad,
                     acc_ys_mad, acc_zs_mad, acc_xs_max, acc_ys_max, acc_zs_max, acc_xs_min, acc_ys_min,
                     acc_zs_min, acc_xs_iqr, acc_ys_iqr, acc_zs_iqr, magn_xs_mean, magn_ys_mean, magn_zs_mean,
                     magn_xs_var, magn_ys_var, magn_zs_var, magn_xs_mad, magn_ys_mad, magn_zs_mad, magn_xs_max,
                     magn_ys_max, magn_zs_max, magn_xs_min, magn_ys_min, magn_zs_min, magn_xs_iqr, magn_ys_iqr,
                     magn_zs_iqr, gps_lat_mean, gps_long_mean, gps_alt_mean, gps_speed_mean, gps_bearing_mean,
                     gps_accuracy_mean, gps_lat_var, gps_long_var, gps_alt_var, gps_speed_var, gps_bearing_var,
                     gps_accuracy_var, gps_lat_mad, gps_long_mad, gps_alt_mad, gps_speed_mad, gps_bearing_mad,
                     gps_accuracy_mad, gps_lat_max, gps_long_max, gps_alt_max, gps_speed_max, gps_bearing_max,
                     gps_accuracy_max, gps_lat_min, gps_long_min, gps_alt_min, gps_speed_min, gps_bearing_min,
                     gps_accuracy_min, gps_lat_iqr, gps_long_iqr, gps_alt_iqr, gps_speed_iqr, gps_bearing_iqr,
                     gps_accuracy_iqr])
            else:
                if set == 0:
                    segments.append(
                        [acc_xs_mean, acc_ys_mean, acc_zs_mean, acc_xs_var, acc_ys_var, acc_zs_var, acc_xs_mad,
                         acc_ys_mad, acc_zs_mad, acc_xs_max, acc_ys_max, acc_zs_max, acc_xs_min, acc_ys_min,
                         acc_zs_min, acc_xs_iqr, acc_ys_iqr, acc_zs_iqr, gps_lat_mean, gps_long_mean,
                         gps_alt_mean, gps_speed_mean, gps_bearing_mean, gps_accuracy_mean, gps_lat_var,
                         gps_long_var, gps_alt_var, gps_speed_var, gps_bearing_var, gps_accuracy_var,
                         gps_lat_mad, gps_long_mad, gps_alt_mad, gps_speed_mad, gps_bearing_mad,
                         gps_accuracy_mad, gps_lat_max, gps_long_max, gps_alt_max, gps_speed_max,
                         gps_bearing_max, gps_accuracy_max, gps_lat_min, gps_long_min, gps_alt_min,
                         gps_speed_min, gps_bearing_min, gps_accuracy_min, gps_lat_iqr, gps_long_iqr,
                         gps_alt_iqr, gps_speed_iqr, gps_bearing_iqr, gps_accuracy_iqr])
        labels.append(label)

    segments = np.asarray(segments, dtype=np.float32)
    for i in range(0, len(labels), 1):
        if labels[i] == "Inactive":
            labels[i] = 0
        else:
            if labels[i] == "Active":
                labels[i] = 1
            else:
                if labels[i] == "Walking":
                    labels[i] = 2
                else:
                    if labels[i] == "Driving":
                        labels[i] = 3
    labels = np.asarray(labels, dtype=np.int)

    n_splits = 10

    if index > -1:
        split_fold = index - 1

        kf = StratifiedKFold(n_splits=n_splits)
        train_idx, test_idx = list(kf.split(segments, labels))[split_fold]

        train_x = segments[train_idx]
        train_y = labels[train_idx]
        test_x = segments[test_idx]
        test_y = labels[test_idx]

        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        kf_2 = StratifiedKFold(n_splits=n_splits)
        train_idx_2, test_idx_2 = list(kf_2.split(train_x, train_y))[split_fold]

        train_split_x = train_x[train_idx_2]
        train_split_y = train_y[train_idx_2]
        test_split_x = train_x[test_idx_2]
        test_split_y = train_y[test_idx_2]

        Cs = [1, 10, 100, 1000, 10000]
        gammas = [0.0001, 0.001, 0.01, 0.1, 1]
        degrees = [1, 2, 3, 4]
        lin_comb = list(itertools.product(['linear'], Cs))
        rbf_comb = list(itertools.product(['rbf'], Cs, gammas))
        poly_comb = list(itertools.product(['poly'], Cs, gammas, degrees))
        models = []
        f1_scores = []
        combinations = []
        for element in lin_comb:
            combinations.append(element)
        for element in rbf_comb:
            combinations.append(element)
        for element in poly_comb:
            combinations.append(element)

        print("# Tuning hyper-parameters for: " + str(split_fold))
        print()

        for (kernel, C) in lin_comb:
            model = OneVsRestClassifier(SVC(max_iter=1000, kernel=kernel, C=C))
            model.fit(train_split_x, train_split_y)

            models.append(model)
            y_true, y_pred = test_split_y, model.predict(test_split_x)

            f1 = f1_score(y_true, y_pred, average='weighted')
            f1_scores.append(f1)
            print (str(f1) + ' | (' + str(kernel) + ', ' + str(C) + ')')

        for (kernel, C, gamma) in rbf_comb:
            model = OneVsRestClassifier(SVC(max_iter=1000, kernel=kernel, C=C, gamma=gamma))
            model.fit(train_split_x, train_split_y)

            models.append(model)
            y_true, y_pred = test_split_y, model.predict(test_split_x)

            f1 = f1_score(y_true, y_pred, average='weighted')
            f1_scores.append(f1)
            print (str(f1) + ' | (' + str(kernel) + ', ' + str(C) + ', ' + str(gamma) + ')')

        for (kernel, C, gamma, degree) in poly_comb:
            model = OneVsRestClassifier(SVC(max_iter=1000, kernel=kernel, C=C, gamma=gamma, degree=degree))
            model.fit(train_split_x, train_split_y)

            models.append(model)
            y_true, y_pred = test_split_y, model.predict(test_split_x)

            f1 = f1_score(y_true, y_pred, average='weighted')
            f1_scores.append(f1)
            print (str(f1) + ' | (' + str(kernel) + ', ' + str(C) + ', ' + str(gamma) + ', ' + str(degree) + ')')

        mf_index = f1_scores.index(max(f1_scores))
        print()
        print("# Best hyper-parameter combination: " + str(combinations[mf_index]))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_y, models[mf_index].predict(test_x)
        print(classification_report(y_true, y_pred))
        print()

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(data=confusion_matrix.astype(float))
        df_cm.to_csv(directory + '/confusion_matrix_' + str(split_fold) + "_" + case.replace('.', '') + '.csv',
                     sep=',', header=True, float_format='%.2f', index=False)
        plt.figure(figsize=(16, 14))
        sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
        plt.title("Confusion matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(directory + '/confusion_matrix_' + str(split_fold) + "_" + case.replace('.', ''))

        linear_scores = open(directory + '/linear_scores_' + str(split_fold) + '.csv', 'w', newline="")
        linear_scores_writer = csv.writer(linear_scores, delimiter=",")
        linear_scores_writer.writerow(f1_scores[0:5])
        rbf_scores = open(directory + '/rbf_scores_' + str(split_fold) + '.csv', 'w', newline="")
        rbf_scores_writer = csv.writer(rbf_scores, delimiter=",")
        rbf_scores_writer.writerow(f1_scores[5:30])
        poly_scores = open(directory + '/poly_scores_' + str(split_fold) + '.csv', 'w', newline="")
        poly_scores_writer = csv.writer(poly_scores, delimiter=",")
        poly_scores_writer.writerow(f1_scores[30:130])

        linear_scores.close()
        rbf_scores.close()
        poly_scores.close()


def set_random_seed(seed_arg):
    seed = int(seed_arg)
    random.seed(seed)
    np.random.seed(seed)
    # tf.set_random_seed(seed)
    return seed


if __name__ == '__main__':
    set_random_seed(6) # Favourite number
    if sys.argv[2]:
        index = int(sys.argv[2])
    else:
        index = -1
    startTime = time.time()
    svm_model(sys.argv[1], index)
    elapsedTime = time.time() - startTime
    print(elapsedTime)