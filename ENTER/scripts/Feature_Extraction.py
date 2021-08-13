import csv
import time
import numpy as np
import pandas as pd
import sys
from scipy import stats
from astropy.stats import median_absolute_deviation


# This does the main work, which is computing all the features and preparing the data lists to be file-written in the
# next steps.
def get_features(id, xs, ys, zs, speeds, bearings, accuracies, aux_count, aux_count_ng, aux_count_ngnm, data_2, data_1,
                 data_0, user, timestamp, activity_id, activity, type):
    xs_mean = np.mean(xs)
    ys_mean = np.mean(ys)
    zs_mean = np.mean(zs)
    xs_var = np.var(xs)
    ys_var = np.var(ys)
    zs_var = np.var(zs)
    xs_mad = median_absolute_deviation(xs)
    ys_mad = median_absolute_deviation(ys)
    zs_mad = median_absolute_deviation(zs)
    xs_max = max(xs)
    zs_max = max(zs)
    ys_max = max(ys)
    xs_min = min(xs)
    ys_min = min(ys)
    zs_min = min(zs)
    xs_iqr = stats.iqr(xs, rng=(25, 75), interpolation='midpoint')
    ys_iqr = stats.iqr(ys, rng=(25, 75), interpolation='midpoint')
    zs_iqr = stats.iqr(zs, rng=(25, 75), interpolation='midpoint')

    if type != 'gps':
        if type == 'acc':
            aux_count += 1
            data_0.append(
                [id, user, timestamp, xs_mean, ys_mean, zs_mean, xs_var, ys_var, zs_var, xs_mad, ys_mad, zs_mad, xs_max,
                 ys_max, zs_max, xs_min, ys_min, zs_min, xs_iqr, ys_iqr, zs_iqr])
            if user != 14 and user != 18:
                aux_count_ng += 1
                data_1.append(
                    [id, user, timestamp, xs_mean, ys_mean, zs_mean, xs_var, ys_var, zs_var, xs_mad, ys_mad, zs_mad,
                     xs_max, ys_max, zs_max, xs_min, ys_min, zs_min, xs_iqr, ys_iqr, zs_iqr])
                if user != 4 and user != 15 and user != 17:
                    aux_count_ngnm += 1
                    data_2.append(
                        [id, user, timestamp, xs_mean, ys_mean, zs_mean, xs_var, ys_var, zs_var, xs_mad, ys_mad, zs_mad,
                         xs_max, ys_max, zs_max, xs_min, ys_min, zs_min, xs_iqr, ys_iqr, zs_iqr])
        else:
            aux_count += 1
            data_0.append(
                [xs_mean, ys_mean, zs_mean, xs_var, ys_var, zs_var, xs_mad, ys_mad, zs_mad, xs_max, ys_max, zs_max,
                 xs_min, ys_min, zs_min, xs_iqr, ys_iqr, zs_iqr])
            if user != 14 and user != 18:
                aux_count_ng += 1
                data_1.append(
                    [xs_mean, ys_mean, zs_mean, xs_var, ys_var, zs_var, xs_mad, ys_mad, zs_mad, xs_max, ys_max, zs_max,
                     xs_min, ys_min, zs_min, xs_iqr, ys_iqr, zs_iqr])
                if user != 4 and user != 15 and user != 17:
                    aux_count_ngnm += 1
                    data_2.append(
                        [xs_mean, ys_mean, zs_mean, xs_var, ys_var, zs_var, xs_mad, ys_mad, zs_mad, xs_max, ys_max,
                         zs_max, xs_min, ys_min, zs_min, xs_iqr, ys_iqr, zs_iqr])
    else:
        speeds_mean = np.mean(speeds)
        bearings_mean = np.mean(bearings)
        accuracies_mean = np.mean(accuracies)
        speeds_var = np.var(speeds)
        bearings_var = np.var(bearings)
        accuracies_var = np.var(accuracies)
        speeds_mad = median_absolute_deviation(speeds)
        bearings_mad = median_absolute_deviation(bearings)
        accuracies_mad = median_absolute_deviation(accuracies)
        speeds_max = max(speeds)
        bearings_max = max(bearings)
        accuracies_max = max(accuracies)
        speeds_min = min(speeds)
        bearings_min = min(bearings)
        accuracies_min = min(accuracies)
        speeds_iqr = stats.iqr(speeds, rng=(25, 75), interpolation='midpoint')
        bearings_iqr = stats.iqr(bearings, rng=(25, 75), interpolation='midpoint')
        accuracies_iqr = stats.iqr(accuracies, rng=(25, 75), interpolation='midpoint')

        aux_count += 1
        data_0.append(
            [xs_mean, ys_mean, zs_mean, speeds_mean, bearings_mean, accuracies_mean, xs_var, ys_var, zs_var, speeds_var,
             bearings_var, accuracies_var, xs_mad, ys_mad, zs_mad, speeds_mad, bearings_mad, accuracies_mad, xs_max,
             ys_max, zs_max, speeds_max, bearings_max, accuracies_max, xs_min, ys_min, zs_min, speeds_min, bearings_min,
             accuracies_min, xs_iqr, ys_iqr, zs_iqr, speeds_iqr, bearings_iqr, accuracies_iqr, activity_id, activity])
        if user != 14 and user != 18:
            aux_count_ng += 1
            data_1.append(
                [xs_mean, ys_mean, zs_mean, speeds_mean, bearings_mean, accuracies_mean, xs_var, ys_var, zs_var,
                 speeds_var, bearings_var, accuracies_var, xs_mad, ys_mad, zs_mad, speeds_mad, bearings_mad,
                 accuracies_mad, xs_max, ys_max, zs_max, speeds_max, bearings_max, accuracies_max, xs_min, ys_min,
                 zs_min, speeds_min, bearings_min, accuracies_min, xs_iqr, ys_iqr, zs_iqr, speeds_iqr, bearings_iqr,
                 accuracies_iqr, activity_id, activity])
            if user != 4 and user != 15 and user != 17:
                aux_count_ngnm += 1
                data_2.append(
                    [xs_mean, ys_mean, zs_mean, speeds_mean, bearings_mean, accuracies_mean, xs_var, ys_var, zs_var,
                     speeds_var, bearings_var, accuracies_var, xs_mad, ys_mad, zs_mad, speeds_mad, bearings_mad,
                     accuracies_mad, xs_max, ys_max, zs_max, speeds_max, bearings_max, accuracies_max, xs_min, ys_min,
                     zs_min, speeds_min, bearings_min, accuracies_min, xs_iqr, ys_iqr, zs_iqr, speeds_iqr, bearings_iqr,
                     accuracies_iqr, activity_id, activity])

    return data_2, data_1, data_0, aux_count, aux_count_ng, aux_count_ngnm


# All the logic regarding the correct application of each sliding window and feature computation.
def extraction(df, type, n_seconds, overlap, cut_seconds, val_sess_df):
    i = 1
    next_i = 0
    init = False
    init_out = False
    finish_window = False
    stop_count = False
    data_0 = []
    data_1 = []
    data_2 = []
    aux_count_ngnm = 0
    aux_count_ng = 0
    aux_count = 0
    aux_data_ngnm = []
    aux_data_ng = []
    aux_data = []
    xs = []
    ys = []
    zs = []
    speeds = []
    bearings = []
    accuracies = []
    timestamp_list = []
    x_value = ''
    y_value = ''
    z_value = ''
    s_value = ''
    b_value = ''
    a_value = ''
    timestamp = 0
    timestamp_beginning = 0
    timestamp_before = 0
    end_timestamp = 0
    user_beginning = ''
    next_time = 0
    if type == 'acc':
        x_value = 'acc_x_axis'
        y_value = 'acc_y_axis'
        z_value = 'acc_z_axis'
    else:
        if type == 'gyro':
            x_value = 'gyro_x_axis'
            y_value = 'gyro_y_axis'
            z_value = 'gyro_z_axis'
        else:
            if type == 'magn':
                x_value = 'magn_x_axis'
                y_value = 'magn_y_axis'
                z_value = 'magn_z_axis'
            else:
                if type == 'gps':
                    x_value = 'gps_lat_increment'
                    y_value = 'gps_long_increment'
                    z_value = 'gps_alt_increment'
                    s_value = 'gps_speed'
                    b_value = 'gps_bearing'
                    a_value = 'gps_accuracy'

    while i < len(df):
        i_before = i
        id = df['id'].values[i]
        user = df['user'].values[i]
        timestamp = float(df['timestamp'].values[i])
        activity_id = df['activity_id'].values[i]
        activity = df['activity'].values[i]
        sess_df = val_sess_df.loc[(val_sess_df['user'] == int(user)) & (val_sess_df['activity_id'] == int(activity_id))]
        if len(sess_df) > 0:
            invalid = False
        else:
            invalid = True

        if not init and not invalid:
            timestamp_beginning = sess_df['init_timestamp'].values[0] + cut_seconds
            end_timestamp = sess_df['end_timestamp'].values[0] - cut_seconds
            next_time = timestamp_beginning + (n_seconds - overlap)
            user_beginning = user
            activity_id_beginning = activity_id
            activity_beginning = activity
            init = True
            if i > 1:
                init_out = True

        if init_out or ((user_beginning != user or activity_id_beginning != activity_id
                         or activity_beginning != activity) and not invalid):
            if timestamp_before <= (timestamp_beginning + n_seconds) <= end_timestamp and len(xs) > 0:
                data_2, data_1, data_0, aux_count, aux_count_ng, aux_count_ngnm = get_features(id, xs, ys, zs, speeds,
                    bearings, accuracies, aux_count, aux_count_ng, aux_count_ngnm, data_2, data_1, data_0,
                    user_beginning, timestamp_before, activity_id_beginning, activity_beginning, type)

            if aux_count > 0:
                aux_data.append([activity_id_beginning, aux_count])
            if aux_count_ng > 0:
                aux_data_ng.append([activity_id_beginning, aux_count_ng])
            if aux_count_ngnm > 0:
                aux_data_ngnm.append([activity_id_beginning, aux_count_ngnm])
            aux_count = 0
            aux_count_ng = 0
            aux_count_ngnm = 0
            timestamp_beginning = sess_df['init_timestamp'].values[0] + cut_seconds
            end_timestamp = sess_df['end_timestamp'].values[0] - cut_seconds
            next_time = timestamp_beginning + (n_seconds - overlap)
            user_beginning = user
            activity_id_beginning = activity_id
            activity_beginning = activity
            xs = []
            ys = []
            zs = []
            speeds = []
            bearings = []
            accuracies = []
            timestamp_list = []
            init_out = False
            stop_count = False
            i += 1
        else:
            if timestamp >= next_time and not stop_count and not invalid:
                next_i = i
                stop_count = True
            if end_timestamp >= timestamp >= timestamp_beginning + n_seconds and not invalid:
                if not xs:
                    print("timestamp: " + str(timestamp))
                    print("timestamp_beginning: " + str(timestamp_beginning))
                    print("timestamp_before: " + str(timestamp_before))
                    print("end_timestamp: " + str(end_timestamp))
                    print("activity_id: " + str(activity_id))
                    print("user: " + str(user))
                    print("next_i: " + str(next_i))
                    print("next_time: " + str(next_time))
                    print("i: " + str(i))
                    print("type: " + str(type))
                data_2, data_1, data_0, aux_count, aux_count_ng, aux_count_ngnm = get_features(id, xs, ys, zs, speeds,
                    bearings, accuracies, aux_count, aux_count_ng, aux_count_ngnm, data_2, data_1, data_0, user,
                    timestamp, activity_id, activity, type)

                timestamp_beginning = next_time
                next_time = timestamp_beginning + (n_seconds - overlap)
                xs = []
                ys = []
                zs = []
                speeds = []
                bearings = []
                accuracies = []
                timestamp_list = []

                user = df['user'].values[i]
                timestamp = float(df['timestamp'].values[i])
                activity_id = df['activity_id'].values[i]
                activity = df['activity'].values[i]
                if next_i < i:
                    if (df['user'].values[next_i] == user) and (df['activity_id'].values[next_i] == activity_id) and (
                            df['activity'].values[next_i] == activity):
                        i = next_i
                    else:
                        i += 1
                else:
                    i += 1
                stop_count = False
                finish_window = True

            else:
                i += 1

        if not finish_window and not invalid and (end_timestamp >= timestamp >= timestamp_beginning):
            xs.append(df[x_value].values[i_before])
            ys.append(df[y_value].values[i_before])
            zs.append(df[z_value].values[i_before])
            if type == 'gps':
                speeds.append(df[s_value].values[i_before])
                bearings.append(df[b_value].values[i_before])
                accuracies.append(df[a_value].values[i_before])
            timestamp_list.append(float(df['timestamp'].values[i_before]))
        timestamp_before = timestamp
        finish_window = False

    if timestamp_before <= (timestamp_beginning + n_seconds) <= end_timestamp and not invalid and len(xs) > 0:
        data_2, data_1, data_0, aux_count, aux_count_ng, aux_count_ngnm = get_features(id, xs, ys, zs, speeds, bearings,
            accuracies, aux_count, aux_count_ng, aux_count_ngnm, data_2, data_1, data_0, user_beginning,
            timestamp_before, activity_id_beginning, activity_beginning, type)

    if aux_count > 0:
        aux_data.append([activity_id_beginning, aux_count])
    if aux_count_ng > 0:
        aux_data_ng.append([activity_id_beginning, aux_count_ng])
    if aux_count_ngnm > 0:
        aux_data_ngnm.append([activity_id_beginning, aux_count_ngnm])
    return data_2, data_1, data_0, aux_data_ngnm, aux_data_ng, aux_data


# Function to initialize and finish all the computation process. It does some calculus in the end to make sure that each
# sensor has the same number of computed windows, getting rid of possible limit situations.
def process_data_split(n_div, n_seconds, overlap, cut_seconds):
    acc_path = './sensoringData_acc_prepared_' + str(n_seconds) + '_' + str(n_div) + '.csv'
    gyro_path = './sensoringData_gyro_prepared_' + str(n_seconds) + '_' + str(n_div) + '.csv'
    magn_path = './sensoringData_magn_prepared_' + str(n_seconds) + '_' + str(n_div) + '.csv'
    gps_path = './sensoringData_gps_prepared_' + str(n_seconds) + '_' + str(n_div) + '.csv'
    val_sess_path = './validSessions_' + str(n_seconds) + '.csv'

    acc_columns = ['id', 'user', 'timestamp', 'acc_x_axis', 'acc_y_axis', 'acc_z_axis', 'activity_id', 'activity']
    gyro_columns = ['id', 'user', 'timestamp', 'gyro_x_axis', 'gyro_y_axis', 'gyro_z_axis', 'activity_id', 'activity']
    magn_columns = ['id', 'user', 'timestamp', 'magn_x_axis', 'magn_y_axis', 'magn_z_axis', 'activity_id', 'activity']
    gps_columns = ['id', 'user', 'timestamp', 'gps_lat_increment', 'gps_long_increment', 'gps_alt_increment',
                   'gps_speed', 'gps_bearing', 'gps_accuracy', 'activity_id', 'activity']
    val_sess_columns = ['user', 'activity_id', 'init_timestamp', 'end_timestamp']
    acc_df = pd.read_csv(acc_path, header=0, names=acc_columns)
    acc_df.head()
    gyro_df = pd.read_csv(gyro_path, header=0, names=gyro_columns)
    gyro_df.head()
    magn_df = pd.read_csv(magn_path, header=0, names=magn_columns)
    magn_df.head()
    gps_df = pd.read_csv(gps_path, header=0, names=gps_columns)
    gps_df.head()
    val_sess_df = pd.read_csv(val_sess_path, header=0, names=val_sess_columns)
    val_sess_df.head()

    data_0 = []
    data_1 = []
    data_2 = []

    results = []
    sensors = [[acc_df, 'acc'], [gyro_df, 'gyro'], [magn_df, 'magn'], [gps_df, 'gps']]
    for sensor in sensors:
        ext_results = extraction(sensor[0], sensor[1], n_seconds, overlap, cut_seconds, val_sess_df)
        results.append(ext_results)

    acc_data_2, acc_data_1, acc_data_0, aux_acc_ngnm, aux_acc_ng, aux_acc = results[0]
    gyro_data_2, gyro_data_1, gyro_data_0, aux_gyro_ngnm, aux_gyro_ng, aux_gyro = results[1]
    magn_data_2, magn_data_1, magn_data_0, aux_magn_ngnm, aux_magn_ng, aux_magn = results[2]
    gps_data_2, gps_data_1, gps_data_0, aux_gps_ngnm, aux_gps_ng, aux_gps = results[3]

    l12 = len(acc_data_2)
    l22 = len(gyro_data_2)
    l32 = len(magn_data_2)
    l42 = len(gps_data_2)
    l11 = len(acc_data_1)
    l31 = len(magn_data_1)
    l41 = len(gps_data_1)
    l10 = len(acc_data_0)
    l40 = len(gps_data_0)
    print(str(l12))
    print(str(l22))
    print(str(l32))
    print(str(l42))
    print(str(l11))
    print(str(l31))
    print(str(l41))
    print(str(l10))
    print(str(l40))

    if l12 != l22 or l12 != l32 or l12 != l42 or l22 != l32 or l22 != l42 or l32 != l42:
        z = 0
        acc_val = 0
        minLen = min(len(aux_acc_ngnm), len(aux_gyro_ngnm), len(aux_magn_ngnm), len(aux_gps_ngnm))
        while z < minLen:
            min_val = min(aux_acc_ngnm[z][1], aux_gyro_ngnm[z][1], aux_magn_ngnm[z][1], aux_gps_ngnm[z][1])
            if aux_acc_ngnm[z][1] != aux_gyro_ngnm[z][1] or aux_acc_ngnm[z][1] != aux_magn_ngnm[z][1] or \
                    aux_acc_ngnm[z][1] != aux_gps_ngnm[z][1]:
                if min_val == aux_acc_ngnm[z][1]:
                    gyro_del = aux_gyro_ngnm[z][1] - min_val
                    k = 0
                    while k < gyro_del:
                        gyro_data_2.pop(acc_val + min_val)
                        k += 1
                    magn_del = aux_magn_ngnm[z][1] - min_val
                    k = 0
                    while k < magn_del:
                        magn_data_2.pop(acc_val + min_val)
                        k += 1
                    gps_del = aux_gps_ngnm[z][1] - min_val
                    k = 0
                    while k < gps_del:
                        gps_data_2.pop(acc_val + min_val)
                        k += 1
                else:
                    if min_val == aux_gyro_ngnm[z][1]:
                        acc_del = aux_acc_ngnm[z][1] - min_val
                        k = 0
                        while k < acc_del:
                            acc_data_2.pop(acc_val + min_val)
                            k += 1
                        magn_del = aux_magn_ngnm[z][1] - min_val
                        k = 0
                        while k < magn_del:
                            magn_data_2.pop(acc_val + min_val)
                            k += 1
                        gps_del = aux_gps_ngnm[z][1] - min_val
                        k = 0
                        while k < gps_del:
                            gps_data_2.pop(acc_val + min_val)
                            k += 1
                    else:
                        if min_val == aux_magn_ngnm[z][1]:
                            acc_del = aux_acc_ngnm[z][1] - min_val
                            k = 0
                            while k < acc_del:
                                acc_data_2.pop(acc_val + min_val)
                                k += 1
                            gyro_del = aux_gyro_ngnm[z][1] - min_val
                            k = 0
                            while k < gyro_del:
                                gyro_data_2.pop(acc_val + min_val)
                                k += 1
                            gps_del = aux_gps_ngnm[z][1] - min_val
                            k = 0
                            while k < gps_del:
                                gps_data_2.pop(acc_val + min_val)
                                k += 1
                        else:
                            if min_val == aux_gps_ngnm[z][1]:
                                acc_del = aux_acc_ngnm[z][1] - min_val
                                k = 0
                                while k < acc_del:
                                    acc_data_2.pop(acc_val + min_val)
                                    k += 1
                                gyro_del = aux_gyro_ngnm[z][1] - min_val
                                k = 0
                                while k < gyro_del:
                                    gyro_data_2.pop(acc_val + min_val)
                                    k += 1
                                magn_del = aux_magn_ngnm[z][1] - min_val
                                k = 0
                                while k < magn_del:
                                    magn_data_2.pop(acc_val + min_val)
                                    k += 1
            acc_val += min_val
            z += 1

    if l11 != l31 or l11 != l41 or l31 != l41:
        z = 0
        acc_val = 0
        minLen = min(len(aux_acc_ng), len(aux_magn_ng), len(aux_gps_ng))
        while z < minLen:
            min_val = min(aux_acc_ng[z][1], aux_magn_ng[z][1], aux_gps_ng[z][1])
            if aux_acc_ng[z][1] != aux_magn_ng[z][1] or aux_acc_ng[z][1] != aux_gps_ng[z][1]:
                if min_val == aux_acc_ng[z][1]:
                    magn_del = aux_magn_ng[z][1] - min_val
                    k = 0
                    while k < magn_del:
                        magn_data_1.pop(acc_val + min_val)
                        k += 1
                    gps_del = aux_gps_ng[z][1] - min_val
                    k = 0
                    while k < gps_del:
                        gps_data_1.pop(acc_val + min_val)
                        k += 1
                else:
                    if min_val == aux_magn_ng[z][1]:
                        acc_del = aux_acc_ng[z][1] - min_val
                        k = 0
                        while k < acc_del:
                            acc_data_1.pop(acc_val + min_val)
                            k += 1
                        gps_del = aux_gps_ng[z][1] - min_val
                        k = 0
                        while k < gps_del:
                            gps_data_1.pop(acc_val + min_val)
                            k += 1
                    else:
                        if min_val == aux_gps_ng[z][1]:
                            acc_del = aux_acc_ng[z][1] - min_val
                            k = 0
                            while k < acc_del:
                                acc_data_1.pop(acc_val + min_val)
                                k += 1
                            magn_del = aux_magn_ng[z][1] - min_val
                            k = 0
                            while k < magn_del:
                                magn_data_1.pop(acc_val + min_val)
                                k += 1
            acc_val += min_val
            z += 1

    if l10 != l40:
        z = 0
        acc_val = 0
        minLen = min(len(aux_acc), len(aux_gps))
        while z < minLen:
            min_val = min(aux_acc[z][1], aux_gps[z][1])
            if aux_acc[z][1] != aux_gps[z][1]:
                if min_val == aux_acc[z][1]:
                    gps_del = aux_gps[z][1] - min_val
                    k = 0
                    while k < gps_del:
                        gps_data_0.pop(acc_val + min_val)
                        k += 1
                else:
                    if min_val == aux_gps[z][1]:
                        acc_del = aux_acc[z][1] - min_val
                        k = 0
                        while k < acc_del:
                            acc_data_0.pop(acc_val + min_val)
                            k += 1
            acc_val += min_val
            z += 1

    j = 0
    minDataLen = min(len(acc_data_2), len(gyro_data_2), len(magn_data_2), len(gps_data_2))
    while j < minDataLen:
        data_2.append(acc_data_2[j] + gyro_data_2[j] + magn_data_2[j] + gps_data_2[j])
        j += 1
    j = 0
    minDataLen = min(len(acc_data_1), len(magn_data_1), len(gps_data_1))
    while j < minDataLen:
        data_1.append(acc_data_1[j] + magn_data_1[j] + gps_data_1[j])
        j += 1
    j = 0
    minDataLen = min(len(acc_data_0), len(gps_data_0))
    while j < minDataLen:
        data_0.append(acc_data_0[j] + gps_data_0[j])
        j += 1

    filePath = './'
    fileName_0 = 'sensoringData_feature_prepared_' + str(n_seconds) + '_' + str(overlap) + '_0_split_' + str(n_div) \
                 + '.csv'
    fileName_1 = 'sensoringData_feature_prepared_' + str(n_seconds) + '_' + str(overlap) + '_1_split_' + str(n_div) \
                 + '.csv'
    fileName_2 = 'sensoringData_feature_prepared_' + str(n_seconds) + '_' + str(overlap) + '_2_split_' + str(n_div) \
                 + '.csv'

    # Extract the table headers.
    headers_2 = ['id', 'user', 'timestamp', 'acc_xs_mean', 'acc_ys_mean', 'acc_zs_mean', 'acc_xs_var', 'acc_ys_var',
                 'acc_zs_var', 'acc_xs_mad', 'acc_ys_mad', 'acc_zs_mad', 'acc_xs_max', 'acc_ys_max', 'acc_zs_max',
                 'acc_xs_min', 'acc_ys_min', 'acc_zs_min', 'acc_xs_iqr', 'acc_ys_iqr', 'acc_zs_iqr', 'gyro_xs_mean',
                 'gyro_ys_mean', 'gyro_zs_mean', 'gyro_xs_var', 'gyro_ys_var', 'gyro_zs_var', 'gyro_xs_mad',
                 'gyro_ys_mad', 'gyro_zs_mad', 'gyro_xs_max', 'gyro_ys_max', 'gyro_zs_max', 'gyro_xs_min',
                 'gyro_ys_min', 'gyro_zs_min', 'gyro_xs_iqr', 'gyro_ys_iqr', 'gyro_zs_iqr', 'magn_xs_mean',
                 'magn_ys_mean', 'magn_zs_mean', 'magn_xs_var', 'magn_ys_var', 'magn_zs_var', 'magn_xs_mad',
                 'magn_ys_mad', 'magn_zs_mad', 'magn_xs_max', 'magn_ys_max', 'magn_zs_max', 'magn_xs_min',
                 'magn_ys_min', 'magn_zs_min', 'magn_xs_iqr', 'magn_ys_iqr', 'magn_zs_iqr', 'gps_lat_mean',
                 'gps_long_mean', 'gps_alt_mean', 'gps_speed_mean', 'gps_bearing_mean', 'gps_accuracy_mean',
                 'gps_lat_var', 'gps_long_var', 'gps_alt_var', 'gps_speed_var', 'gps_bearing_var', 'gps_accuracy_var',
                 'gps_lat_mad', 'gps_long_mad', 'gps_alt_mad', 'gps_speed_mad', 'gps_bearing_mad', 'gps_accuracy_mad',
                 'gps_lat_max', 'gps_long_max', 'gps_alt_max', 'gps_speed_max', 'gps_bearing_max', 'gps_accuracy_max',
                 'gps_lat_min', 'gps_long_min', 'gps_alt_min', 'gps_speed_min', 'gps_bearing_min', 'gps_accuracy_min',
                 'gps_lat_iqr', 'gps_long_iqr', 'gps_alt_iqr', 'gps_speed_iqr', 'gps_bearing_iqr', 'gps_accuracy_iqr',
                 'activity_id', 'activity']

    headers_1 = ['id', 'user', 'timestamp', 'acc_xs_mean', 'acc_ys_mean', 'acc_zs_mean', 'acc_xs_var', 'acc_ys_var',
                 'acc_zs_var', 'acc_xs_mad', 'acc_ys_mad', 'acc_zs_mad', 'acc_xs_max', 'acc_ys_max', 'acc_zs_max',
                 'acc_xs_min', 'acc_ys_min', 'acc_zs_min', 'acc_xs_iqr', 'acc_ys_iqr', 'acc_zs_iqr', 'magn_xs_mean',
                 'magn_ys_mean', 'magn_zs_mean', 'magn_xs_var', 'magn_ys_var', 'magn_zs_var', 'magn_xs_mad',
                 'magn_ys_mad', 'magn_zs_mad', 'magn_xs_max', 'magn_ys_max', 'magn_zs_max', 'magn_xs_min',
                 'magn_ys_min', 'magn_zs_min', 'magn_xs_iqr', 'magn_ys_iqr', 'magn_zs_iqr', 'gps_lat_mean',
                 'gps_long_mean', 'gps_alt_mean', 'gps_speed_mean', 'gps_bearing_mean', 'gps_accuracy_mean',
                 'gps_lat_var', 'gps_long_var', 'gps_alt_var', 'gps_speed_var', 'gps_bearing_var', 'gps_accuracy_var',
                 'gps_lat_mad', 'gps_long_mad', 'gps_alt_mad', 'gps_speed_mad', 'gps_bearing_mad', 'gps_accuracy_mad',
                 'gps_lat_max', 'gps_long_max', 'gps_alt_max', 'gps_speed_max', 'gps_bearing_max', 'gps_accuracy_max',
                 'gps_lat_min', 'gps_long_min', 'gps_alt_min', 'gps_speed_min', 'gps_bearing_min', 'gps_accuracy_min',
                 'gps_lat_iqr', 'gps_long_iqr', 'gps_alt_iqr', 'gps_speed_iqr', 'gps_bearing_iqr', 'gps_accuracy_iqr',
                 'activity_id', 'activity']

    headers_0 = ['id', 'user', 'timestamp', 'acc_xs_mean', 'acc_ys_mean', 'acc_zs_mean', 'acc_xs_var', 'acc_ys_var',
                 'acc_zs_var', 'acc_xs_mad', 'acc_ys_mad', 'acc_zs_mad', 'acc_xs_max', 'acc_ys_max', 'acc_zs_max',
                 'acc_xs_min', 'acc_ys_min', 'acc_zs_min', 'acc_xs_iqr', 'acc_ys_iqr', 'acc_zs_iqr', 'gps_lat_mean',
                 'gps_long_mean', 'gps_alt_mean', 'gps_speed_mean', 'gps_bearing_mean', 'gps_accuracy_mean',
                 'gps_lat_var', 'gps_long_var', 'gps_alt_var', 'gps_speed_var', 'gps_bearing_var', 'gps_accuracy_var',
                 'gps_lat_mad', 'gps_long_mad', 'gps_alt_mad', 'gps_speed_mad', 'gps_bearing_mad', 'gps_accuracy_mad',
                 'gps_lat_max', 'gps_long_max', 'gps_alt_max', 'gps_speed_max', 'gps_bearing_max', 'gps_accuracy_max',
                 'gps_lat_min', 'gps_long_min', 'gps_alt_min', 'gps_speed_min', 'gps_bearing_min', 'gps_accuracy_min',
                 'gps_lat_iqr', 'gps_long_iqr', 'gps_alt_iqr', 'gps_speed_iqr', 'gps_bearing_iqr', 'gps_accuracy_iqr',
                 'activity_id', 'activity']

    # Open CSV file for writing.
    csvFile = csv.writer(open(filePath + fileName_0, 'w', newline=""), delimiter=',')
    csvFile_1 = csv.writer(open(filePath + fileName_1, 'w', newline=""), delimiter=',')
    csvFile_2 = csv.writer(open(filePath + fileName_2, 'w', newline=""), delimiter=',')

    # Add the headers and data to the CSV file.
    csvFile.writerow(headers_0)
    csvFile.writerows(data_0)
    csvFile_1.writerow(headers_1)
    csvFile_1.writerows(data_1)
    csvFile_2.writerow(headers_2)
    csvFile_2.writerows(data_2)


# This function calls the main process and writes all the data into CSV files.
def prepare_data(n_seconds, overlap, cut_seconds, n_div, index):
    # Set:
    # 0 - Acc + GPS (all users)
    # 1 - Acc + Magn + GPS (all users but the ones missing magnetometer)
    # 2 - Acc + Gyro + Magn + GPS (all users but the ones missing gyroscope and magnetometer)

    if index > 0:
        process_data_split(index, n_seconds, overlap, cut_seconds)
    else:
        # File path and name.
        filePath = './'
        fileName_0 = 'sensoringData_feature_prepared_' + str(n_seconds) + '_' + str(overlap) + '_0.csv'
        fileName_1 = 'sensoringData_feature_prepared_' + str(n_seconds) + '_' + str(overlap) + '_1.csv'
        fileName_2 = 'sensoringData_feature_prepared_' + str(n_seconds) + '_' + str(overlap) + '_2.csv'

        # Extract the table headers.
        headers_2 = ['id', 'user', 'timestamp', 'acc_xs_mean', 'acc_ys_mean', 'acc_zs_mean', 'acc_xs_var', 'acc_ys_var',
                     'acc_zs_var', 'acc_xs_mad', 'acc_ys_mad', 'acc_zs_mad', 'acc_xs_max', 'acc_ys_max', 'acc_zs_max',
                     'acc_xs_min', 'acc_ys_min', 'acc_zs_min', 'acc_xs_iqr', 'acc_ys_iqr', 'acc_zs_iqr', 'gyro_xs_mean',
                     'gyro_ys_mean', 'gyro_zs_mean', 'gyro_xs_var', 'gyro_ys_var', 'gyro_zs_var', 'gyro_xs_mad',
                     'gyro_ys_mad', 'gyro_zs_mad', 'gyro_xs_max', 'gyro_ys_max', 'gyro_zs_max', 'gyro_xs_min',
                     'gyro_ys_min', 'gyro_zs_min', 'gyro_xs_iqr', 'gyro_ys_iqr', 'gyro_zs_iqr', 'magn_xs_mean',
                     'magn_ys_mean', 'magn_zs_mean', 'magn_xs_var', 'magn_ys_var', 'magn_zs_var', 'magn_xs_mad',
                     'magn_ys_mad', 'magn_zs_mad', 'magn_xs_max', 'magn_ys_max', 'magn_zs_max', 'magn_xs_min',
                     'magn_ys_min', 'magn_zs_min', 'magn_xs_iqr', 'magn_ys_iqr', 'magn_zs_iqr', 'gps_lat_mean',
                     'gps_long_mean', 'gps_alt_mean', 'gps_speed_mean', 'gps_bearing_mean', 'gps_accuracy_mean',
                     'gps_lat_var', 'gps_long_var', 'gps_alt_var', 'gps_speed_var', 'gps_bearing_var',
                     'gps_accuracy_var', 'gps_lat_mad', 'gps_long_mad', 'gps_alt_mad', 'gps_speed_mad',
                     'gps_bearing_mad', 'gps_accuracy_mad', 'gps_lat_max', 'gps_long_max', 'gps_alt_max',
                     'gps_speed_max', 'gps_bearing_max', 'gps_accuracy_max', 'gps_lat_min', 'gps_long_min',
                     'gps_alt_min', 'gps_speed_min', 'gps_bearing_min', 'gps_accuracy_min', 'gps_lat_iqr',
                     'gps_long_iqr', 'gps_alt_iqr', 'gps_speed_iqr', 'gps_bearing_iqr', 'gps_accuracy_iqr',
                     'activity_id', 'activity']

        headers_1 = ['id', 'user', 'timestamp', 'acc_xs_mean', 'acc_ys_mean', 'acc_zs_mean', 'acc_xs_var', 'acc_ys_var',
                     'acc_zs_var', 'acc_xs_mad', 'acc_ys_mad', 'acc_zs_mad', 'acc_xs_max', 'acc_ys_max', 'acc_zs_max',
                     'acc_xs_min', 'acc_ys_min', 'acc_zs_min', 'acc_xs_iqr', 'acc_ys_iqr', 'acc_zs_iqr', 'magn_xs_mean',
                     'magn_ys_mean', 'magn_zs_mean', 'magn_xs_var', 'magn_ys_var', 'magn_zs_var', 'magn_xs_mad',
                     'magn_ys_mad', 'magn_zs_mad', 'magn_xs_max', 'magn_ys_max', 'magn_zs_max', 'magn_xs_min',
                     'magn_ys_min', 'magn_zs_min', 'magn_xs_iqr', 'magn_ys_iqr', 'magn_zs_iqr', 'gps_lat_mean',
                     'gps_long_mean', 'gps_alt_mean', 'gps_speed_mean', 'gps_bearing_mean', 'gps_accuracy_mean',
                     'gps_lat_var', 'gps_long_var', 'gps_alt_var', 'gps_speed_var', 'gps_bearing_var',
                     'gps_accuracy_var', 'gps_lat_mad', 'gps_long_mad', 'gps_alt_mad', 'gps_speed_mad',
                     'gps_bearing_mad', 'gps_accuracy_mad', 'gps_lat_max', 'gps_long_max', 'gps_alt_max',
                     'gps_speed_max', 'gps_bearing_max', 'gps_accuracy_max', 'gps_lat_min', 'gps_long_min',
                     'gps_alt_min', 'gps_speed_min', 'gps_bearing_min', 'gps_accuracy_min', 'gps_lat_iqr',
                     'gps_long_iqr', 'gps_alt_iqr', 'gps_speed_iqr', 'gps_bearing_iqr', 'gps_accuracy_iqr',
                     'activity_id', 'activity']

        headers_0 = ['id', 'user', 'timestamp', 'acc_xs_mean', 'acc_ys_mean', 'acc_zs_mean', 'acc_xs_var', 'acc_ys_var',
                     'acc_zs_var', 'acc_xs_mad', 'acc_ys_mad', 'acc_zs_mad', 'acc_xs_max', 'acc_ys_max', 'acc_zs_max',
                     'acc_xs_min', 'acc_ys_min', 'acc_zs_min', 'acc_xs_iqr', 'acc_ys_iqr', 'acc_zs_iqr', 'gps_lat_mean',
                     'gps_long_mean', 'gps_alt_mean', 'gps_speed_mean', 'gps_bearing_mean', 'gps_accuracy_mean',
                     'gps_lat_var', 'gps_long_var', 'gps_alt_var', 'gps_speed_var', 'gps_bearing_var',
                     'gps_accuracy_var', 'gps_lat_mad', 'gps_long_mad', 'gps_alt_mad', 'gps_speed_mad',
                     'gps_bearing_mad', 'gps_accuracy_mad', 'gps_lat_max', 'gps_long_max', 'gps_alt_max',
                     'gps_speed_max', 'gps_bearing_max', 'gps_accuracy_max', 'gps_lat_min', 'gps_long_min',
                     'gps_alt_min', 'gps_speed_min', 'gps_bearing_min', 'gps_accuracy_min', 'gps_lat_iqr',
                     'gps_long_iqr', 'gps_alt_iqr', 'gps_speed_iqr', 'gps_bearing_iqr', 'gps_accuracy_iqr',
                     'activity_id', 'activity']

        i = 1
        data_0 = []
        data_1 = []
        data_2 = []
        while i <= n_div:
            split_path_0 = 'sensoringData_feature_prepared_' + str(n_seconds) + '_' + str(
                overlap) + '_0_split_' + str(i) + '.csv'
            split_path_1 = 'sensoringData_feature_prepared_' + str(n_seconds) + '_' + str(
                overlap) + '_1_split_' + str(i) + '.csv'
            split_path_2 = 'sensoringData_feature_prepared_' + str(n_seconds) + '_' + str(
                overlap) + '_2_split_' + str(i) + '.csv'

            split_0_input = open(split_path_0, 'r')
            split_1_input = open(split_path_1, 'r')
            split_2_input = open(split_path_2, 'r')

            for row in csv.reader(split_0_input):
                if row[0] != "id":
                    data_0.append(row)
            for row in csv.reader(split_1_input):
                if row[0] != "id":
                    data_1.append(row)
            for row in csv.reader(split_2_input):
                if row[0] != "id":
                    data_2.append(row)

            i += 1

        # Open CSV file for writing.
        csvFile = csv.writer(open(filePath + fileName_0, 'w', newline=""), delimiter=',')
        csvFile_1 = csv.writer(open(filePath + fileName_1, 'w', newline=""), delimiter=',')
        csvFile_2 = csv.writer(open(filePath + fileName_2, 'w', newline=""), delimiter=',')

        # Add the headers and data to the CSV file.
        csvFile.writerow(headers_0)
        csvFile.writerows(data_0)
        csvFile_1.writerow(headers_1)
        csvFile_1.writerows(data_1)
        csvFile_2.writerow(headers_2)
        csvFile_2.writerows(data_2)


if __name__ == '__main__':
    # 20, 19, 5, 8
    if sys.argv[5]:
        index = int(sys.argv[5])
    else:
        index = -1
    startTime = time.time()
    prepare_data(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), index)
    elapsedTime = time.time() - startTime
    print(elapsedTime)
