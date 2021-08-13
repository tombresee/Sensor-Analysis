import csv
import sys
import pandas as pd


def sensor_split(input, writers, row_count, last_act_ids, last_usernames, val_sess_df, is_gps):
    counter = 0
    row_counter = 0
    username = ""
    activity_id = 0
    row_list = []
    n_div = 0
    writer_now = writers[0]
    usernames = []
    act_ids = []
    invalid = False
    last_valid_username = ""
    last_valid_activity_id = 0

    for row in csv.reader(input):
        if row[0] != "id":
            if counter == 0:
                if row_count < 0:
                    if username == last_usernames[n_div] and activity_id == last_act_ids[n_div]:
                        row_counter = 0
                        n_div += 1
                        writer_now = writers[n_div]
                else:
                    if row_count <= row_counter:
                        row_counter = 0
                        n_div += 1
                        writer_now = writers[n_div]
                        if not invalid:
                            usernames.append(username)
                            act_ids.append(activity_id)
                        else:
                            usernames.append(last_valid_username)
                            act_ids.append(last_valid_activity_id)
                username = int(row[1])
                if is_gps:
                    activity_id = int(row[9])
                else:
                    activity_id = int(row[6])
                sess_df = val_sess_df.loc[(val_sess_df['user'] == username) &
                                          (val_sess_df['activity_id'] == activity_id)]
                if len(sess_df) > 0:
                    invalid = False
                    last_valid_username = username
                    last_valid_activity_id = activity_id
                else:
                    invalid = True
            username_now = int(row[1])
            if is_gps:
                activity_id_now = int(row[9])
            else:
                activity_id_now = int(row[6])
            if username_now != username or activity_id_now != activity_id:
                if not invalid:
                    for list_row in row_list:
                        writer_now.writerow(list_row)
                row_list.clear()
                counter = 0
            else:
                counter += 1
            row_list.append(row)
            row_counter += 1
        else:
            for writer in writers:
                writer.writerow(row)

    if not invalid:
        usernames.append(username)
        act_ids.append(activity_id)
    else:
        usernames.append(last_valid_username)
        act_ids.append(last_valid_activity_id)

    return act_ids, usernames


def split_data(n_seconds, n_div):
    # We select gyroscope as the marker, as it is the most absent in every session.
    input_gyro = open('sensoringData_gyro_prepared_' + str(n_seconds) + '.csv', 'r')
    row_count_gyro = sum(1 for row in csv.reader(input_gyro))
    row_count_gyro_div = row_count_gyro / n_div

    val_sess_path = './validSessions_' + str(n_seconds) + '.csv'
    val_sess_columns = ['user', 'activity_id', 'init_timestamp', 'end_timestamp']
    val_sess_df = pd.read_csv(val_sess_path, header=0, names=val_sess_columns)
    val_sess_df.head()

    i = 1
    writers_acc = []
    writers_gyro = []
    writers_magn = []
    writers_gps = []
    while i <= n_div:
        writers_acc.append(csv.writer(open('sensoringData_acc_prepared_' + str(n_seconds) + '_' + str(i) + '.csv',
                                      'w', newline=""), delimiter=","))
        writers_gyro.append(csv.writer(open('sensoringData_gyro_prepared_' + str(n_seconds) + '_' + str(i) + '.csv',
                                       'w', newline=""), delimiter=","))
        writers_magn.append(csv.writer(open('sensoringData_magn_prepared_' + str(n_seconds) + '_' + str(i) + '.csv',
                                       'w', newline=""), delimiter=","))
        writers_gps.append(csv.writer(open('sensoringData_gps_prepared_' + str(n_seconds) + '_' + str(i) + '.csv',
                                      'w', newline=""), delimiter=","))
        i += 1

    input_acc = open('sensoringData_acc_prepared_' + str(n_seconds) + '.csv', 'r')
    input_gyro = open('sensoringData_gyro_prepared_' + str(n_seconds) + '.csv', 'r')
    input_magn = open('sensoringData_magn_prepared_' + str(n_seconds) + '.csv', 'r')
    input_gps = open('sensoringData_gps_prepared_' + str(n_seconds) + '.csv', 'r')

    last_act_ids, last_usernames = sensor_split(input_gyro, writers_gyro, row_count_gyro_div, [], [], val_sess_df, False)
    sensor_split(input_acc, writers_acc, -1, last_act_ids, last_usernames, val_sess_df, False)
    sensor_split(input_magn, writers_magn, -1, last_act_ids, last_usernames, val_sess_df, False)
    sensor_split(input_gps, writers_gps, -1, last_act_ids, last_usernames, val_sess_df, True)


if __name__ == '__main__':
    split_data(int(sys.argv[1]), int(sys.argv[2]))


