
import csv
import sys
import pandas as pd


def time_adapt_data(n_seconds, cut_seconds):
    acc_input = open('sensoringData_acc_clean.csv', 'r')
    gyro_input = open('sensoringData_gyro_clean.csv', 'r')
    magn_input = open('sensoringData_magn_clean.csv', 'r')
    gps_input = open('sensoringData_gps_clean.csv', 'r')

    acc_output = open('sensoringData_acc_prepared_' + str(n_seconds) + '.csv', 'w', newline="")
    gyro_output = open('sensoringData_gyro_prepared_' + str(n_seconds) + '.csv', 'w', newline="")
    magn_output = open('sensoringData_magn_prepared_' + str(n_seconds) + '.csv', 'w', newline="")
    gps_output = open('sensoringData_gps_prepared_' + str(n_seconds) + '.csv', 'w', newline="")

    acc_writer = csv.writer(acc_output, delimiter=",")
    gyro_writer = csv.writer(gyro_output, delimiter=",")
    magn_writer = csv.writer(magn_output, delimiter=",")
    gps_writer = csv.writer(gps_output, delimiter=",")

    act_columns = ['id', 'user', 'init_timestamp', 'end_timestamp', 'activity_id', 'activity']
    act_changes = pd.read_csv('activityChanges.csv', header=0, names=act_columns)

    invalid_sess = []

    def adaptation(input, output, writer, invalid_sess):
        username = ""
        activity_id = 0
        timestamp = 0.0
        timestamp_before = 0.0
        ignore = False

        for row in csv.reader(input):
            if row[0] != "id":
                username_now = row[1]
                activity_id_now = row[6]
                timestamp_now = float(row[2])
                if username_now != username or activity_id_now != activity_id:
                    username = row[1]
                    activity_id = row[6]
                    act_df = act_changes.loc[(act_changes['user'] == int(username)) &
                                             (act_changes['activity_id'] == int(activity_id))]
                    timestamp = act_df['init_timestamp'].values[0] + cut_seconds
                    end_timestamp = act_df['end_timestamp'].values[0] - cut_seconds
                    ignore = False
                    for user_inv, act_inv in invalid_sess:
                        if user_inv == username and act_inv == activity_id:
                            ignore = True
                    if not ignore:
                        if timestamp_now > (timestamp + cut_seconds):
                            ignore = True
                            invalid_sess.append([username, activity_id])
                    timestamp_before = timestamp_now
                if end_timestamp >= timestamp_now >= timestamp and not ignore:
                    if timestamp_now > (timestamp_before + cut_seconds):
                        ignore = True
                        invalid_sess.append([username, activity_id])
                    else:
                        writer.writerow(row)
                timestamp_before = timestamp_now
            else:
                writer.writerow(row)

        input.close()
        output.close()

        return invalid_sess

    def gps_adaptation(input, output, writer, invalid_sess):
        counter = -1
        i = 0
        username = ""
        activity_id = 0
        timestamp = 0.0
        gps_seconds = 1
        row_list = []
        session_list = []
        gps_values = []
        ignore = False

        for row in csv.reader(input):
            if row[0] != "id":
                username_now = row[1]
                activity_id_now = row[9]
                timestamp_now = float(row[2])
                if username_now != username or activity_id_now != activity_id:
                    if counter >= 0 and not ignore:
                        for list_row in row_list:
                            gps_values.append(list_row.copy())
                            i += 1
                        if counter == 0:
                            timestamp_before = float(row_before[2])
                            while timestamp_before >= (timestamp + gps_seconds):
                                timestamp += gps_seconds
                                last_row = row_list[len(row_list) - 1]
                                aux_last_row = last_row.copy()
                                if float(aux_last_row[2]) >= timestamp:
                                    aux_last_row[2] = str(timestamp)
                                    gps_values.insert(i - 1, aux_last_row.copy())
                                    i += 1
                        while end_timestamp >= (float(row_before[2]) + gps_seconds):
                            aux_time = float(row_before[2]) + gps_seconds
                            row_before[2] = aux_time
                            gps_values.append(row_before.copy())
                            i += 1
                    username = row[1]
                    activity_id = row[9]
                    act_df = act_changes.loc[(act_changes['user'] == int(username)) &
                                             (act_changes['activity_id'] == int(activity_id))]
                    timestamp = act_df['init_timestamp'].values[0]
                    end_timestamp = act_df['end_timestamp'].values[0]
                    ignore = False
                    for user_inv, act_inv in invalid_sess:
                        if user_inv == username and act_inv == activity_id:
                            ignore = True
                    if not ignore:
                        row_sess = [username, activity_id, timestamp, end_timestamp]
                        if row_sess not in session_list:
                            session_list.append(row_sess)
                    counter = 0
                    row_list.clear()
                else:
                    if timestamp_now >= (timestamp + gps_seconds) and not ignore:
                        for list_row in row_list:
                            gps_values.append(list_row.copy())
                            i += 1
                        while timestamp_now >= (timestamp + gps_seconds):
                            timestamp += gps_seconds
                            last_row = row_list[len(row_list) - 1]
                            aux_last_row = last_row.copy()
                            if counter == 0 and (float(aux_last_row[2]) >= timestamp):
                                aux_last_row[2] = str(timestamp)
                                gps_values.insert(i - 1, aux_last_row.copy())
                                i += 1
                            else:
                                aux_last_row[2] = str(timestamp)
                                gps_values.append(aux_last_row.copy())
                                i += 1
                        row_list.clear()
                    counter += 1
                row_list.append(row)
                row_before = row
            else:
                writer.writerow(row)

        if end_timestamp >= (float(row_before[2]) + gps_seconds) and not ignore:
            for list_row in row_list:
                gps_values.append(list_row.copy())
            while end_timestamp >= (float(row_before[2]) + gps_seconds):
                aux_time = float(row_before[2]) + gps_seconds
                row_before[2] = aux_time
                gps_values.append(row_before.copy())

        session_output = open('validSessions_' + str(n_seconds) + '.csv', 'w', newline="")
        session_writer = csv.writer(session_output, delimiter=",")
        session_row = ['username', 'activity_id', 'init_timestamp', 'end_timestamp']
        session_writer.writerow(session_row)
        for sess_row in session_list:
            session_writer.writerow(sess_row)

        writer.writerows(gps_values)

        input.close()
        output.close()
        session_output.close()

    invalid_sess = adaptation(acc_input, acc_output, acc_writer, invalid_sess)
    invalid_sess = adaptation(gyro_input, gyro_output, gyro_writer, invalid_sess)
    invalid_sess = adaptation(magn_input, magn_output, magn_writer, invalid_sess)
    gps_adaptation(gps_input, gps_output, gps_writer, invalid_sess)


if __name__ == '__main__':
    # 20, 5
    time_adapt_data(int(sys.argv[1]), int(sys.argv[2]))