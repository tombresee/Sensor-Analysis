import csv
import sys

def anomaly_detection(inp, out, th_lat, th_long, th_timestamp, th_sensor, th_alt, type):
    writer = csv.writer(out, delimiter=",")

    for row in csv.reader(inp):
        if row[0] != "id":
            if type == 'gps':
                gps_lat_increment = float(row[3])
                gps_long_increment = float(row[4])
                gps_alt_increment = float(row[5])
            else:
                gps_lat_increment = 0.0
                gps_long_increment = 0.0
                gps_alt_increment = 0.0
            if type == 'magn':
                magn_z = float(row[5])
            else:
                magn_z = 0.0
            timestamp = row[2]
            if (gps_lat_increment < th_lat) and (gps_long_increment < th_long) and (len(timestamp) < th_timestamp) and \
                    (not timestamp.startswith('1970')) and (magn_z < th_sensor) and (gps_alt_increment < th_alt):
                writer.writerow(row)
            else:
                if gps_lat_increment >= th_lat:
                    print("GPS Latitude increment too high: " + str(gps_lat_increment))
                if gps_long_increment >= th_long:
                    print("GPS Longitude increment too high: " + str(gps_long_increment))
                if gps_alt_increment >= th_alt:
                    print("GPS Altitude increment too high: " + str(gps_alt_increment))
                if len(timestamp) >= th_timestamp or timestamp.startswith('1970'):
                    print("Wrong timestamp: " + timestamp)
                if (magn_z > th_sensor):
                    print("Wrong sensor value: " + str(magn_z))
                print("")
        else:
            writer.writerow(row)

    inp.close()
    out.close()


if __name__ == '__main__':
    acc_input = open('sensoringData_acc.csv', 'r')
    gyro_input = open('sensoringData_gyro.csv', 'r')
    magn_input = open('sensoringData_magn.csv', 'r')
    gps_input = open('sensoringData_gps.csv', 'r')

    acc_output = open('sensoringData_acc_clean.csv', 'w', newline="")
    gyro_output = open('sensoringData_gyro_clean.csv', 'w', newline="")
    magn_output = open('sensoringData_magn_clean.csv', 'w', newline="")
    gps_output = open('sensoringData_gps_clean.csv', 'w', newline="")

    # 0.2, 0.2, 27, 2000, 500
    anomaly_detection(acc_input, acc_output, float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                      float(sys.argv[5]), 'acc')
    anomaly_detection(gyro_input, gyro_output, float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]),
                      int(sys.argv[4]), float(sys.argv[5]), 'gyro')
    anomaly_detection(magn_input, magn_output, float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]),
                      int(sys.argv[4]), float(sys.argv[5]), 'magn')
    anomaly_detection(gps_input, gps_output, float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                      float(sys.argv[5]), 'gps')