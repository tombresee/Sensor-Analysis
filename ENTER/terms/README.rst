

HAR 
  Human Activity Recognition (HAR) is a challenging time series classification task.  The goal of human activity recognition (HAR) is to automatically analyze and understand human actions from signals acquired by multimodal wearable and/or environmental devices, such as accelerometer, gyroscope, microphones, and camera.  When the wearable device is a smartphone, the most commonly used sensors are the accelerometer, gyroscope, and magnetometer.



Sensor-based, single-user activity recognition
    Sensor-based activity recognition integrates the emerging area of sensor networks with novel data mining and machine learning techniques to model a wide range of human activities.  Mobile devices (e.g. smart phones) provide sufficient sensor data and calculation power to enable physical activity recognition to provide an estimation of the energy consumption during everyday life. Sensor-based activity recognition researchers believe that by empowering ubiquitous computers and sensors to monitor the behavior of agents (under consent), these computers will be better suited to act on our behalf.


ARP
  Activity Recognition Process


Wearable Devices
  Wearable devices encompass all accessories attached to the person’s body or clothing incorporating computer technologies, such as smart clothing, and ear-worn devices. They enable to capture attributes of interest as motion, location, temperature, and ECG, among others.


Accelerometer
  The accelerometer is an electromechanical sensor that captures the rate of change of the velocity of an object over a time laps, that is, the acceleration. It is composed of many other sensors, including some microscopic crystal structures that become stressed due to accelerative forces. The accelerometer interprets the voltage coming from the crystals to understand how fast the device is moving and which direction it is pointing in. A smartphone records three-dimension acceleration, which join the reference devices axes. Thus, a trivariate time series is produced. The measuring unit is meters over second squared (m/s2) or g forces. Accelerometer is the most popular sensor in HAR, because it measures the directional movement of a subject’s motion status over time. Accelerometer signal combines the linear acceleration due to body motion and due to gravity. The presence of the gravity is a bias that can influence the accuracy of the classifier, and thus is a common practice to remove the gravity component from the raw signal.


Gyroscope
  The gyroscope measures three-axial angular velocity. Its unit is measured in degrees over second (degrees/s).


Magnetometer
  A magnetometer measures the change of a magnetic field at a particular location. The measurement unit is Tesla (T), and is usually recorded on the three axes.  In addition to accelerometers, gyroscopes, and magnetometers, other less common sensors are used in HAR. For example, Garcia-Ceja and Brena use a barometer to classify vertical activities, such as ascending and descending stairs. Cheng et al. and Foubert et al. use pressure sensors arrays to detect respectively activities and lying and sitting transitions.  Other researchers use biometric sensors.  For example, Zia et al. use electromyography (EMG) for fine-grained motion detection, and Liu et al. use electrocardiography in conjunction with accelerometer to recognize activities.



sampling rate 
  defined as the number of data points recorded in a second and is expressed in Hertz. For instance, if the sampling rate is equal to 50Hz, it means that 50 values per second are recorded. This parameter is normally set during the acquisition phase.


activity-defined windowing
  the initial and end points of each window are selected by detecting patterns of the activity changes.

event-defined windowing
  the window is created around a detected event. In some studies, it is also mentioned as windows around peak.

sliding windowing
  data are split into windows of fixed size, without gap between two consecutive windows, and, in some cases, overlapped. Sliding windowing is the most widely employed segmentation technique in activity recognition, especially for periodic and static activities.

features extraction 
  reduces the data dimensionality while extracting the most important peculiarity of the signal by abstracting each data segment into a high-level representation of the same segment.
  See:  https://link.springer.com/article/10.1007/s40860-021-00147-0/tables/1

SVM
  Among HAR classifiers, SVM is the most popular one



|
|
|

mean 
  Mean value

std 
  Standard deviation

mad 
  Median absolute value

max 
  Largest values in array

min 
  Smallest value in array

sma 
  Signal magnitude area

energy 
  Average sum of the squares

iqr 
  Interquartile range

entropy 
  Signal Entropy

arCoeff 
  Autorregresion coefficients

correlation 
  Correlation coefficient

maxFreqInd 
  Largest frequency component

meanFreq 
  Frequency signal weighted average

skewness 
  Frequency signal Skewness

kurtosis 
  Frequency signal Kurtosis

energyBand 
  Energy of a frequency interval

angle 
  Angle between two vectors


|
|
|

ADL 
  Activity of daily living

F
  Falls

A
  Accelerometer

LA
  Linear Acceleration Sensor 

G
  Gyroscope

M
  Magnetometer

AT
  attitude

OR
  orientation

L
  light

S
  sound

SM
  sound magnitude 

GPS
  Global Positioning System

CO
  compass

LO
  location

ST
  phone state

H
  highest frequency as possible

SP
  smartphone

SW
  smartwatch

IMU
  inertial measurement unit


