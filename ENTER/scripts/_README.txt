
**************************************************

	*The expected flow to use these scripts is the one that follows:
		1.- Anomaly_Detector.py
			Used to preprocess data and detect possible outliers
			that might corrupt the model training. Parameters:
			- GPS latitude increments threshold (0.2).
			- GPS longitude increments threshold (0.2).
			- GPS altitude increments threshold (500).
			- Timestamp value length (27).
			- Z-axis magnetometer value threshold (2000).
		2.- Data_Adapter.py.
			This script cuts the first and final X seconds from
			each activity session. It also detects corrupted sessions,
			which are the ones that have gaps in the data related
			to all sensors but GPS (time gaps higher than five seconds).
			Also, replicates GPS data, in order to have at least
			one observation from this sensor in each sliding window.
			A validSessions file is also created to fasten feature
			extraction process by not evaluation sessions that did
			not record any GPS observation. Parameters:
			- Window size, in seconds (20).
			- Seconds to be cut from the first and final part of
			  each session (5).
		3.- Data_Splitter.py
			Script used to split the data got from the previous
			script in X parts in order to fasten the feature 
			extraction process. Parameters:
			- Window size used, in seconds (20).
			- Number of divisions to be applied (8).
		4.- Feature_Extraction.py
			Here the feature computation for each window size is
			made. For each sliding window, we compute mean, var, 
			mad, max, min and iqr functions over related data. This
			creates a file for each of the sets defined:
				0 - Acc + GPS (all users)
				1 - Acc + Magn + GPS (all users but the ones missing 
		    	    	    magnetometer)
				2 - Acc + Gyro + Magn + GPS (all users but the ones 
		    	    	    missing gyroscope and magnetometer)
			It is coded in a Slurm way to be executed as a job array 
			(one job for every data split). Parameters:
			- Window size used, in seconds (20).
			- Overlap between windows, in seconds (19).
			- Number of seconds set to cut the first and final part of
			  session (5).
			- Number of divisions applied over data (8).
			- Slurm job array index, from 1 to the number of divisions
			  specified before (or -1 to join all data).
		5.- SVM.py	
			Script used to train and test the SVM model proposed and
			obtaining the results. It is coded in a Slurm way to be
			executed as a job array (one job for every fold computed
			over the data). Parameters:
			- String formed by the window size, overlap size and
			  corresponding set, divided by low bars (20_19.0_2).
			- Slurm job array index, from 1 to 10.
			
	*Note: the parenthesis in each parameter means the value used in our work.			
			

**************************************************