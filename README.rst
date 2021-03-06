
Milestone II - University of Michigan 
#######################################


|


:Authors: Tom Bresee, Michael Phillips
:Version: 0.1
:University: Michigan
:Course: SIADS 694/695: Milestone II


|
|
|


by placing multiple sensors at carefully selected places on the human
body, the precision of classifying the activities of the subjects goes up significantly


MASTER LIST OF THINGS WE NEED TO DO FOR THIS PROJECT 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset I - Supervised Learning**

* `Raw Dataset Description and Download <https://lbd.udc.es/research/real-life-HAR-dataset/>`_ - A Public Domain Dataset For Real-life Human Activity Recognition Using Smartphone Sensors

* This `location <https://data.mendeley.com/datasets/3xm88g6m6d/2>`_ also has those above raw files (as the authors prefer to store here). Moderate corrections were posted to the above `here <https://www.mdpi.com/1424-8220/20/16/4650/htm>`_.  Original paper in html form is `here <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7218897/>`_. 

* Load up above **raw** (not their processed) dataset into initial jupyter notebook, start looking around and getting a feel for the data.  This `paper <https://github.com/tombresee/Michigan_Milestone_Initial_Work/raw/main/ENTER/RAW%20DATASET%20I/sensors-20-02200-v3.pdf>`_ helps *tremendously* in getting a feel for the raw data

* Examine this initial dataset and start thinking about pre-processing and feature engineering.  Determine some initial ML approaches for predicting the 'state' a user is in (one of four categories).  Classifying into one of four categories is 'multi-class', and thus we will need a confusion matrix as well...

* add more

::
    Garcia-Gonzalez, D.; Rivero, D.; Fernandez-Blanco, E.; Luaces, M.R. A Public Domain Dataset for Real-Life Human Activity Recognition Using Smartphone Sensors. Available Online: https://data.mendeley.com/datasets/3xm88g6m6d/2 (accessed on 18 August 2020)

* Note:  Related but not used original UCI-level dataset can be found `here <Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set>`_ 

* `Really good video of experiement in action <https://www.youtube.com/watch?v=XOEN9W05_4A>`_ 


|



**Dataset II - Unsupervised Learning**

* **Plot**:  Take the list of sensor node GPS coordinates, and plot nicely into something like Leaflet or some visualization, this is an easy but nice to show visualization technique that will get us points.  Raw lat/long data files kept `here <https://github.com/tombresee/Michigan_Milestone_Initial_Work/blob/main/ENTER/RAW%20DATASET%20II/nodes.csv>`_.  The online arc gis like map from AoT is viewable `here <https://data.cityofchicago.org/Environment-Sustainable-Development/Array-of-Things-Locations-Map/2dng-xkng>`_, maybe we build our own version of this.  Should be easy to do, just taking lats/lons into some pretty picture.  


* **Great Lakes**:  Do some initial processing on the Michigan Great Lakes cluster, to build ability to later to ask for compute time for potential capstone extension analysis...

* **Heroku**:  This is not always easy to work on, need to start gradually adding some viz and verbage to https://michigan-milestone.herokuapp.com/ instance.  Incrementally learn/add to this.  Good opportunity for learning plotly/dash and advanced visualization here. 

* add more 



|
|
|



Master Reference Links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `Dataset 1 <https://lbd.udc.es/research/real-life-HAR-dataset/>`_ - A Public Domain Dataset For Real-life Human Activity Recognition Using Smartphone Sensors

* `R based Analysis <http://rstudio-pubs-static.s3.amazonaws.com/100601_62cc5079d5514969a72c34d3c8228a84.html>`_

* `UCI Similar Dataset <https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions>`_

* https://www.sciencedirect.com/science/article/pii/S0925231215010930

* https://scholar.google.com/citations?user=A7yUhT8AAAAJ&hl=en

* https://gizmodo.com/all-the-sensors-in-your-smartphone-and-how-they-work-1797121002

* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6386882/


::
    Relevant Papers:

    Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. International Workshop of Ambient Assisted Living (IWAAL 2012). Vitoria-Gasteiz, Spain. Dec 2012

    Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge L. Reyes-Ortiz. Energy Efficient Smartphone-Based Activity Recognition using Fixed-Point Arithmetic. Journal of Universal Computer Science. Special Issue in Ambient Assisted Living: Home Care. Volume 19, Issue 9. May 2013

    Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. 4th International Workshop of Ambient Assited Living, IWAAL 2012, Vitoria-Gasteiz, Spain, December 3-5, 2012. Proceedings. Lecture Notes in Computer Science 2012, pp 216-223.

    Jorge Luis Reyes-Ortiz, Alessandro Ghio, Xavier Parra-Llanas, Davide Anguita, Joan Cabestany, Andreu Catal??. Human Activity and Motion Disorder Recognition: Towards Smarter Interactive Cognitive Environments. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

    Citation Request:

	Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.



|
|



* `Dataset 2 <https://www.mcs.anl.gov/research/projects/waggle/downloads/datasets/index.php>`_ - We will use huge file 'AoT_Chicago.complete.latest.tar', where `this <https://github.com/waggle-sensor/waggle/blob/master/data/README.md>`_ explains how to unzip it 

Files:
::
    data.csv.gz	    # compressed file of all data values
    nodes.csv	    # list of nodes in the dataset and their metadata
    README.md	    # An explaination of the database fields 
    sensors.csv	    # A list of sensors and their metadata
    offsets.csv     # data.csv.gz file byte offsets


* `Array of Things Overview <http://arrayofthings.github.io/>`_

* `Array of Things Locations View <https://data.cityofchicago.org/Environment-Sustainable-Development/Array-of-Things-Locations-Map/2dng-xkng>`_ - City of Chicago

* `Array of Things Past Workshops <http://www.urbanccd.org/past-events>`_

* `Current AoT Node Architecture <http://arrayofthings.github.io/node.html>`_

* `Heroku Link <https://michigan-milestone.herokuapp.com/>`_


.. figure:: https://github.com/tombresee/Michigan_Milestone_Initial_Work/raw/main/ENTER/IMAGES/AoT-Diagram.jpg
   :scale: 50 %
   :alt: map to buried treasure

   Fig:  Current Architecture


|
|
|
|
|
|
|
|
|
|
|
|
|
|






































































 
  





|
|
|
|
|
|
|
|
