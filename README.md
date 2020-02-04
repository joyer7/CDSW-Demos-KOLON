# CDSW-Demos
This repo contains CDSW demos for field use. You can pull them down into a project by putting http://github.mtv.cloudera.com/SE-SPEC-DPML/CDSW-Demos into the "git url" box when you create a new project and they'll all be pulled down into a single CDSW project.

Currently the following demos are in good working order and recommended for use:

1. AAS-network-traffic-anomaly-detection
1. AAS-predicting-forest-cover
1. basketball-stats
1. ds-for-teclo
1. flight-analytics
1. tensorflow-tutorial
1. DataRobot 
1. NLTK-demo 
1. wordcloud_alice 
1. deep-learning-pipelines

The following are currently In-Progess. They are close to being ready but may require some additional work not documented in the readme.md to get working. Try with caution:

1. hail-genetics-tutorial
1. ibis-test
1. PolyReg-demo 
1. Predictive Maintenance 
1. Shiny-demo
1. German Credit 

If you have a cool demo that you think may be of interest to the field, or shows off a cool feature we don't have above, please push it into this repo. The more the merrier! We only ask that you adhere to the recommend structure below to help make it easy for users to use the demos. 

Recommended structure of directories is as follows: 

1. readme.md - markup that provides an overview of the demo, explains the order to run scripts, recommnedations for building jobs, etc. Also, try to adhere to the structure that we're using in all the README.md files: status, use case, steps, recommended sessions size, estimated runtime, recommended jobs/pipeline, notes, demo script, related content. For example, see: http://github.mtv.cloudera.com/SE-SPEC-DPML/CDSW-Demos/tree/master/ds-for-telco 

1. /data - stores any data needed for the demo. Should be small/medium sized data sets. 

1. setup.sh - script that runs any setup commands prior to running main CDSW scripts. This is stuff like pulling down large data sets, loading data into HDFS, etc. 

1. \<scripts\>.\<py/r/scala\> - main CDSW scripts. This is where a bulk of the demo lives. Good commenting is encouraged to help people in the field demo. 

1. cleanup.sh - script that cleans out the system so \<scripts\>.\<py/r/scala\> can be run again w/o error. 

1. spark-defaults.conf - recommend spark settings to be used by CDSW

When you are finished, email the SE Specialization team (se-spec-data-processing-ml@cloudera.com) and we'll verify that it's working on a fresh install. Thanks!
