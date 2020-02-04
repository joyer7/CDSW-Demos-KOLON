# Spark Deep Learning Pipelines Demo
Created by Seth Hendrickson (shendrickson@cloudera.com)<br>
This is a simple example of how to leverage the 
[Spark Deep Learning Pipelines](https://github.com/databricks/spark-deep-learning)
package to do transfer learning and batch inference for deep learning models 
on Spark.

<b>Status</b>: Demo Ready<br>
<b>Use Case</b>: Spark package, deep learning

<b>Steps</b>:<br>
1. In your projects, go to Settings > Engine and set the following environment variables: SPARK_CONFIG = deep-learning-pipelines/spark-defaults.conf, PYSPARK_PYTHON=./SPARKDL/sparkdl/bin/python<br>
2. Open a terminal and run setup.sh<br>
3. Create a Python Session and run classify_flower.py<br>

<b>Recommended Session Sizes</b>: 4 CPU, 8 GB RAM

<b>Estimated Runtime</b>: <br>
classify_flower.py --> 200 seconds 

<b>Recommended Jobs/Pipeline</b>:<br>
None

<b>Demo Script</b><br>
TBD

<b>Related Content</b>:<br>
https://docs.databricks.com/applications/deep-learning/deep-learning-pipelines.html

