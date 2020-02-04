# NLTK Demo
Created by Aki Ariga (aki@cloudera.com)<br>
This demonstrates how to distribute libraries out the worker nodes without having them pre-installed on the nodes. In this instance, we'll distribute the NLTK library. 

<b>Status</b>: Demo Ready<br>
<b>Use Case</b>: Library Distribution

<b>Steps</b>:<br>
1. In your projects, go to Settings > Engine and set the following environment variables: SPARK_CONFIG = NLTK-demo/spark-defaults.conf, PYSPARK_PYTHON=./NLTK/nltk_env/bin/python<br>
2. Open a terminal and run setup.sh<br>
3. Create a Python Session and run pyspark_nltk.py<br>

<b>Recommended Session Sizes</b>: 2 CPU, 4 GB RAM

<b>Estimated Runtime</b>: <br>
pyspark_ntlk.py --> < 15 seconds 

<b>Recommended Jobs/Pipeline</b>:<br>
None

<b>Demo Script</b><br>
TBD

<b>Related Content</b>:<br>
https://wiki.cloudera.com/display/~ariga/2017/04/11/How+to+Run+your+favorite+Python+library+On+PySpark+cluster+without+IT+permission+on+CDSW

