import os
import sys

from IPython.core.display import display, HTML

# Path for spark source folder
os.environ['SPARK_HOME']="/opt/cloudera/parcels/CDH/lib/spark/"
os.environ['PYSPARK_PYTHON']="/usr/bin/python"
#os.environ['PYSPARK_PYTHON']="/opt/conda/bin/python"
os.environ['HDFS_USER_NAME']="barker"
os.environ['PYSPARK_SUBMIT_ARGS']="--master yarn pyspark-shell"

sys.path.append("/opt/cloudera/parcels/CDH/lib/spark/python/")
sys.path.append("/opt/cloudera/parcels/CDH/lib/spark/python/lib/py4j-0.9-src.zip")

import atexit
import platform
import py4j
import pyspark
from pyspark.context import SparkContext
from pyspark.sql import SQLContext, HiveContext
from pyspark.storagelevel import StorageLevel
from pyspark.sql.types import *

sc = SparkContext(appName="pySpark_barker", pyFiles='')
atexit.register(lambda: sc.stop())
sc._jvm.org.apache.hadoop.hive.conf.HiveConf()
sqlCtx = HiveContext(sc)

print("""Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version %s
      /_/
""" % sc.version)
print("Using Python version %s (%s, %s)" % (
    platform.python_version(),
    platform.python_build()[0],
    platform.python_build()[1]))
print("SparkContext available as sc, %s available as sqlCtx." % sqlCtx.__class__.__name__)

# Pretty table, needs some work
htm = "<table> "
htm = htm + "<tr><th><a href='http://lannister-001.edh.cloudera.com:8088/cluster/app/%s'>YARN APPLICATION</a></th></tr> "
htm = htm + "<tr><th><a href='http://lannister-001.edh.cloudera.com:18089/history/%s'>SPARK HISTORY</a></th></tr> "
htm = htm + "</table>"

appid = sc.applicationId

display(HTML(htm % (appid ,appid )))
