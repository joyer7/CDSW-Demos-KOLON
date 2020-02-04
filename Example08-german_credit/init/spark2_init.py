from __future__ import print_function
import sys
import numpy as np
from pyspark.sql import SparkSession

import os
import sys

from IPython.core.display import display, HTML

import atexit
import platform
import py4j
import pyspark
from pyspark.context import SparkContext
from pyspark.sql import SQLContext, HiveContext
from pyspark.storagelevel import StorageLevel
from pyspark.sql.types import *

os.environ['PYSPARK_SUBMIT_ARGS']="--master yarn pyspark-shell"

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
print("TODO: add specific arg settings")

# Pretty table, needs some work
htm = "<table> "
htm = htm + "<tr><th><a href='http://cdh-barker-1.vpc.cloudera.com:8088/cluster/app/%s'>YARN APPLICATION</a></th></tr> "
htm = htm + "<tr><th><a href='http://cdh-barker-1.vpc.cloudera.com:8088/proxy/%s'>APPLICATION MASTER</a></th></tr> "
htm = htm + "<tr><th><a href='http://cdh-barker-1.vpc.cloudera.com:18089/history/%s'>SPARK HISTORY</a></th></tr> "
htm = htm + "</table>"

appid = sc.applicationId

display(HTML(htm % (appid, appid, appid)))