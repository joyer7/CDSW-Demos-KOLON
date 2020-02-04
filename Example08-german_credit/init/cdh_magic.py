# magic functions to use with dsw
# the following are intended to simplify notebook precesses for the user while developing

from IPython.core.magic import register_line_cell_magic
from IPython.core.display import display, HTML
import subprocess
import shlex
import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import Row


# Configuration parameters
max_show_lines = 50         # Limit on the number of lines to show with %sql_show and %sql_display
detailed_explain = True     # Set to False if you want to see only the physical plan when running explain

@register_line_cell_magic
def hdfs(x, cell=None):
    y = shlex.split("hdfs " + x)
    print(subprocess.Popen(y, stdout=subprocess.PIPE).communicate()[0])
    return None

# TODO: change to beeline & add impala
@register_line_cell_magic
def hive(x, cell=None):
    y = shlex.split("hive " + x)
    print(subprocess.Popen(y, stdout=subprocess.PIPE).communicate()[0])
    return None  

# SECTION FOR HELP
@register_line_cell_magic
def sql(line, cell=None):
    print("TODO: lists magics")
    
    return None
  
# SECTION FOR SPARK SQL magics
# TODO: add file reference read as option
@register_line_cell_magic
def sql(line, cell=None):
    return sqlCtx.sql(line).collect()

@register_line_cell_magic
def sql_show(line, cell=None):
    return sqlCtx.sql(line).show(max_show_lines) 

@register_line_cell_magic
def sql_display(line, cell=None):
    return sqlCtx.sql(line).limit(max_show_lines).toPandas() 