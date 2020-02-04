from pyspark.sql import SparkSession


session = SparkSession.builder.appName("dada").getOrCreate()

sc = session.

rdd = sc.parallelize([1, 1, 2, 3])