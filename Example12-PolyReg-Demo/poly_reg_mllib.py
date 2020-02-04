# # Polynomial Regression - pyspark, MLlib
# To demonstrate the differences of running models in local python and pyspark, we'll use a simple [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression).
# Note that there are two classes you can run machine learning models with the spark [python API](http://spark.apache.org/docs/2.0.0/api/python/); [ML](http://spark.apache.org/docs/2.0.0/api/python/pyspark.ml.html) & [MLlib](http://spark.apache.org/docs/2.0.0/api/python/pyspark.mllib.html). This demonstration will be done using DataFrames which requires the **MLlib** class.

# ## Initialize a pyspark Spark Context
# Since procuring a spark context will be a pretty common task for any spark analysis, I recommend defining an initialization magic to be sourced for every python session in the project. This is accomplished by writting the initialization magic to a file, [init_pyspark.py](http://github.mtv.cloudera.com/barker/demo_poly_reg/blob/master/init/init_pyspark.py), and sourcing that file at startup using [startup_init_pyspark.py](http://github.mtv.cloudera.com/barker/demo_poly_reg/blob/master/.ipython/profile_default/startup/startup_init_pyspark.py).
# Now you can initialize your sparkylr spark ml spark context via magic:

from pyspark.sql import SparkSession

spark = SparkSession.builder \
          .appName("spark-pypspark-ml") \
          .getOrCreate() 

# ---
# The initialization creates two global environment variables; `sc` & `sqlCtx`.

# ## Generate Spark DataFrame Data
# We'll generate sample data for a multivariate linear regression with known coefficients and randomly generated error. Specifically;
# $$ y = \beta_0 + \sum_i (\beta_i x_i) + \epsilon   \thinspace \thinspace \thinspace \thinspace \thinspace     \forall i \in {1..3}$$
# $$ \beta_0: 4 $$
# $$ \beta_1: 6 $$
# $$ \beta_2: 2 $$
# $$ \beta_3: -1 $$

from pyspark.mllib.random import RandomRDDs
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import pandas as pd
import numpy as np

x1      = RandomRDDs.uniformRDD(spark, 10000).map(lambda x: 6.0*x-2)
epsilon = RandomRDDs.normalRDD(spark, 10000).map(lambda x: 0.04*x)
def gen_poly(x):
    x0 = 1.0
    x1 = float(x[0])
    x2 = float(np.power(x1,2))
    x3 = float(np.power(x1,3))
    X  = Vectors.dense(x0,x1,x2,x3)  
    epsilon = float(x[1])
    y  = 4.0 + 6.0*x1 + 2.0*x2 - 1*x3 + epsilon
    return(y,X)
gen_dat = x1.zip(epsilon).map(gen_poly)

# ## Prepare Data as LabeledPoint Form
# This example will use the MLlib class which requires our data to be structured as a spark RDD of LabeledPoints. NOTE: the RDD approach was deprecated in 2.0.0.

dat = gen_dat.map(lambda obs: LabeledPoint(obs[0],obs[1]))

# ## Traing Model and Review Model Coefficients
from pyspark.mllib.regression import LinearRegressionModel
from pyspark.mllib.regression import LinearRegressionWithSGD

model = LinearRegressionWithSGD.train(data=dat,
                                      iterations=3000,
                                      step=0.01,
                                      miniBatchFraction=1.0,
                                      initialWeights=[2,-1,1,-2],
                                      regType=None,
                                      intercept=False,
                                      validateData=False,
                                      convergenceTol=0.00001)

model.intercept
model.weights

# TODO: figure out whats wrong with fit 
