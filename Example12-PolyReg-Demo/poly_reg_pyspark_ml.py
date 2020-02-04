# # Polynomial Regression - pyspark, ML
# To demonstrate the differences of running models in local python and pyspark, we'll use a simple [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression).
# Note that there are two classes you can run machine learning models with the spark [python API](http://spark.apache.org/docs/2.0.0/api/python/); [ML](http://spark.apache.org/docs/2.0.0/api/python/pyspark.ml.html) & [MLlib](http://spark.apache.org/docs/2.0.0/api/python/pyspark.mllib.html). This demonstration will be done using DataFrames which requires the **ML** class.

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
from pyspark.ml.linalg import Vectors
import pandas as pd
import numpy as np

x1      = RandomRDDs.uniformRDD(spark, 10000).map(lambda x: 6.0*x-2)
epsilon = RandomRDDs.normalRDD(spark, 10000).map(lambda x: 4.0*x)
def gen_poly(x):
    x1 = float(x[0])
    x2 = float(np.power(x1,2))
    x3 = float(np.power(x1,3))
    X  = Vectors.dense(x1,x2,x3)  
    epsilon = float(x[1])
    y  = 4.0 + 6.0*x1 + 2.0*x2 - 1*x3 + epsilon
    return(y,X)
gen_dat = x1.zip(epsilon).map(gen_poly)

# ## Prepare Data as DataFrame
# This example will use the ML class which requires our data to be structured as a spark DataFrame.
# Well define our column for labels, y, and a column for feature vectors, X.

dat = spark.createDataFrame(gen_dat,["y", "X"])

# ## Train Model and Review Model Coefficients
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="X",labelCol="y")
lrModel = lr.fit(dat)

coef = pd.DataFrame(data=np.dstack((np.append(np.array([lrModel.intercept, ]), lrModel.coefficients), [4,6,2,-1]))[0])
coef.columns = ['ml_coef', 'true_coef']
coef
