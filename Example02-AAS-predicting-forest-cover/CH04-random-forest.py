## Predicting Forest Cover With Random Forest
#_Adapted from Chapter 4 of
#[Advanced Analytics with Spark](http://shop.oreilly.com/product/0636920035091.do)
#from O'Reilly Media. _

#_Original source code at https://github.com/sryza/aas/tree/master/ch04-rdf _


#_Source repository for this code: https://github.com/srowen/aaws-cdsw-examples _ 

### Example
#
#This example will demonstrate the usage of Spark MLLib and Random Forest to predict forest 
#cover
#
### Dataset
#
#The dataset used for demonstration is the well-known Covtype data set available 
#[online](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/). The dataset 
#records the types of forest covering parcels of land in Colorado, USA. Each example 
#contains several features describing each parcel of land, like its elevation, slope, 
#distance to water, shade, and soil type, along with the known forest type covering the 
#land. The forest cover type is to be predicted from the rest of the features, of which 
#there are 54 in total. The covtype.data file was extracted and copied into HDFS.

#To start, read the data set into a Spark `DataFrame` using the read method for parsing 
#comma-separated strings on the SQLContext and allowing the schema to be inferred based 
#on the data.


from __future__ import print_function
import numpy as np
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession\
    .builder\
    .appName("Forest Cover")\
    .getOrCreate()

spark.conf.get("spark.executor.instances")
spark.conf.get("spark.executor.cores")
    
dataWithoutHeader = spark\
    .read\
    .option("header", "false")\
    .option("inferSchema", "true")\
    .csv("hdfs:///tmp/Covtype/covtype.data")
    
colNames = ["Elevation", "Aspect", "Slope",
            "Horizontal_Distance_To_Hydrology", 
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points"] + \
["Wilderness_Area_%s" % i for i in range(4)] + \
["Soil_Type_%s" % i for i in range(40)] + ["Cover_Type"]

data = dataWithoutHeader.toDF(*colNames)
description = data.describe()
description.select(*description.columns[:5]).show()

# Display a few sample records in the data. 
  
data.select(*data.columns[:5]).show(5)
  

# Review the distribution of the labels in the "Cover_Type" variable. The number of 
# observations falling under each of the categories add up to 581012. 
  
label_hist = data.select("Cover_Type").groupBy("Cover_Type").count()\
    .orderBy("count")
label_hist.show()
label_hist.toPandas().plot.barh(x='Cover_Type', y='count')
  
#Before we begin building a random forest model, we would want to take a second look at
#the features. Notice we have several dummy variables in the data - "Wilderness_Area_0-3" 
#and "Soil_Type_0-39". It will be more optimal if these are represented by a single 
#variable i.e. one variable for Wilderness_Area and one for Soil_Type. This will allow the 
#Random Forest model to create decisions based on groups of categories in one decision 
#rather than considering each of the dummy variable versions. Below VectorAssembler is 
#deployed to combine the 4 and 40 wilderness and soil type columns into two Vector-valued 
#columns.

from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector, Vectors

wildernessCols = ["Wilderness_Area_%d" % i for i in range(4)]
data.select(*map(lambda c: col(c), wildernessCols)).show()

wildernessAssembler = VectorAssembler(inputCols=wildernessCols, outputCol="wilderness")

#Defining a UDF that transforms a vector's values into a numeric column that indicates 
#the location that has the value 1. 

from pyspark.sql.types import FloatType
unhotUDF = udf(lambda vec: float(vec.toArray().tolist().index(1.0)), FloatType())

withWilderness = wildernessAssembler.transform(data).\
    drop(*wildernessCols).\
    withColumn("wilderness", unhotUDF("wilderness"))

soilCols = ["Soil_Type_%d" % i for i in range(40)]

soilAssembler = VectorAssembler(inputCols=soilCols, outputCol="soil")

data_unencodeOneHot = soilAssembler.transform(withWilderness)\
    .drop(*soilCols)\
    .withColumn("soil", unhotUDF("soil"))
    
'''
Reviewing a few sample observations observation 
'''
  
data_unencodeOneHot.show(5)

#Spark MLlib requires all of the features to be collected into one column, whose values 
#is a Vector. This is achieved through the VectorAssembler. 
  
assembler = VectorAssembler(inputCols=filter(lambda c: c != "Cover_Type", data_unencodeOneHot.columns),
                           outputCol="featureVector")
dataAssembled = assembler.transform(data_unencodeOneHot)
dataAssembled.select("Cover_Type", "featureVector").show(20, truncate=False)


#Please note that we still want the model to consider these features as "categorical" and 
#not "numeric" for which a VectorIndexer will be used.
#The VectorIndexer helps index categorical features in datasets of Vectors. It can both
#automatically decide which features are categorical and convert original values to 
#category indices. 

from pyspark.ml.feature import VectorIndexer

indexer = VectorIndexer(maxCategories=40, inputCol="featureVector",
                            outputCol="indexedVector")
indexerModel = indexer.fit(dataAssembled)
print(indexerModel.categoryMaps)

#Create new column "indexedVector" with categorical values transformed to indices 

indexedData = indexerModel.transform(dataAssembled)
indexedData.select("indexedVector").show(20, truncate=False)

#Split the data into 90% train (+ validation), 10% test. Make sure you specify the seed 
#value, this will ensure that the results are reproducible later on. 

unencTrainData, unencTestData = indexedData.randomSplit([0.9, 0.1], seed=123)
unencTrainData.cache()
unencTestData.cache()

#Setup the RandomForestClassifier to specify the target and feature columns along with
#other hyperparameters like measure of impurity - Gini or entropy, number 
#of bins and the prediction column. 
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel

classifier = RandomForestClassifier().\
    setSeed(123).\
    setLabelCol("Cover_Type").\
    setFeaturesCol("indexedVector").\
    setPredictionCol("prediction").\
    setImpurity("entropy").\
    setMaxDepth(10).\
    setMaxBins(200)
  
# It's usually helpful to understand the nobs on our training algorithms,
# so we can know what they mean and how to set them.
print(classifier.explainParams())

#It is not always obvious what the parameter values should be. Below we are setting up 
#a parameter grid for number of trees ranging from 1 to 10 and minimum info gain from 0 
#to 0.05 to help build a better model 
from pyspark.ml.tuning import ParamGridBuilder

paramGrid = ParamGridBuilder()\
    .addGrid(classifier.minInfoGain, [0.0, 0.05])\
    .addGrid(classifier.numTrees, [1, 5])\
    .build()
    
#Setting up a MulticlassClassificationEvaluator that can compute accuracy and other metrics
#that evaluate the quality of the model’s predictions. 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
  
multiclassEval = MulticlassClassificationEvaluator().\
    setLabelCol("Cover_Type").\
    setPredictionCol("prediction").\
    setMetricName("accuracy")
    
#The TrainValidationSplit below splits the data into 90% training and 10%
#validation set and allows us to obtain a more reliable model based on the parameter grid.
#It is similar to the Cross Validation split but with k=1 making it less expensive.
from pyspark.ml.tuning import TrainValidationSplit

validator = TrainValidationSplit().\
    setSeed(123).\
    setEstimator(classifier).\
    setEvaluator(multiclassEval).\
    setEstimatorParamMaps(paramGrid).\
    setTrainRatio(0.9)
validatorModel = validator.fit(unencTrainData)

#In order to query the parameters chosen by the RandomForestClassifier, it’s necessary 
#to manually extract the RandomForestClassificationModel. 

forestModel = validatorModel.bestModel

# Extract number of trees associated with the bestModel - forestModel 
forestModel.getNumTrees

#Printing a representation of the model shows some of its tree structure. It consists of 
#a series of nested decisions about features, comparing feature values to thresholds.
  
print(forestModel.toDebugString[:1000])

#Assess the importance of input features as part of their building process. That is, 
#estimate how much each input feature contributes to making correct predictions. Pairing
#the importance values with features names, higher the better. 
  
inputCols = filter(lambda c: c != "Cover_Type", unencTrainData.columns)
importances = sorted(zip(forestModel.featureImportances.toArray(), inputCols), reverse=True)
importances

#The resulting forestModel is itself a Transformer, because it can transform a DataFrame 
#containing feature vectors into a DataFrame also containing predictions. For example, 
#it might be interesting to see what the model predicts on the "test" data. 
  
predictions = forestModel.transform(unencTestData)
  
#The "prediction" column below is the model's predicted label and the "probability" 
#column is a vector with the individual class probabilities of the "Cover_Type". 

predictions.select("Cover_Type", "prediction", "probability").show(5, truncate=False)

#Using the MulticlassClassificationEvaluator to compute accuracy to help evaluate the 
#quality of the model’s predictions. 
  
testAccuracy = multiclassEval.evaluate(forestModel.transform(unencTestData))
testAccuracy

#A confusion matrix can help us derive more insights into the quality of predictions. 
#Note, it is a 7*7 matrix with the rows indicating the actual "Cover_Type" and the columns 
#indicating the "predicted" values. The values along the diagonal indicate the entries that 
#have been classified correctly. An ideal model would have non-zero values as the diagonal 
#entries and zero else where. 
  
confusionMatrix = predictions.\
    groupBy("Cover_Type").\
    pivot("prediction", range(1, 8)).\
    count().\
    na.fill(0.0).\
    orderBy("Cover_Type")
confusionMatrix.show()

#The figure below displays the relative variable importances for each of the twelve 
#predictor variables. Not suprisingly, soil and elevation are the most relevant predictors!
imp_df = pd.DataFrame(importances, columns=['importance', 'feature'])
imp_df.plot.barh(x='feature', y='importance')