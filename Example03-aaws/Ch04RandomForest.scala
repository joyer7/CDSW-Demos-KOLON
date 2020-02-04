/*
# Predicting Forest Cover With Random Forest
_Adapted from Chapter 4 of
[Advanced Analytics with Spark](http://shop.oreilly.com/product/0636920035091.do)
from O'Reilly Media. _
*/
/* 
_Original source code at https://github.com/sryza/aas/tree/master/ch04-rdf _
*/
/*
_Source repository for this code: https://github.com/srowen/aaws-cdsw-examples _
*/
  
/*
## Example
This example will demonstrate the usage of Spark MLLib and Random Forest to predict forest 
cover

## Dataset
The dataset used for demonstration is the well-known Covtype data set available 
[online](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/). The dataset 
records the types of forest covering parcels of land in Colorado, USA. Each example 
contains several features describing each parcel of land, like its elevation, slope, 
distance to water, shade, and soil type, along with the known forest type covering the 
land. The forest cover type is to be predicted from the rest of the features, of which 
there are 54 in total. The covtype.data file was extracted and copied into HDFS.
*/

/* 
To start, read the data set into a Spark `DataFrame` using the read method for parsing 
comma-separated strings on the SQLContext and allowing the schema to be inferred based 
on the data. 
*/
  
val dataWithoutHeader = spark.read.
      option("inferSchema", true).
      option("header", false).
      csv("hdfs:///user/nisha/Data/covtype.data")
val colNames = Seq(
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
      ) ++ (
        (0 until 4).map(i => s"Wilderness_Area_$i")
      ) ++ (
        (0 until 40).map(i => s"Soil_Type_$i")
      ) ++ Seq("Cover_Type")

/* 
Assign column names to the data. Apache Spark ML's Random Forests require the target 
to be of the type double and the labels should start at 0 instead of 1. 
*/
import spark.implicits._
  
val data = dataWithoutHeader.toDF(colNames:_*).
    withColumn("Cover_Type", $"Cover_Type".cast("double")-1)
s"Number of observations: ${data.count()}"

/* 
Display the few sample records in the data. 
*/
  
data.show(5)
  
/* 
Review the distribution of the labels in the "Cover_Type" variable. The number of 
observations falling under each of the categories add up to 581,012. 
*/
  
data.select("Cover_Type").groupBy("Cover_Type").count().
  orderBy($"count".desc).show(10)

/*
Before we begin building a random forest model, we would want to take a second look at
the features. Notice we have several dummy variables in the data - "Wilderness_Area_0-3" 
and "Soil_Type_0-39". It will be more optimal if these are represented by a single 
variable i.e. one variable for Wilderness_Area and one for Soil_Type. This will allow the 
Random Forest model to create decisions based on groups of categories in one decision 
rather than considering each of the dummy variable versions. Below VectorAssembler is 
deployed to combine the 4 and 40 wilderness and soil type columns into two Vector-valued 
columns.
*/
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler}  
import org.apache.spark.ml.linalg.Vector

val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray

val wildernessAssembler = new VectorAssembler().
    setInputCols(wildernessCols).
    setOutputCol("wilderness")

/* 
Defining a UDF that transforms a vector's values into a numeric column that indicates 
the location that has the value 1. 
*/
  
val unhotUDF = udf((vec: Vector) => vec.toArray.indexOf(1.0).toDouble)

val withWilderness = wildernessAssembler.transform(data).
    drop(wildernessCols:_*).
    withColumn("wilderness", unhotUDF($"wilderness"))

val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray

val soilAssembler = new VectorAssembler().
    setInputCols(soilCols).
    setOutputCol("soil")

val data_unencodeOneHot = soilAssembler.transform(withWilderness).
    drop(soilCols:_*).
    withColumn("soil", unhotUDF($"soil"))

/* 
Reviewing a few sample observations observation 
*/
  
data_unencodeOneHot.show(5)
  
/* 
Spark MLlib requires all of the fastures to be collected into one column, whose values 
is a Vector. This is achieved through the VectorAssembler. 
*/
  
val assembler = new VectorAssembler().
    setInputCols(data_unencodeOneHot.columns.filter(_ != "Cover_Type")).
    setOutputCol("featureVector")
val dataAssembled = assembler.transform(data_unencodeOneHot)
dataAssembled.select("featureVector").show(20, truncate=false)
  
/*
Please note that we still want the model to consider these features as "categorical" and 
not "numeric" for which a VectorIndexer will be used.
The VectorIndexer helps index categorical features in datasets of Vectors. It can both
automatically decide which features are categorical and convert original values to 
category indices. 
*/
import org.apache.spark.ml.feature.{VectorIndexer}

val indexer = new VectorIndexer().
    setMaxCategories(40).
    setInputCol("featureVector").
    setOutputCol("indexedVector")
val indexerModel = indexer.fit(dataAssembled)
  
/* 
Create new column "indexedVector" with categorical values transformed to indices 
*/
  
val indexedData = indexerModel.transform(dataAssembled)
indexedData.select("indexedVector").show(20, truncate=false)

/* 
Split the data into 90% train (+ validation), 10% test. Make sure you specify the seed 
value, this will ensure that the results are reproducible later on. 
*/
  
val Array(unencTrainData, unencTestData) = indexedData.
  randomSplit(Array(0.9, 0.1), seed=123)
unencTrainData.cache()
unencTestData.cache()
  
/* 
Setup the RandomForestClassifier to specify the target and feature columns along with
other hyperparameters like measure of impurity - Gini or entropy, number 
of bins and the prediction column. 
*/
import org.apache.spark.ml.classification.{RandomForestClassifier, 
                                           RandomForestClassificationModel}
import scala.util.Random

val classifier = new RandomForestClassifier().
    setSeed(123).
    setLabelCol("Cover_Type").
    setFeaturesCol("indexedVector").
    setPredictionCol("prediction").
    setImpurity("entropy").
    setMaxDepth(20).
    setMaxBins(300)

/* 
It is not always obvious what the parameter values should be. Below we are setting up 
a parameter grid for number of trees ranging from 1 to 10 and minimum info gain from 0 
to 0.05 to help build a better model 
*/  
import org.apache.spark.ml.tuning.{ParamGridBuilder}

val paramGrid = new ParamGridBuilder().
    addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
    addGrid(classifier.numTrees, Seq(1, 5)).
    build()

/* 
Setting up a MulticlassClassificationEvaluator that can compute accuracy and other metrics
that evaluate the quality of the model’s predictions. 
*/
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
  
val multiclassEval = new MulticlassClassificationEvaluator().
    setLabelCol("Cover_Type").
    setPredictionCol("prediction").
    setMetricName("accuracy")

/* 
The TrainValidationSplit below splits the data into 90% training and 10%
validation set and allows us to obtain a more reliable model based on the parameter grid.
It is similar to the Cross Validation split but with k=1 making it less expensive.
*/  
import org.apache.spark.ml.tuning.{TrainValidationSplit}

val validator = new TrainValidationSplit().
    setSeed(123).
    setEstimator(classifier).
    setEvaluator(multiclassEval).
    setEstimatorParamMaps(paramGrid).
    setTrainRatio(0.9)
val validatorModel = validator.fit(unencTrainData)

/* 
In order to query the parameters chosen by the RandomForestClassifier, it’s necessary 
to manually extract the RandomForestClassificationModel. 
*/

val forestModel = validatorModel.bestModel.asInstanceOf[RandomForestClassificationModel]

/* 
Extract parameters and number of trees associated with the bestModel - forestModel 
*/
  
forestModel.extractParamMap
forestModel.getNumTrees

/* 
Printing a representation of the model shows some of its tree structure. It consists of 
a series of nested decisions about features, comparing feature values to thresholds.
*/
  
forestModel.toDebugString
  
/* 
Assess the importance of input features as part of their building process. That is, 
estimate how much each input feature contributes to making correct predictions. Pairing
the importance values with features names, higher the better. 
*/
  
val inputCols = unencTrainData.columns.filter(_ != "Cover_Type")
forestModel.featureImportances.toArray.zip(inputCols).
  sorted.reverse.foreach(println)

/*
The resulting forestModel is itself a Transformer, because it can transform a DataFrame 
containing feature vectors into a DataFrame also containing predictions. For example, 
it might be interesting to see what the model predicts on the "test" data. 
*/
  
val predictions = forestModel.transform(unencTestData)
  
/* 
The "prediction" column below is the model's predicted label and the "probability" 
column is a vector with the individual class probabilities of the "Cover_Type". 
*/
  
predictions.select("Cover_Type", "prediction", "probability").show(5, truncate=false)

/*
Using the MulticlassClassificationEvaluator to compute accuracy to help evaluate the 
quality of the model’s predictions. 
*/  
  
val testAccuracy = multiclassEval.evaluate(forestModel.transform(unencTestData))
testAccuracy

/* 
A confusion matrix can help us derive more insights into the quality of predictions. 
Note, it is a 7*7 matrix with the rows indicating the actual "Cover_Type" and the columns 
indicating the "predicted" values. The values along the diagonal indicate the entries that 
have been classified correctly. An ideal model would have non-zero values as the diagonal 
entries and zero else where. 
*/ 
  
val confusionMatrix = predictions.
    groupBy("Cover_Type").
    pivot("prediction", (0 to 6)).
    count().
    na.fill(0.0).
    orderBy("Cover_Type")
confusionMatrix.show()
  
/* 
Data prep for the variable importance plot. 
*/

val varImp = forestModel.featureImportances.toArray.toSeq
case class treeData(feature: String, importance: Double, index: Double)
val featureNames = unencTrainData.columns.filter(inputCols)
val treeResults = featureNames.indices.map{ i =>
      treeData(featureNames(i), varImp(i), i)
    }.sortBy(-_.importance)

val features = treeResults.map(_.feature)
val featureImportance = treeResults.map(_.importance)
val featureIndex = treeResults.map(_.index)

/*
The figure below displays the relative variable importances for each of the twelve 
predictor variables. Not suprisingly, soil and elevation are the most relevant predictors!
*/
import org.jfree.chart.{ChartFactory}
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.category.{DefaultCategoryDataset}
import java.awt.Color

val dataset = new DefaultCategoryDataset()
for(i <- 0 until featureImportance.length)
  dataset.addValue(featureImportance(i) , features(i), "")
val barChart = ChartFactory.createBarChart(
  "Feature Importance Plot",      // chart title
  "Feature",                      // domain axis label
  "Feature Importance Measure",   // range axis label
  dataset,
  PlotOrientation.VERTICAL,
  true, false, false)             // legend, tool-tip, url
barChart.getPlot().setBackgroundPaint(Color.WHITE)
barChart.setBorderVisible(false)
import org.jfree.chart.ChartUtilities
import java.io.File
val varImpChart = new File( "./images/Ch04RandomForest.jpeg" )
ChartUtilities.saveChartAsJPEG(varImpChart, barChart, 1600, 600)
/*
![alt text](./images/Ch04RandomForest.jpeg)
*/
   