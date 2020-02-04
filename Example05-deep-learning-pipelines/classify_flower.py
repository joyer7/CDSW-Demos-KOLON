import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import sys, os

%cd deep-learning-pipelines/

spark = SparkSession\
    .builder\
    .appName("DL")\
    .getOrCreate()

sys.path.append(os.path.expanduser("~/.ivy2/jars/databricks_spark-deep-learning-0.3.0-spark2.1-s_2.11.jar"))
sys.path.append(os.path.expanduser("~/.ivy2/jars/databricks_tensorframes-0.2.9-s_2.11.jar"))
sys.path.append(os.path.expanduser("~/.conda/envs/sparkdl/lib/python3.6/site-packages/"))
from sparkdl import readImages

img_dir = "flower_photos"
tulips_df = readImages(img_dir + "/tulips").withColumn("label", lit(1))
daisy_df = readImages(img_dir + "/daisy").withColumn("label", lit(0))

# downsample to make the file run faster
sampling_ratio = 0.2
tulips_train, tulips_test = tulips_df.sample(withReplacement=False, fraction=sampling_ratio).randomSplit([0.6, 0.4])
daisy_train, daisy_test = daisy_df.sample(withReplacement=False, fraction=sampling_ratio).randomSplit([0.6, 0.4])
train_df = tulips_train.unionAll(daisy_train)
test_df = tulips_test.unionAll(daisy_test)

# Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
# This ensure that each of the paritions has a small size.
train_df = train_df.repartition(100)
test_df = test_df.repartition(100)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer


# Extract features from the pre-trained model
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")

# In practice, we'd probably save the featurized data to HDFS here, then
# we could quickly run logistic regression with multiple hyper-parameters
# to find the best model
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])

p_model = p.fit(train_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
tested_df = p_model.transform(test_df)
tested_df.cache()
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(tested_df.select("prediction", "label"))))

import matplotlib.pyplot as plt
def display_predictions(df, n):
  rows = df.select("image", "label", "probability").take(n)
  fig, axs = plt.subplots(1, len(rows), figsize=(12, 5))
  axs = np.array(axs).ravel()
  for i, row in enumerate(rows):
    im_data = row[0]
    channels = im_data[0]
    bts = type(im_data[4])
    image_np = np.frombuffer(im_data[4], dtype=np.uint8).reshape((im_data[1], im_data[2], im_data[3]))
    axs[i].imshow(image_np)
    axs[i].set_title("Daisy prob = %0.3f" % (row[2][0]))

# Daisies classified as tulips
display_predictions(tested_df.filter("prediction != label and label == 0"), 4)

# Tulips classified as daisies
display_predictions(tested_df.filter("prediction != label and label == 1"), 4)

# Daisies classified as daisies
display_predictions(tested_df.filter("prediction == label and label == 0"), 4)

# Tulips classified as tulips
display_predictions(tested_df.filter("prediction == label and label == 1"), 4)

tested_df.unpersist()