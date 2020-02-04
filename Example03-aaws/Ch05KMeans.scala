/*
# Anomaly Detection in Network Traffic with K-means Clustering

_Adapted from Chapter 5 of
[Advanced Analytics with Spark](http://shop.oreilly.com/product/0636920035091.do)
from O'Reilly Media.
Original source code at https://github.com/sryza/aas/tree/master/ch05-kmeans 
Source repository for this code: https://github.com/srowen/aaws-cdsw-examples _

*/

/*
## Example

This example will demonstrate usage of Spark MLlib and k-means clustering
for simple anomaly detection.

## Data Set

This example will use the [KDD Cup 1999 Data](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
which contains information 4.9M network sessions. The data set records
login attemps, bytes sent, etc. It is also actually labeled; each
session is tagged as `normal` or an instance of one of several possible
network intrustion types, like `smurf`.

First, we read this as a CSV file into a `DataFrame`.
Note that the column names need to be specified manually in order to
be used in the `DataFrame` because the CSV does not contain a header.
We also partition into more partitions than usual here for better
parallelism on a cluster.
 */
  
val data = spark.read.
  option("inferSchema", true).
  option("header", false).
  csv("hdfs:///user/sowen/aaws-examples/ch05/kddcup.data").
  toDF(
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label").
  repartition(24)

/*
Cache the data, because it will be used over and over.
 */
data.cache()
  
/*
Try counting the data, to verify all is working.
 */
data.count()

  
/*
## Clustering, Take 0

In this first attempt, we'll just directly cluster the data and 
see what happens.
 */

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import scala.util.Random

import spark.implicits._

/*
First, look at the distribution of labels by count. There are lots of
`smurf` and `neptune` attacks in the data.
 */
data.select("label").groupBy("label").count().
  orderBy($"count".desc).show(25)

/*
k-means clustering can only operate on numeric data, so retain only
numeric columns for now.
 */
val numericOnly = data.drop("protocol_type", "service", "flag").cache()

/*
Build a pipeline that will convert the data into a feature vector,
cluster with k-means, and then print the resulting cluster centers.
Note that the label is not used as input.
 */

val assembler = new VectorAssembler().
  setInputCols(numericOnly.columns.filter(_ != "label")).
  setOutputCol("featureVector")

val kmeans = new KMeans().
  setSeed(Random.nextLong()).
  setPredictionCol("cluster").
  setFeaturesCol("featureVector")

val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
val pipelineModel = pipeline.fit(numericOnly)
val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

kmeansModel.clusterCenters.foreach(println)

/*
The centroids are points in space and don't mean much to us, except
for the noticeable fact that there are only two.

Try clustering each point and then counting how many of each label
appear in each cluster.
 */

val withCluster = pipelineModel.transform(numericOnly)

withCluster.select("cluster", "label").
  groupBy("cluster", "label").count().
  orderBy($"cluster", $"count".desc).
  show(25)

numericOnly.unpersist()
  
/*
All but 4 points landed in cluster 1! everything else went in cluster 0.
This didn't work. $k=2$ clusters, the default, isn't nearly enough to
capture the structure of the points.
 */
  
  
/*
## Clustering, Take 1

We need to pick a better $k$, but, how would we even know a good clustering?
A good clustering picks cluster centers such that all points fall close
to a cluster center. We could measure sum of squared distance of each point
to its cluster center as a 'cost' and then pick a $k$ that minimizes it.
 */
  
import org.apache.spark.sql.DataFrame

def clusteringScore0(data: DataFrame, k: Int): Double = {
  val assembler = new VectorAssembler().
    setInputCols(data.columns.filter(_ != "label")).
    setOutputCol("featureVector")

  val kmeans = new KMeans().
    setSeed(Random.nextLong()).
    setK(k).
    setPredictionCol("cluster").
    setFeaturesCol("featureVector")

  val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

  val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
  kmeansModel.computeCost(assembler.transform(data)) / data.count()
}

val numericOnly = data.drop("protocol_type", "service", "flag").cache()

/*
Try several values of $k$ and output their costs.
 */

(20 to 100 by 40).map(k => (k, clusteringScore0(numericOnly, k))).foreach(println)

/*
Larger values of $k$ usually produce smaller costs, but of course they do: in the
limit, where $k$ is the number of data points, every point is a cluster and has 
cost 0! Further, larger $k$ should always yield lower cost, but this isn't
necessarily true in the output.

To fix the latter, we can try again but let it clustering run for more iterations
and run longer to convergence.
 */

def clusteringScore1(data: DataFrame, k: Int): Double = {
  val assembler = new VectorAssembler().
    setInputCols(data.columns.filter(_ != "label")).
    setOutputCol("featureVector")

  val kmeans = new KMeans().
    setSeed(Random.nextLong()).
    setK(k).
    setPredictionCol("cluster").
    setFeaturesCol("featureVector").
    setMaxIter(40).
    setTol(1.0e-5)

  val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

  val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
  kmeansModel.computeCost(assembler.transform(data)) / data.count()
}

(20 to 100 by 40).map(k => (k, clusteringScore1(numericOnly, k))).foreach(println)

numericOnly.unpersist()

/*
## Clustering, Take 2

One problem is that the various features are not on the same scale. Some
are 0/1 indicator variables, some are ratio between 0 and 1, and some are 
counts of _bytes_, which can run into the tens of thousands. 

The latter values are dominating the clustering computation because 
almost all of the distance between points comes from this "dimension".

To remedy this, we can try again, but scale each dimension onto roughly
the same scale by dividing each by its standard deviation.
 */  

import org.apache.spark.ml.feature.StandardScaler

def clusteringScore2(data: DataFrame, k: Int): Double = {
  val assembler = new VectorAssembler().
    setInputCols(data.columns.filter(_ != "label")).
    setOutputCol("featureVector")

  val scaler = new StandardScaler().
    setInputCol("featureVector").
    setOutputCol("scaledFeatureVector").
    setWithStd(true).
    setWithMean(false)

  val kmeans = new KMeans().
    setSeed(Random.nextLong()).
    setK(k).
    setPredictionCol("cluster").
    setFeaturesCol("scaledFeatureVector").
    setMaxIter(40).
    setTol(1.0e-5)

  val pipeline = new Pipeline().setStages(Array(assembler, scaler, kmeans))
  val pipelineModel = pipeline.fit(data)

  val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
  kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
}

val numericOnly = data.drop("protocol_type", "service", "flag").cache()

(90 to 270 by 60).map(k => (k, clusteringScore2(numericOnly, k))).foreach(println)

numericOnly.unpersist()
  
  
/*
## Clustering, Take 3

The result is beginning to make more sense. However, so far we have ignored
three features because they are non-numeric. It is possible to encode a non-numeric
value that takes on $N$ possible values as a series of $N$ 0/1 indicator values.
For example, an _animal_ feature that takes on values "cat", "dog" and "mouse"
could be encoded with 3 values: `isCat`, `isDog`, `isMouse`, exactly one of which
is 1, with the rest 0. This is called one-hot encoding.

The existing pipeline can be augmented once again to reincorporate these
ignored values.
 */

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

def oneHotPipeline(inputCol: String): (Pipeline, String) = {
  val indexer = new StringIndexer().
    setInputCol(inputCol).
    setOutputCol(inputCol + "_indexed")
  val encoder = new OneHotEncoder().
    setInputCol(inputCol + "_indexed").
    setOutputCol(inputCol + "_vec")
  val pipeline = new Pipeline().setStages(Array(indexer, encoder))
  (pipeline, inputCol + "_vec")
}

def clusteringScore3(data: DataFrame, k: Int): Double = {
  val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
  val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
  val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

  // Original columns, without label / string columns, but with new
  // vector encoded cols
  val assembleCols = Set(data.columns: _*) --
    Seq("label", "protocol_type", "service", "flag") ++
    Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
  val assembler = new VectorAssembler().
     setInputCols(assembleCols.toArray).
    setOutputCol("featureVector")

  val scaler = new StandardScaler().
    setInputCol("featureVector").
    setOutputCol("scaledFeatureVector").
    setWithStd(true).
    setWithMean(false)

  val kmeans = new KMeans().
    setSeed(Random.nextLong()).
    setK(k).
    setPredictionCol("cluster").
    setFeaturesCol("scaledFeatureVector").
    setMaxIter(40).
    setTol(1.0e-5)

  val pipeline = new Pipeline().setStages(Array(
    protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
  val pipelineModel = pipeline.fit(data)

  val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
  kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
}

(90 to 270 by 60).map(k => (k, clusteringScore3(data, k))).foreach(println)


/*
## Clustering, Take 4

So far, the problem of choosing $k$ has been to choose the one that yields
the best intra-cluster distance. This is a simple, common metric, yet in this
case there is additional information available: the label for each event.

It won't be possible to use this information directly in the clustering,
because of course label would never be known ahead of time for future data. 
However, if some or all points are labeled, as here, it could be used to
create an alternative measure of clustering quality.

We can use the notion of 
[entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)), which
can capture the level of "mixedness" in a group things. Here we'll use it
to measure how uniform, or varied, each clusters' labels are. Because we assume
that events with the same label are similar and should probably be in the
same clusters, we'll say a clustering is good when the labels found in each
cluster are uniform (low entropy), and bad when there are many different labels
(high entropy)
 */

def entropy(counts: Iterable[Int]): Double = {
  val values = counts.filter(_ > 0)
  val n = values.map(_.toDouble).sum
  values.map { v =>
    val p = v / n
    -p * math.log(p)
  }.sum
}

/*
If a cluster has $n$ elements, and has $l_i$ instances of each label $i$ that
appears in the cluster, then define $p_i = l_i / n$ to be the proportion of 
label $i$ in the cluster. Then the overall entropy of the labels is 

$I = -\sum_i p_i \log p_i$
 */

import org.apache.spark.ml.PipelineModel

def fitPipeline4(data: DataFrame, k: Int): PipelineModel = {
  val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
  val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
  val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

  // Original columns, without label / string columns, but with new
  //vector encoded cols
  val assembleCols = Set(data.columns: _*) --
    Seq("label", "protocol_type", "service", "flag") ++
    Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
  val assembler = new VectorAssembler().
    setInputCols(assembleCols.toArray).
    setOutputCol("featureVector")

  val scaler = new StandardScaler().
    setInputCol("featureVector").
    setOutputCol("scaledFeatureVector").
    setWithStd(true).
    setWithMean(false)

  val kmeans = new KMeans().
    setSeed(Random.nextLong()).
    setK(k).
    setPredictionCol("cluster").
    setFeaturesCol("scaledFeatureVector").
    setMaxIter(40).
    setTol(1.0e-5)

  val pipeline = new Pipeline().setStages(Array(
    protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
  pipeline.fit(data)
}

def clusteringScore4(data: DataFrame, k: Int): Double = {
  val pipelineModel = fitPipeline4(data, k)

  // Predict cluster for each datum
  val clusterLabel = pipelineModel.transform(data).
    select("cluster", "label").as[(Int, String)]
  val weightedClusterEntropy = clusterLabel.
    // Extract collections of labels, per cluster
    groupByKey { case (cluster, _) => cluster }.
    mapGroups { case (_, clusterLabels) =>
      val labels = clusterLabels.map { case (_, label) => label }.toSeq
      // Count labels in collections
      val labelCounts = labels.groupBy(identity).values.map(_.size)
      labels.size * entropy(labelCounts)
    }.collect()

  // Average entropy weighted by cluster size
  weightedClusterEntropy.sum / data.count()
}

(90 to 270 by 60).map(k => (k, clusteringScore4(data, k))).foreach(println)

/*
Your results may vary, but you may find that there is a local minimum near 
about $k=180$. It's not true that entropy will decrease with $k$, so it's 
reasonable to look to this value as a likely good choice.

Of course, the point wasn't to choose $k$, but to build a clustering. This
can now be done with $k=180$.
*/
  
val pipelineModel = fitPipeline4(data, 180)
  
/*
We can print, as before, clusters with count of labels found within
the clusters.
 */

val countByClusterLabel = pipelineModel.transform(data).
  select("cluster", "label").
  groupBy("cluster", "label").count().
  orderBy("cluster", "label")

countByClusterLabel.show()
  
/*
This shows a much more reasonable distribution of labels. Clusters tend
to have a high concentration of one label type each.

Finally, it's possible to use the model for anomaly detection. New points
that lie far from any cluster will be considered anomalous. To establish how
far is 'too far', we might simply rank the input points by distance from 
nearest cluster centroid, and take, perhaps, the 100th-largest as a threshold.
 */

import org.apache.spark.ml.linalg.{Vector, Vectors}

val kMeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
val centroids = kMeansModel.clusterCenters

val clustered = pipelineModel.transform(data)
val threshold = clustered.
  select("cluster", "scaledFeatureVector").as[(Int, Vector)].
  map { case (cluster, vec) => Vectors.sqdist(centroids(cluster), vec) }.
  orderBy($"value".desc).take(100).last

/*
Finally it should be possible to detect 'anomalies', here even in the 
existing input, according to this definition.
 */
  
val originalCols = data.columns
val anomalies = clustered.filter { row =>
  val cluster = row.getAs[Int]("cluster")
  val vec = row.getAs[Vector]("scaledFeatureVector")
  Vectors.sqdist(centroids(cluster), vec) >= threshold
}.select(originalCols.head, originalCols.tail:_*)

/*
This is the most-anomalous point in the input -- why?
 */

anomalies.first()
  
/*
A network security expert would be more able to interpret 
why this is or is not actually a strange connection. 
It appears unusual at least because it is labeled `normal`, 
but involved connections to 69 different hosts.
 */
