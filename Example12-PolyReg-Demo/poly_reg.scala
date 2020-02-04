/* # Polynomial Regression - scala, Spark ML
The scala instances in CDSW currently initialize differently than R & python. Specifically, by default it
initializes a spark context. To demonstrate, we'll use a simple [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression).
... to confirm the existance of our sc: */

sc

/* ## Generate Spark DataFrame Data
 We'll generate sample data for a multivariate linear regression with known coefficients and randomly generated error. Specifically;
 $$ y = \beta_0 + \sum_i (\beta_i x_i) + \epsilon   \thinspace \thinspace \thinspace \thinspace \thinspace     \forall i \in {1..3}$$
 $$ \beta_0: 4 $$
 $$ \beta_1: 6 $$
 $$ \beta_2: 2 $$
 $$ \beta_3: -1 $$ */

import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression

val numExamples = 10000
val x1:      RDD[Double] = RandomRDDs.uniformRDD(sc, numExamples).map( x => 6.0*x-2 )
val epsilon: RDD[Double] =  RandomRDDs.normalRDD(sc, numExamples).map( x => 4.0*x )

def gen_poly(x: Tuple2[Double, Double]): Tuple2[Double, org.apache.spark.ml.linalg.Vector] = {
    var x1:Double = x._1
    var x2:Double = scala.math.pow(x1,2)
    var x3:Double = scala.math.pow(x1,3)
    var  X        = Vectors.dense(x1,x2,x3)
    var epsilon:Double = x._2
    var  y:Double = 4.0 + 6.0*x1 + 2.0*x2 - 1*x3 + epsilon
    return (y,X)
}

val zip_dat = x1.zip(epsilon)
val gen_dat = zip_dat.map( x => gen_poly(x))

/* ## Prepare Data as DataFrame & Train Model
 This example will use the ML class which requires our data to be structured as a spark DataFrame.
 Notice that we will be using the first column as our label (Double),and we will use our second column as our features (Vectors.dense([Double])).
*/

val dat = spark.createDataFrame(gen_dat)

val lr = new LinearRegression().setLabelCol("_1").setFeaturesCol("_2")

val lrModel = lr.fit(dat)

// Our results are consistant with the know values of the intercept and coefficients:  //

lrModel.intercept
lrModel.coefficients

/* ## Collect and Plot Data
Plotting in scala is slightly more cumbersome since there is currently not pre-configured
method using to [Toree](https://toree.incubator.apache.org/documentation/user/using-with-jupyter-notebooks.html).
However, since markdown is enabled, we'll first save the graphic to file, then use the markdown
syntax ***\!\[<alt text>\]\(<file path>\)***

**NOTE:** Since this is a local reference, you will have to change to a global reference if you want to 
have the graphic appear in a session share view.

**NOTE:** There is an alternate method of adding libraies by including them in you spark-configs.conf in your [project settings](../settings/environment) page.

**TODO:** Add regression fit line to plot
*/

%AddDeps org.jfree jfreechart 1.0.19 --transitive 
%AddDeps org.jfree jcommon 1.0.24 --transitive 

import org.jfree.chart.ChartFrame
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.ChartFactory
import org.jfree.data.xy.{DefaultXYDataset}

val y = gen_dat.map( x => x._1 ).collect()
val x = gen_dat.map( x => x._2(0) ).collect()

val dataset = new DefaultXYDataset
dataset.addSeries("Poly Observations",Array(x,y))

val scatterPlot =  ChartFactory.createScatterPlot(
    "Plot",
    "Y Label",
    "X Label",
    dataset,
    PlotOrientation.VERTICAL,
    true,false,false
)

import org.jfree.chart.ChartUtilities
import java.io.File

val scatterChart = new File( "./img/scatterPlot3.jpeg" )
ChartUtilities.saveChartAsJPEG(scatterChart, scatterPlot, 800, 600)

/* ![alt text](./img/scatterPlot3.jpeg) */
