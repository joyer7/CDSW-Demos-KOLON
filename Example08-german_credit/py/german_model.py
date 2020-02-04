## Imports
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import Row

## Constants
german_cfi = {0:4,5:5,6:5,7:4,8:4,9:3,10:4,11:4,13:3,14:3,15:4,16:4,17:2,18:2,19:2}

## Functions
def german_lp(x):
    vals=x.asDict()
    label=vals['cred']
    feats=[vals['acct_bal']-1,
           vals['dur_cred'],
           vals['pay_stat'],
           vals['purpose'],
           vals['cred_amt'],
           vals['value']-1,
           vals['len_emp']-1,
           vals['install_pc']-1,
           vals['sex_married']-1,
           vals['guarantors']-1,
           vals['dur_addr']-1,
           vals['max_val']-1,
           vals['age'],
           vals['concurr']-1,
           vals['typ_aprtmnt']-1,
           vals['no_creds']-1,
           vals['occupation']-1,
           vals['no_dep']-1,
           vals['telephone']-1,
           vals['foreign_wkr']-1]
    return LabeledPoint(label, feats)

def train_decision_tree(lp,german_cfi):
    return DecisionTree.trainClassifier(lp, numClasses=2,
                                        categoricalFeaturesInfo=german_cfi,
                                        impurity='gini',
                                        maxDepth=3, 
                                        maxBins=5)

## Main functionality

def main(sc):
   dat = sqlCtx.sql('SELECT * FROM german')
   lp = dat.rdd.filter(lambda x: x.cred).map(german_lp).cache()
   predictions = model.predict(lp.map(lambda x: x.features))
   labelsAndPredictions = lp.map(lambda lp: lp.label).zip(predictions)
   trainErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(lp.count())

if __name__ == "__main__":
    conf = SparkConf().setAppName("german_spark_submit")
    conf = conf.setMaster("yarn cluster")
    sc   = SparkContext(conf=conf)
    main(sc)