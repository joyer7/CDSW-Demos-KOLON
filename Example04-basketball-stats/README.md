# BasketballStats
Basketball Statistics Demo<br>
Created by Jordan Volz (jordan.volz@cloudera.com)

<b>Status</b>: Demo Ready<br>
<b>Use Case</b>: End-to-end spark workflow: data processing, ad-hoc analytics, and predictive analytics

<b>Steps</b>:<br>
1. Open a CDSW terminal and run setup.sh<br>
2. Create a Scala Session and run data-processing.scala<br>
3. Create a Python Session and run analysis.py<br>
4. Return to the Scala Session and run machine-learning.scala<br>
5. When finished, run cleanup.scala in your spark session and cleanup.sh in the terminal<br>

<b>Recommended Session Sizes</b>: 4 CPU, 8 GB RAM

<b>Recommended Jobs/Pipeline</b>:<br>
data-processing.scala --> analysis.py --> machine-learning.scala

<b>Notes</b>: <br>
1. Raw stats are in /data<br>
2. data-processing.scala --> data transformations + table creations<br>
3. analysis.py --> ad-hoc analysis with pandas<br>
4. machine-learning.scala --> Regression analysis with spark mllib<br>
5. Your user will need write access into Hive. <br>

<b>Estimated Runtime</b>: <br>
data-processing.scala --> approx 1 min <br>
analysis.py --> < 1 min <br>
machine-learning.scala --> approx 30 min <br>

<b>Demo Script</b><br>
http://github.mtv.cloudera.com/foe/BasketballStats/blob/master/BasketballStatsDemoScript.docx

<b>Related Content</b>:<br>
http://blog.cloudera.com/blog/2016/06/how-to-analyze-fantasy-sports-using-apache-spark-and-sql/ <br>
http://blog.cloudera.com/blog/2016/06/how-to-analyze-fantasy-sports-with-apache-spark-and-sql-part-2-data-exploration/
