## NLP example: Wordcloud of "alice in wonder land"
# Basic code is taken from https://github.com/amueller/word_cloud/blob/master/examples/masked.py

from pyspark.sql import SparkSession
from wordcloud import WordCloud, STOPWORDS

spark = SparkSession.builder \
      .appName("Word count") \
      .getOrCreate()
    
threshold = 5

# Download from http://www.umich.edu/~umfandsf/other/ebooks/alice30.txt
text_file = spark.sparkContext.textFile("/tmp/alice30.txt")

stopwords = set(STOPWORDS)
stopwords.add("said")

counts = text_file.flatMap(lambda line: line.split(" ")) \
             .filter(lambda word: word not in stopwords) \
             .map(lambda word: (word.lower(), 1)) \
             .reduceByKey(lambda a, b: a + b)

from pyspark.sql.types import *
schema = StructType([StructField("word", StringType(), True),
                     StructField("frequency", IntegerType(), True)])

filtered = counts.filter(lambda pair: pair[1] >= threshold)
counts_df = spark.createDataFrame(filtered, schema)

frequencies = counts_df.toPandas().set_index('word').T.to_dict('records')

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 
alice_mask = np.array(Image.open(path.join("wordcloud_alice/resources", "alice-mask.jpg")))

wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask,
               stopwords=stopwords)
wc.generate_from_frequencies(dict(*frequencies))

plt.imshow(wc, interpolation='bilinear')

# Can't overlay mask figure
#plt.axis("off")
#plt.figure()
#plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
#plt.axis("off")
#plt.show()
