#!/bin/bash

wget http://www.umich.edu/~umfandsf/other/ebooks/alice30.txt -P /tmp
hdfs dfs -put /tmp/alice30.txt /tmp/
mkdir -p wordcloud_alice/resources
wget "http://www.clipartbest.com/cliparts/niB/RKA/niBRKARMT.jpg" -O wordcloud_alice/resources/alice-mask.jpg

pip install -r wordcloud_alice/requirements.txt -c wordcloud_alice/constraints.txt
