#!/usr/bin/env bash

hdfs dfs -mkdir /tmp/Covtype
curl https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz | gunzip | hdfs dfs -put - /tmp/Covtype/covtype.data
