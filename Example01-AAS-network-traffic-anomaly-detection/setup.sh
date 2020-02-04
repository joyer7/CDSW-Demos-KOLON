hdfs dfs -mkdir /tmp/KDDCup
curl http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz | gunzip | hdfs dfs -put - /tmp/KDDCup/kddcup.data
