# TensorFlow on Cloudera Data Science Workbench

This short demo shows how to use TensorFlow to run single-node training on a cluster
edge node using Cloudera Data Science Workbench.

The cases of running single node training on the cluster, and distributed training on
the cluster are not covered here.

## Preliminaries

Install TensorFlow:

```
! pip install tensorflow
%cd tensorflow-demo
```

## Reading data from local files

This is the simplest case, where training data is loaded onto the local file system
of the edge node, and TensorFlow reads it during training.

The TensorFlow
[beginners tutorial](https://www.tensorflow.org/get_started/mnist/beginners)
takes this approach; it downloads the MNIST dataset, trains a neural net using the
training data, then evaluates it using the test data.

Start a Python session in Cloudera Data Science Workbench, then
type the following in the console to run the program.

```
! python mnist_softmax.py
```

You can also run the example interactively by opening *mnist_softmax_interactive.py*
and running it (using the Run menu to run selected lines, or the whole file).

## Reading data from HDFS

For simplicity of exposition, the beginners tutorial automatically downloads
the MNIST dataset and uses TensorFlow's
[feeding method](https://www.tensorflow.org/programmers_guide/reading_data) of
reading data. It's more likely that you'll want to read data from files in HDFS
for your TensorFlow programs, although this is a little more complicated.

The first step is to get the data in a format that TensorFlow can read. There
are various options, but the standard format is a
[TFRecords file of ProtoBufs](https://www.tensorflow.org/programmers_guide/reading_data#standard_tensorflow_format).

Type the following to convert the MNIST data into this format:

```
! python convert_to_records.py
! hadoop fs -mkdir mnist
! hadoop fs -put /tmp/data/*.tfrecords mnist
```

The converted files are stored under _/tmp/data_ on the local file system,
then copied to HDFS.

The next step is to train a model using these files. This can be achieved with
the [fully_connected_reader.py](https://www.tensorflow.org/code/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py)
program.

```
%env HADOOP_HDFS_HOME=/opt/cloudera/parcels/CDH/lib/hadoop-hdfs
%env LD_LIBRARY_PATH=/opt/cloudera/parcels/CDH/lib/hadoop/lib/native:/opt/cloudera/parcels/GPLEXTRAS/lib/hadoop/lib/native:/opt/cloudera/parcels/CDH/lib64/:/usr/java/jdk1.8.0_40-cloudera/jre/lib/amd64/server

! CLASSPATH=$(hadoop classpath --glob) python fully_connected_reader.py --train_dir hdfs://bottou02.sjc.cloudera.com/user/$HADOOP_USER_NAME/mnist
```

Notice that you need to set several environment variables for TensorFlow to connect to HDFS.
These are covered on the [How to run TensorFlow on Hadoop](https://www.tensorflow.org/deploy/hadoop) page.