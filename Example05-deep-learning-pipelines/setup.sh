cd deep-learning-pipelines
conda create -n sparkdl --copy -y python=3.6
source activate sparkdl
conda install -yq --file requirements.txt


ln -s ~/.conda/envs/sparkdl/ sparkdl &&
zip -r sparkdl_env.zip sparkdl &&
rm sparkdl

curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
rm flower_photos.tgz
hadoop fs -put flower_photos
rm -r flower_photos