#!/bin/sh

# git clone
git clone https://github.com/NVIDIA/DeepLearningExamples.git

# set working directory
cd DeepLearningExamples/TensorFlow/Detection/SSD/

# build docker
docker build . -t nvidia_ssd

# Get COCO 2017 data sets
CONTAINER="nvidia_ssd"
COCO_DIR="./data/coco2017_tfrecords"
CHECKPOINT_DIR="./checkpoints"
mkdir -p $COCO_DIR
chmod 777 $COCO_DIR

# Download backbone checkpoint
mkdir -p $CHECKPOINT_DIR
chmod 777 $CHECKPOINT_DIR
cd $CHECKPOINT_DIR
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xzf resnet_v1_50_2016_08_28.tar.gz
mkdir -p resnet_v1_50
mv resnet_v1_50.ckpt resnet_v1_50/model.ckpt

# Create TFRecords
bash ${PWD}/models/research/object_detection/dataset_tools/download_and_preprocess_mscoco.sh \
    ${PWD}/data/coco2017_tfrecords

# run docker
docker run --gpus all --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/data/coco2017_tfrecords:/data/coco2017_tfrecords -v ${PWD}/checkpoints:/checkpoints --ipc=host nvidia_ssd bash ./examples/SSD320_FP16_8GPU.sh ./checkpoints
