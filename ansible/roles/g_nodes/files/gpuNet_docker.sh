#!/bin/sh

# Git clone
git clone https://github.com/NVIDIA/DeepLearningExamples.git

# Set working directory
cd DeepLearningExamples/PyTorch/Classification/GPUNet/

# Create directory for logs and outputs
mkdir -p logs && mkdir -p output

# Download the data
mkdir data && cd data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar  # get ILSVRC2012_img_val.tar (about 6.3 GB)
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar # get ILSVRC2012_img_train.tar (about 138 GB)

# Extract the training data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; >
cd ..

# Extract the validation data and move images to subfolders:
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

cd ..
rm *.tar

# Build and run the docker
docker build -t gpunet .
docker run --gpus all --rm --network=host --shm-size 600G --ipc=host -v ${PWD}/data:/root/data/imagenet/ gpunet ./train.sh $(nvidia-smi --list-gpus | wc -l) /root/data/imagenet/ --model gpunet_0 --sched step --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf -b 192 --epochs 5 --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .06 --num-classes 1000 --enable-distill False --crop-pct 1.0 --img-size 320 --amp
