#!/bin/sh
# git clone
git clone https://github.com/NVIDIA/DeepLearningExamples.git

# set working directory
cd DeepLearningExamples/TensorFlow/Detection/SSD/

# create directory for logs and outputs
mkdir -p data && mkdir -p checkpoint

# run the script
./download_dataset.sh

# build docker
docker build -t gpunet .
docker run --gpus all --rm --network=host --shm-size 600G --ipc=host -v /usr/local/pytorch_GPUNET/DeepLearningExamples/PyTorch/Classification/GPUNet/data:/root/data/imagenet/ gpunet ./train.sh $(nvidia-smi --list-gpus | wc -l) /root/data/imagenet/ --model gpunet_0 --sched step --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf -b 192 --epochs 5 --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --lr .06 --num-classes 1000 --enable-distill False --crop-pct 1.0 --img-size 320 --amp
