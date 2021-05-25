#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
echo "Training..."

# Multi Device
python train_imagenet.py \
--dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
-d mnist -a my_model -b 256 -j 2 -c 10 --epoch 20 ./data/datasets/ --lr-scheduler imagenet \
#> ./data/logs/log_my_model_mnist_multi_device.txt

# Single Device
python train_imagenet.py \
-d mnist -a my_model -b 256 -j 2 -c 10 --epoch 20 ./data/datasets/ --lr-scheduler imagenet \
#> ./data/logs/log_my_model_mnist_single_device.txt

echo "done."