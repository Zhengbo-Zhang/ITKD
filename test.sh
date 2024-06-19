#!/bin/bash

# conda activate
source /mnt/data1/zyx/miniconda3/bin/activate ITKD

export CUDA_VISIBLE_DEVICES=7

export TMPDIR=`pwd`/tmp

# good ckpt path
# 1. reward redistribution + correct batch update +    outputs/cifar100/rl-rs56-rs20/distillation/vz6641uh/checkpoints/distiller-epoch=239-val_acc=0.7187.ckpt


# run
# test
python test.py -c config/cifar100/rl/test.yaml \
        -t 'ITKD+dkd test config' \
        -v "outputs/cifar100/rl-rs56-rs20/distillation/vz6641uh/checkpoints/distiller-epoch=239-val_acc=0.7187.ckpt"

# ITKD+dkd rs56-rs20
# python main.py -c config/cifar100/rl/rl-dkd-rs56-rs20.yaml \
#         -t 'ITKD+dkd best config'

# ITKD+dkd rs110-rs20
# python main.py -c config/cifar100/rl/rl-dkd-rs110-rs20.yaml \
#         -t 'ITKD+dkd best config'


# ITKD+similarity rs56-rs20
# python main.py -c config/cifar100/rl/rl-sim-rs56-rs20.yaml \
#         -t 'ITKD+similarity best config'