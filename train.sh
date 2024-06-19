#!/bin/bash

# conda activate
source /mnt/data3/zyx/miniconda3/bin/activate ITKD

device=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$device

echo "==> use gpu device id ${device}"

export TMPDIR=`pwd`/tmp

# run
# cifar100

# ITKD+dkd rs32x4-rs8x4
python main.py -c config/cifar100/rl/rl-dkd-rs32x4-rs8x4.yaml \
        -t 'ITKD+dkd best config'

# ITKD kd rs56-rs20
# python main.py -c config/cifar100/rl/rl-rs56-rs20.yaml \
#         -t 'ITKD+kd best config'

# ITKD+dkd vgg13-vgg8
# python main.py -c config/cifar100/rl/rl-dkd-vgg13-vgg8.yaml \
#         -t 'ITKD+dkd best config'


# ITKD+kd rs56-rs20
# python main.py -c config/cifar100/rl/rl-rs56-rs20.yaml \
#         -t 'ITKD+kd best config'

# ITKD+similarity rs56-rs20
# python main.py -c config/cifar100/rl/rl-sim-rs56-rs20.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+similarity wrn40_2-wrn40_1
# python main.py -c config/cifar100/rl/rl-sim-wrn40_2-wrn40_1.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+similarity wrn40_2-wrn16_2
# python main.py -c config/cifar100/rl/rl-sim-wrn40_2-wrn16_2.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+similarity rs32x4-rs8x4
# python main.py -c config/cifar100/rl/rl-sim-rs32x4-rs8x4.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+similarity vgg13-vgg8
# python main.py -c config/cifar100/rl/rl-sim-vgg13-vgg8.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+srrl rs32x4-rs8x4
# python main.py -c config/cifar100/rl/rl-srrl-rs32x4-rs8x4.yaml \
#         -t 'ITKD+srrl best config'

# ITKD+srrl rs56-rs20
# python main.py -c config/cifar100/rl/rl-srrl-rs56-rs20.yaml \
#         -t 'ITKD+srrl best config'

# ITKD+srrl rs110-rs20
# python main.py -c config/cifar100/rl/rl-srrl-rs110-rs20.yaml \
#         -t 'ITKD+srrl best config'

# ITKD+srrl rs32x4-rs8x4
# python main.py -c config/cifar100/rl/rl-srrl-rs32x4-rs8x4.yaml \
#         -t 'ITKD+srrl best config'

# ITKD+srrl wrn40_2-wrn40_1
# python main.py -c config/cifar100/rl/rl-srrl-wrn40_2-wrn40_1.yaml \
#         -t 'ITKD+srrl best config'

# ITKD+srrl wrn40_2-wrn16_2
# python main.py -c config/cifar100/rl/rl-srrl-wrn40_2-wrn16_2.yaml \
#         -t 'ITKD+srrl best config'

# ITKD+srrl vgg13-vgg8
# python main.py -c config/cifar100/rl/rl-srrl-vgg13-vgg8.yaml \
#         -t 'ITKD+srrl best config'

# ITKD+dkd rs110-rs20
# python main.py -c config/cifar100/rl/rl-dkd-rs110-rs20.yaml \
#         -t 'ITKD+dkd best config'

# ITKD+similarity rs110-rs20
# python main.py -c config/cifar100/rl/rl-sim-rs110-rs20.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+similarity rs110-rs32
# python main.py -c config/cifar100/rl/rl-sim-rs110-rs32.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+similarity rs32x4-rs8x4
# python main.py -c config/cifar100/rl/rl-sim-rs32x4-rs8x4.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+similarity wrn40_2 wrn40_1
# python main.py -c config/cifar100/rl/rl-sim-wrn40_2-wrn40_1.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+similarity wrn40_2 wrn16_2
# python main.py -c config/cifar100/rl/rl-sim-wrn40_2-wrn16_2.yaml \
#         -t 'ITKD+similarity best config'd

# ITKD+similarity vgg13 vgg8
# python main.py -c config/cifar100/rl/rl-sim-vgg13-vgg8.yaml \
#         -t 'ITKD+similarity best config'

# ITKD+vid wrn40_2 wrn40_1
# python main.py -c config/cifar100/rl/rl-vid-wrn40_2-wrn40_1.yaml \
#         -t 'ITKD+vid best config'

# ITKD+pkt wrn40_2 wrn16_2
# python main.py -c config/cifar100/rl/rl-pkt-wrn40_2-wrn16_2.yaml \
#         -t 'ITKD+pkt best config'

# ITKD+pkt vgg13 vgg8
# python main.py -c config/cifar100/rl/rl-pkt-vgg13-vgg8.yaml \
#         -t 'ITKD+pkt best config'

# ITKD+srrl wrn40_2 wrn40_1
# python main.py -c config/cifar100/rl/rl-srrl-wrn40_2-wrn40_1.yaml \
#         -t 'ITKD+srrl best config'

# ITKD+kd rs56-rs20
# python main.py -c config/cifar100/rl/rl-rs56-rs20.yaml

# ITKD rs110-rs32
# python main.py -c config/cifar100/rl/rl-rs110-rs32.yaml

# ITKD wrn40_2 wrn16_2
# python main.py -c config/cifar100/rl/rl-wrn40_2-wrn16_2.yaml

# ITKD vgg13 vgg8
# python main.py -c config/cifar100/rl/rl-vgg13-vgg8.yaml


# ITKD crd vgg13 vgg8
# python main.py -c config/cifar100/rl/rl-crd-vgg13-vgg8.yaml

