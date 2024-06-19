#!/bin/bash

# conda activate
source /mnt/data3/zyx/miniconda3/bin/activate RLKD

device=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$device

echo "==> use gpu device id ${device}"

export TMPDIR=`pwd`/tmp

# run
# cifar100

# rlkd+dkd rs32x4-rs8x4
python main.py -c config/cifar100/rl/rl-dkd-rs32x4-rs8x4.yaml \
        -t 'rlkd+dkd best config'

# rlkd kd rs56-rs20
# python main.py -c config/cifar100/rl/rl-rs56-rs20.yaml \
#         -t 'rlkd+kd best config'

# rlkd+dkd vgg13-vgg8
# python main.py -c config/cifar100/rl/rl-dkd-vgg13-vgg8.yaml \
#         -t 'rlkd+dkd best config'


# rlkd+kd rs56-rs20
# python main.py -c config/cifar100/rl/rl-rs56-rs20.yaml \
#         -t 'rlkd+kd best config'

# rlkd+similarity rs56-rs20
# python main.py -c config/cifar100/rl/rl-sim-rs56-rs20.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+similarity wrn40_2-wrn40_1
# python main.py -c config/cifar100/rl/rl-sim-wrn40_2-wrn40_1.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+similarity wrn40_2-wrn16_2
# python main.py -c config/cifar100/rl/rl-sim-wrn40_2-wrn16_2.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+similarity rs32x4-rs8x4
# python main.py -c config/cifar100/rl/rl-sim-rs32x4-rs8x4.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+similarity vgg13-vgg8
# python main.py -c config/cifar100/rl/rl-sim-vgg13-vgg8.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+srrl rs32x4-rs8x4
# python main.py -c config/cifar100/rl/rl-srrl-rs32x4-rs8x4.yaml \
#         -t 'rlkd+srrl best config'

# rlkd+srrl rs56-rs20
# python main.py -c config/cifar100/rl/rl-srrl-rs56-rs20.yaml \
#         -t 'rlkd+srrl best config'

# rlkd+srrl rs110-rs20
# python main.py -c config/cifar100/rl/rl-srrl-rs110-rs20.yaml \
#         -t 'rlkd+srrl best config'

# rlkd+srrl rs32x4-rs8x4
# python main.py -c config/cifar100/rl/rl-srrl-rs32x4-rs8x4.yaml \
#         -t 'rlkd+srrl best config'

# rlkd+srrl wrn40_2-wrn40_1
# python main.py -c config/cifar100/rl/rl-srrl-wrn40_2-wrn40_1.yaml \
#         -t 'rlkd+srrl best config'

# rlkd+srrl wrn40_2-wrn16_2
# python main.py -c config/cifar100/rl/rl-srrl-wrn40_2-wrn16_2.yaml \
#         -t 'rlkd+srrl best config'

# rlkd+srrl vgg13-vgg8
# python main.py -c config/cifar100/rl/rl-srrl-vgg13-vgg8.yaml \
#         -t 'rlkd+srrl best config'

# rlkd+dkd rs110-rs20
# python main.py -c config/cifar100/rl/rl-dkd-rs110-rs20.yaml \
#         -t 'rlkd+dkd best config'

# rlkd+similarity rs110-rs20
# python main.py -c config/cifar100/rl/rl-sim-rs110-rs20.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+similarity rs110-rs32
# python main.py -c config/cifar100/rl/rl-sim-rs110-rs32.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+similarity rs32x4-rs8x4
# python main.py -c config/cifar100/rl/rl-sim-rs32x4-rs8x4.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+similarity wrn40_2 wrn40_1
# python main.py -c config/cifar100/rl/rl-sim-wrn40_2-wrn40_1.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+similarity wrn40_2 wrn16_2
# python main.py -c config/cifar100/rl/rl-sim-wrn40_2-wrn16_2.yaml \
#         -t 'rlkd+similarity best config'd

# rlkd+similarity vgg13 vgg8
# python main.py -c config/cifar100/rl/rl-sim-vgg13-vgg8.yaml \
#         -t 'rlkd+similarity best config'

# rlkd+vid wrn40_2 wrn40_1
# python main.py -c config/cifar100/rl/rl-vid-wrn40_2-wrn40_1.yaml \
#         -t 'rlkd+vid best config'

# rlkd+pkt wrn40_2 wrn16_2
# python main.py -c config/cifar100/rl/rl-pkt-wrn40_2-wrn16_2.yaml \
#         -t 'rlkd+pkt best config'

# rlkd+pkt vgg13 vgg8
# python main.py -c config/cifar100/rl/rl-pkt-vgg13-vgg8.yaml \
#         -t 'rlkd+pkt best config'

# rlkd+srrl wrn40_2 wrn40_1
# python main.py -c config/cifar100/rl/rl-srrl-wrn40_2-wrn40_1.yaml \
#         -t 'rlkd+srrl best config'

# rlkd+kd rs56-rs20
# python main.py -c config/cifar100/rl/rl-rs56-rs20.yaml

# rlkd rs110-rs32
# python main.py -c config/cifar100/rl/rl-rs110-rs32.yaml

# rlkd wrn40_2 wrn16_2
# python main.py -c config/cifar100/rl/rl-wrn40_2-wrn16_2.yaml

# rlkd vgg13 vgg8
# python main.py -c config/cifar100/rl/rl-vgg13-vgg8.yaml


# rlkd crd vgg13 vgg8
# python main.py -c config/cifar100/rl/rl-crd-vgg13-vgg8.yaml

