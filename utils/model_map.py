from models.cifar.resnet import (
    resnet110,
    resnet14,
    resnet20,
    resnet32,
    resnet32x4,
    resnet44,
    resnet56,
    resnet8,
    resnet8x4
)

from models.cifar.resnetv2 import (
    ResNet101,
    ResNet152,
    ResNet18,
    ResNet34,
    ResNet50
)

from models.cifar.wrn import (
    wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2,
)

from models.cifar.vgg import (
    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, vgg8, vgg8_bn
)

from models.cifar.mobilenetv2 import mobile_half
from models.cifar.ShuffleNetv1 import ShuffleV1
from models.cifar.ShuffleNetv2 import ShuffleV2

import os


cifar_model_dict = {
    # teachers
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet32x4": resnet32x4,
    "ResNet50": ResNet50,
    "vgg13": vgg13_bn,
    # students
    "resnet8": resnet8,
    "resnet14": resnet14,
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet8x4": resnet8x4,
    "resnet18": ResNet18,
    "wrn_16_1": wrn_16_1,
    "wrn_16_2": wrn_16_2,
    "wrn_40_1": wrn_40_1,
    "wrn_40_2": wrn_40_2,
    "vgg8": vgg8_bn,
    "vgg11": vgg11_bn,
    "vgg16": vgg16_bn,
    "vgg19": vgg19_bn,
    "MobileNetV2": mobile_half,
    "ShuffleV1": ShuffleV1,
    "ShuffleV2": ShuffleV2,
}
