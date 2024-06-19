from collections import namedtuple
import os
import torch
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet
from torchvision import transforms
import pprint
import torch.nn as nn
from pytorch_lightning.callbacks import Callback

from utils.dataloader import (
    CIFAR100DataLoader,
    ImagenetDataloader,
    TinyImagenetDataLoader,
)
from utils.loss import SRRL
from utils.mlp.mlp import MLP_Loss, MLPDKD_Loss

from .model_map import cifar_model_dict
from .loss import (
    PKT,
    CRDLoss,
    DKD_Loss,
    KL_Loss,
    Instance_Temp_KD_Loss,
    ITKD_Loss,
    Similarity,
    VIDLoss,
)


def check_folder_exists_or_create(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)


"""
check dataset and return train_set, test_set
"""


def check_and_get_dataset(
    dataset: str,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    data_dir: str,
    **kwargs,
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    if dataset != "":
        check_folder_exists_or_create(data_dir)
        nc = 0

        if dataset == "cifar100":
            
            dm = CIFAR100DataLoader(data_dir, batch_size, num_workers, if_crd=True if kwargs["kd"]=="crd" else False)
            # dm.train_agent_set
            nc = 100
            return dm, nc

        elif dataset == "cifar10":
            train_set = CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=train_transform,
            )
            val_set = CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=val_transform,
            )

            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            nc = 10
            return train_set, val_set, train_loader, val_loader, nc
        elif dataset == "imagenet":
            dm = ImagenetDataloader(data_dir, batch_size, num_workers)
            nc = 1000
            return dm, nc
        elif dataset == "tinyimagenet":
            dm = TinyImagenetDataLoader(data_dir, batch_size, num_workers)
            nc = 200
            return dm, nc

        else:
            raise ValueError("dataset not supported")
    else:
        raise ValueError("dataset not specified")


"""
check model and return model
"""


def check_and_get_model(model_name: str) -> tuple:
    if model_name != "":
        if model_name == "resnet20":
            return cifar_model_dict["resnet20"]
        elif model_name == "resnet32x4":
            return cifar_model_dict["resnet32x4"]
        elif model_name == "resnet8x4":
            return cifar_model_dict["resnet8x4"]
        elif model_name == "resnet18":
            return cifar_model_dict["resnet18"]
        elif model_name == "resnet32":
            return cifar_model_dict["resnet32"]
        elif model_name == "resnet110":
            return cifar_model_dict["resnet110"]
        elif model_name == "resnet56":
            return cifar_model_dict["resnet56"]
        elif model_name == "wrn_16_2":
            return cifar_model_dict["wrn_16_2"]
        elif model_name == "wrn_40_2":
            return cifar_model_dict["wrn_40_2"]
        elif model_name == "wrn_40_1":
            return cifar_model_dict["wrn_40_1"]
        elif model_name == "resnet32x4":
            return cifar_model_dict["resnet32x4"]
        elif model_name == "ShuffleV1":
            return cifar_model_dict["ShuffleV1"]
        elif model_name == "ShuffleV2":
            return cifar_model_dict["ShuffleV2"]
        elif model_name == "vgg13":
            return cifar_model_dict["vgg13"]
        elif model_name == "vgg8":
            return cifar_model_dict["vgg8"]
        else:
            raise ValueError("model not supported")
    else:
        raise ValueError("model not specified")


"""
get classification loss and kd loss
"""


def weights_init_normal(m, bias=0.0):
    if (
        isinstance(m, nn.Conv2d)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.BatchNorm2d)
    ):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def weights_init_xavier(m, bias=0.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, bias)


def weights_init_kaiming(m, bias=0.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, bias)


def weights_init_orthogonal(m, bias=0.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, bias)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, bias)


def init_weights(model, init_type="normal", bias=0.0):
    if init_type == "normal":
        init_function = weights_init_normal
    elif init_type == "xavier":
        init_function = weights_init_xavier
    elif init_type == "kaiming":
        init_function = weights_init_kaiming
    elif init_type == "orthogonal":
        init_function = weights_init_orthogonal
    else:
        raise NotImplementedError(
            "initialization method [%s] is not implemented" % init_type
        )

    if isinstance(model, list):
        for m in model:
            init_weights(m, init_type)
    else:
        for m in model.modules():
            init_function(m, bias=bias)


def check_and_get_loss(loss: str, **kwargs) -> tuple:
    cls_loss = nn.CrossEntropyLoss()
    kl_loss = None
    if loss != "":
        if loss == "rl":
            kl_loss = ITKD_Loss()
            return cls_loss, kl_loss
        elif loss == "mlp":
            kl_loss = MLP_Loss(in_chan=kwargs["nc"] * 2)
            return cls_loss, kl_loss
        elif loss == "mlpdkd":
            kl_loss = MLPDKD_Loss(in_chan=kwargs["nc"] * 2)
            return cls_loss, kl_loss
        elif loss == "kd":
            kl_loss = KL_Loss()
            return cls_loss, kl_loss
        elif loss == "manual_kd":
            kl_loss = Instance_Temp_KD_Loss(
                beta=1.0,
                loop_min=kwargs["manual_temperature_min"],
                loop_max=kwargs["manual_temperature_max"],
            )
            return cls_loss, kl_loss
        elif loss == "dkd":
            kl_loss = DKD_Loss(
                kwargs["alpha"], kwargs["beta"], kwargs["warmup_epoch"]
            )  # as dkd says, the warmup is 20, alpha=1,beta=8
            return cls_loss, kl_loss
        elif loss == "similarity":
            kl_loss = Similarity()
            return cls_loss, kl_loss
        elif loss == "srrl":
            feats_s, feats_t = kwargs["feats_s"], kwargs["feats_t"]
            s_n = feats_s[-1].shape[1]
            t_n = feats_t[-1].shape[1]
            kl_loss = SRRL(s_n=s_n, t_n=t_n)
            return cls_loss, kl_loss
        elif loss == "vid":
            feats_s, feats_t = kwargs["feats_s"], kwargs["feats_t"]
            s_n = [f.shape[1] for f in feats_s[1:-1]]
            t_n = [f.shape[1] for f in feats_t[1:-1]]
            kl_loss = [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
            return cls_loss, kl_loss
        elif loss == "pkt":
            kl_loss = PKT()
            return cls_loss, kl_loss
            pass
        elif loss == "crd":
            feats_s, feats_t = kwargs["feats_s"], kwargs["feats_t"]
            opt = namedtuple(
                    "opt",
                    [
                        "s_dim",
                        "t_dim",
                        "feat_dim",
                        "n_data",
                        "nce_k",
                        "nce_t",
                        "nce_m",
                    ],
                )
            opt.s_dim = feats_s[-1].shape[1]
            opt.t_dim = feats_t[-1].shape[1]
            opt.feat_dim = 128
            opt.nce_k = 16384
            opt.nce_t = 0.07
            opt.nce_m = 0.5
            opt.n_data = 50000
            kl_loss = CRDLoss(opt)
            return cls_loss, kl_loss

        else:
            raise ValueError("loss not supported")
    else:
        raise ValueError("loss not specified")


def show_args(args: dict):
    pprint.pprint(args)


class MyEarlyStopCallBack(Callback):
    def __init__(self, begin_epoch: int, verbose: bool, patience: int) -> None:
        super().__init__()
        self.begin_epoch = begin_epoch
        self.best_acc = 0.0
        self.verbose = verbose
        self.patience = patience
        self.index = 0

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.begin_epoch:
            if pl_module.val_acc > self.best_acc:
                self.index = 0
                increase = pl_module.val_acc - self.best_acc
                self.best_acc = pl_module.val_acc
                if self.verbose:
                    print(
                        "\n\n ==> Best Acc: {:.4f}, increase: {:4f}".format(
                            self.best_acc, increase
                        )
                    )
            else:
                self.index += 1
                if self.index >= self.patience:
                    trainer.should_stop = True
                    print(
                        "\n\n ==> Early Stopping at epoch {}".format(
                            trainer.current_epoch
                        )
                    )
                else:
                    if self.verbose:
                        print(
                            "\n\n ==> Best Acc: {:.4f}, not increase".format(
                                self.best_acc
                            )
                        )


