from typing import Any, Callable, Optional, Tuple
import pytorch_lightning as pl
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder, ImageNet
from torchvision import transforms
import torch
import torch.nn as nn
import albumentations as ALB
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image
import os
from datasets import load_dataset
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm


class CIFAR100Instance(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)

        if train:
            # self.transform = ALB.Compose(
            #     [
            #         ALB.RandomCrop(32,32),
            #         ALB.HorizontalFlip(),
            #         ALB.Normalize((0.5071, 0.4867, 0.4408),
            #                         (0.2675, 0.2565, 0.2761)),
            #         ToTensor(),
            #     ]
            # )

            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )

        else:
            # self.transform = ALB.Compose(
            #     [
            #         ALB.Normalize((0.5071, 0.4867, 0.4408),
            #                         (0.2675, 0.2565, 0.2761)),
            #         ToTensor(),
            #     ]
            # )

            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )

        # self.aug_transform = ALB.Compose(
        #     [
        #         ALB.RandomCrop(32,32),
        #         ALB.HorizontalFlip(),
        #         # ALB.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=0, always_apply=False, p=0.5),
        #         ALB.Normalize((0.5071, 0.4867, 0.4408),
        #                          (0.2675, 0.2565, 0.2761)),
        #         ToTensor(),
        #     ]
        # )

        self.train = train
        self.is_aug = False

        self.remain, self.lost, self.mix_remain = (
            torch.arange(len(self.data)),
            None,
            None,
        )

        self.is_aug = False

        self.mixup_data = torch.tensor(self.data, requires_grad=False).cpu().numpy()
        # self.mixup_data = (
        #     torch.empty(0, dtype=torch.int)
        #     .cpu()
        #     .numpy()
        #     .reshape(
        #         0,
        #         self.data[0].shape[0],
        #         self.data[0].shape[1],
        #         self.data[0].shape[2],
        #     )
        #     .astype(int)
        # )
        self.mixup_idx = torch.empty(0, dtype=torch.int).cpu().numpy()

    def set_aug(self, is_aug):
        self.is_aug = is_aug
        if not self.is_aug:
            self.mixup_data = torch.tensor(self.data, requires_grad=False).cpu().numpy()
            # self.mixup_data = (
            #     torch.empty(0, dtype=torch.int)
            #     .cpu()
            #     .numpy()
            #     .reshape(
            #         0,
            #         self.data[0].shape[0],
            #         self.data[0].shape[1],
            #         self.data[0].shape[2],
            #     )
            #     .astype(int)
            # )

    def __len__(self) -> int:
        if self.is_aug:
            # self.mixup_targets = np.array(self.targets)[self.mixup_idx.astype(int)]
            # return len(self.mixup_idx)
            return len(self.mixup_data)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.is_aug:
            img, target = (
                self.mixup_data[index],
                self.targets[index],
            )
        else:
            img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        img = self.transform(img)

        return img, target, index

    def mixup(self, _entropy, mix_remain, lost):
        dataset = MIXUPDataset(
            self.mixup_data[mix_remain.copy()],
            self.mixup_data[lost.copy()],
            # self.data[mix_remain.copy()],
            # self.data[lost.copy()],
            mix_remain,
            lost,
            _entropy[mix_remain.copy()],
            _entropy[lost.copy()],
        )

        self.lost = lost

        mix_loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=1
        )

        mixnet = MIXUPNet(len(mix_remain))

        for data in mix_loader:
            remain_data, lost_data, remain, lost, remain_en, lost_en, idx = data
            remain_data = remain_data  # .cuda()
            lost_data = lost_data  # .cuda()
            mixup = mixnet(
                remain_data, lost_data, remain, lost, remain_en, lost_en, idx
            )
            self.mixup_data[lost] = mixup

            # self.mixup_idx = np.hstack((self.mixup_idx, lost))


class CIFAR100DataLoader(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size, num_workers, if_crd: False) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set = None
        self.val_set = None
        self.train_agent_set = None
        self.mixup_data = None
        self.remain, self.lost, self.mix_remain = None, None, None
        self.if_crd = if_crd

    def prepare_data(self):
        # download data
        CIFAR100Instance(
            root=self.data_dir,
            train=True,
            download=True,
        )
        CIFAR100Instance(
            root=self.data_dir,
            train=False,
            download=True,
        )

    def setup(self, stage: str) -> None:
        # setup data
        if self.if_crd:
            self.train_set = CIFAR100InstanceSample(
                root=self.data_dir,
                train=True,
                download=False,
                k=16384,
                mode="exact",
                is_sample=True,
                percent=1.0,
            )

        else:
            self.train_set = CIFAR100Instance(
                root=self.data_dir,
                train=True,
                download=False,
            )
        self.val_set = CIFAR100Instance(
            root=self.data_dir,
            train=False,
            download=False,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class MIXUPNet(nn.Module):
    def __init__(self, mix_len):
        super(MIXUPNet, self).__init__()
        self.mix_len = mix_len

    def forward(self, remain_data, lost_data, remain, lost, remain_en, lost_en, idx):
        with torch.no_grad():
            # sum_en = np.exp(remain_en) + np.exp(lost_en)
            # remain_rate = np.exp(remain_en) / sum_en
            # lost_rate = np.exp(lost_en) / sum_en
            remain_rate = (1 - 0.7) * (
                (self.mix_len * torch.ones(len(idx)) - idx) / self.mix_len
            ) + 0.7
            lost_rate = torch.ones(len(idx)) - remain_rate

            minup = (
                remain_data[0].unsqueeze(0) * remain_rate[0]
                + lost_data[0].unsqueeze(0) * lost_rate[0]
            )
            for i in range(1, remain.shape[0]):
                minup = torch.cat(
                    (
                        minup,
                        remain_data[i].unsqueeze(0) * remain_rate[i]
                        + lost_data[i].unsqueeze(0) * lost_rate[i],
                    )
                )

        return torch.tensor(minup.clone().detach(), dtype=torch.uint8).cpu()


class MIXUPDataset(torch.utils.data.Dataset):
    def __init__(self, remain_data, lost_data, remain, lost, remain_en, lost_en):
        self.remain = remain
        self.lost = lost

        self.remain_data = remain_data
        self.lost_data = lost_data

        self.remain_en = remain_en
        self.lost_en = lost_en

    def __len__(self):
        return len(self.remain_data)

    def __getitem__(self, idx):
        return (
            self.remain_data[idx],
            self.lost_data[idx],
            self.remain[idx],
            self.lost[idx],
            self.remain_en[idx],
            self.lost_en[idx],
            idx,
        )


class CIFAR10DataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size,
        num_workers,
    ) -> None:
        super().__init__()

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        train_set = CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform,
        )
        val_set = CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.val_transform,
        )
        self.train_set, self.val_set = train_set, val_set

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch._utils.data.DataLoader(
            self.val_set, batch_size=1, shuffle=False, num_workers=self.num_workers
        )


"""
imagenet dataset
"""


class ImagenetInstance(ImageFolder):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: None = None,
    ):
        if transform == None:
            if split == "train":
                self.transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                        ),
                    ]
                )
            elif split == "val":
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                        ),
                    ]
                )
            else:
                raise NotImplementedError
        else:
            self.transform = transform
        data = os.path.join(root, split)
        super().__init__(
            data,
            self.transform,
        )
        self.is_aug = False

        self.remain, self.lost, self.mix_remain = (
            torch.arange(len(self.samples)),
            None,
            None,
        )

        self.is_aug = False

        # self.mixup_data = torch.tensor(self.data, requires_grad=False).cpu().numpy()
        self.mixup_data = None
        # self.mixup_data = (
        #     torch.empty(0, dtype=torch.int)
        #     .cpu()
        #     .numpy()
        #     .reshape(
        #         0,
        #         self.data[0].shape[0],
        #         self.data[0].shape[1],
        #         self.data[0].shape[2],
        #     )
        #     .astype(int)
        # )
        # self.mixup_idx = torch.empty(0).cpu().numpy()

    def __getitem__(self, index):
        # if self.is_aug:
        #     img, target = (
        #         self.mixup_data[index],
        #         self.targets[index],
        #     )
        # else:
        #     img, target = self.data[index], self.targets[index]

        # img = Image.fromarray(img)

        path, target = self.imgs[index]
        img = self.loader(path)

        img = self.transform(img)

        return img, target, index


"""
imagenet dataloader
"""


class ImagenetDataloader(pl.LightningDataModule):
    def __init__(self, root: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root

    def prepare_data(self) -> None:
        self.train_set = ImagenetInstance(
            self.root, split="train", transform=self.train_transform
        )
        self.val_set = ImagenetInstance(
            self.root, split="val", transform=self.val_transform
        )

    def setup(self, stage: str) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class TinyImagenetInstance(ImageFolder):
    def __init__(
        self,
        root: str,
        transform,
    ):
        super().__init__(root, transform)
        self.transform = transform
        self.root = root

        self.IMAGE_SHAPE = (64, 64, 3)

        # preload
        self.data = np.zeros((len(self.samples),) + self.IMAGE_SHAPE)
        self.targets = []
        loop = tqdm(enumerate(self.imgs), total=len(self.imgs))
        loop.set_description("Loading images...")
        for i, (path, target) in loop:
            img = self.loader(path)
            img = np.array(img)
            self.data[i] = img
            self.targets.append(target)

        self.is_aug = False

        self.remain, self.lost, self.mix_remain = (
            torch.arange(len(self.data)),
            None,
            None,
        )

        self.is_aug = False

        self.mixup_data = torch.tensor(self.data, requires_grad=False).cpu().numpy()

    def set_aug(self, is_aug):
        self.is_aug = is_aug
        if not self.is_aug:
            self.mixup_data = torch.tensor(self.data, requires_grad=False).cpu().numpy()

    def __getitem__(self, index: int):
        if self.is_aug:
            img, target = (
                self.mixup_data[index],
                self.targets[index],
            )
        else:
            img, target = self.data[index], self.targets[index]
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, index

    def __len__(self) -> int:
        if self.is_aug:
            # self.mixup_targets = np.array(self.targets)[self.mixup_idx.astype(int)]
            # return len(self.mixup_idx)
            return len(self.mixup_data)
        else:
            return len(self.data)

    def mixup(self, _entropy, mix_remain, remain, lost):
        dataset = MIXUPDataset(
            self.mixup_data[mix_remain.copy()],
            self.mixup_data[lost.copy()],
            # self.data[mix_remain.copy()],
            # self.data[lost.copy()],
            mix_remain,
            lost,
            _entropy[mix_remain.copy()],
            _entropy[lost.copy()],
        )

        self.lost = lost

        mix_loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=1
        )

        mixnet = MIXUPNet(len(mix_remain))

        for data in mix_loader:
            remain_data, lost_data, remain, lost, remain_en, lost_en, idx = data
            remain_data = remain_data  # .cuda()
            lost_data = lost_data  # .cuda()
            mixup = mixnet(
                remain_data, lost_data, remain, lost, remain_en, lost_en, idx
            )
            self.mixup_data[lost] = mixup


class TinyImagenetDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

    def prepare_data(self) -> None:
        self.train_set = TinyImagenetInstance(
            split="train", transform=self.train_transform
        )
        self.val_set = TinyImagenetDataLoader(split="val", transform=self.val_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )


class CIFAR100InstanceSample(CIFAR100Instance):
    """
    CIFAR100Instance+Sample Dataset
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        k=4096,
        mode="exact",
        is_sample=True,
        percent=1.0,
    ):
        super(CIFAR100InstanceSample, self).__init__(
            root, train, transform, target_transform, download
        )
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [
            np.asarray(self.cls_positive[i]) for i in range(num_classes)
        ]
        self.cls_negative = [
            np.asarray(self.cls_negative[i]) for i in range(num_classes)
        ]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [
                np.random.permutation(self.cls_negative[i])[0:n]
                for i in range(num_classes)
            ]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        if self.is_aug:
            img, target = (
                self.mixup_data[index],
                self.targets[index],
            )
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # sample contrastive examples
        if self.mode == "exact":
            pos_idx = index
        elif self.mode == "relax":
            pos_idx = np.random.choice(self.cls_positive[target], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)
        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return img, target, index, sample_idx


if __name__ == "__main__":
    dataset = ImagenetInstance("/mnt/data2/imagenet", split="train")
    print(len(dataset))
