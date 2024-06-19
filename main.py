import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as Callbacks
import torchvision
from pytorch_lightning import loggers
import click
import yaml

import utils.helper as helper
from utils.trainer import (
    BaseDistiller,
    DKDDistiller,
    MKDDistiller,
    MLPDKDDistiller,
    MLPDistiller,
    ITKDDistiller,
)


@click.command()
@click.option("--config", "-c", type=str, required=True, help="path to config file")
@click.option("--tags", "-t", multiple=True, type=str, help="comment for this exp")
def main(config: str, tags: list):
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    helper.show_args(cfg)

    # my_cfg = CFG(cfg)

    # dataset
    datamodule, nc = helper.check_and_get_dataset(
        dataset=cfg["dataset"],
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        val_batch_size=cfg["batch_size"],
        kd=cfg["kd"],
    )

    # model
    if cfg["dataset"] == "imagenet":
        tea_model = torchvision.models.resnet34(pretrained=True)
        stu_model = helper.check_and_get_model(model_name="resnet18")(num_classes=nc)
    elif cfg["dataset"] == "tinyimagenet":
        tea_model = helper.check_and_get_model(model_name=cfg["tea_model"])(
            num_classes=nc
        )
        stu_model = helper.check_and_get_model(model_name=cfg["stu_model"])(
            num_classes=nc
        )
        # load teacher model
        teacher_checkpoint = torch.load(cfg["teacher_checkpoint"])
        tea_model.load_state_dict(teacher_checkpoint["model"])

    else:
        tea_model = helper.check_and_get_model(model_name=cfg["tea_model"])(
            num_classes=nc
        )
        stu_model = helper.check_and_get_model(model_name=cfg["stu_model"])(
            num_classes=nc
        )

        # load teacher model
        teacher_checkpoint = torch.load(cfg["teacher_checkpoint"])
        tea_model.load_state_dict(teacher_checkpoint["model"])
    # checkpoint = torch.load('/home/zyx/code/ITKD/outputs/distillation/d8lmhexr/checkpoints/distiller-epoch=186-val_acc=0.7129.ckpt')['state_dict']
    # # # checkpoint = torch.load('outputs/distillation/im7fnp3o/checkpoints/distiller-epoch=190-val_acc=0.71.ckpt')['state_dict']

    # # # # print(checkpoint.keys())
    # student_checkpoint = {k[8:]: v for k, v in checkpoint.items() if "student" in k}
    # stu_model.load_state_dict(student_checkpoint)

    # config
    pl.seed_everything(cfg["seed"])
    distiller = None
    if cfg["rl"] is True:
        datamodule.prepare_data()
        if cfg["dataset"] == "cifar100":
            data = torch.randn(2, 3, 32, 32)
        elif cfg["dataset"] == "imagenet":
            data = torch.randn(2, 3, 224, 224)

        tea_model.eval()
        stu_model.eval()

        with torch.no_grad():
            _, feats_t = tea_model(data)
            _, feats_s = stu_model(data)

        cls_criterion, kd_criterion = helper.check_and_get_loss(
            cfg["kd"],
            alpha=cfg["dkd_alpha"] if cfg.get("dkd_alpha") else 0.0,
            beta=cfg["dkd_beta"] if cfg.get("dkd_beta") else 0.0,
            warmup_epoch=cfg["warmup_epoch"],
            feats_t=feats_t["feats"] if cfg["dataset"] == "cifar100" else feats_t,
            feats_s=feats_s["feats"] if cfg["dataset"] == "cifar100" else feats_s,
        )
        distiller = ITKDDistiller(
            student=stu_model,
            teacher=tea_model,
            kd_criterion=kd_criterion,
            nc=nc,
            cls_criterion=cls_criterion,
            datamodule=datamodule,
            discount=0.99,
            tau=0.005,
            reward_scale=1.0,
            optimizer=cfg["optimizer"],
            scheduler=cfg["scheduler"],
            temp=cfg["temperature"],
            gamma=cfg["gamma"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            rl=cfg["rl"],
            milestone=cfg["milestone"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            kd=cfg["kd"],
            batch_size=cfg["batch_size"],
            state_dim=cfg["batch_size"],
            action_dim=cfg["batch_size"],
            discrete_action_dim=cfg["batch_size"],
            clipping_ratio=cfg["clipping_ratio"],
            a_epoch=cfg["a_epoch"],
            c_epoch=cfg["c_epoch"],
            warmup_epoch=cfg["warmup_epoch"],
            temperature=cfg["temperature"],
            rl_balance=cfg["rl_balance"],
            rl_epoch=cfg["rl_epoch"],
            aug_epoch=cfg["aug_epoch"],
            dataset=cfg["dataset"],
        )

    elif cfg["kd"] == "mlp":
        cls_criterion, kd_criterion = helper.check_and_get_loss(cfg["kd"], nc=nc)
        datamodule.prepare_data()
        distiller = MLPDistiller(
            student=stu_model,
            teacher=tea_model,
            kd_criterion=kd_criterion,
            optimizer=cfg["optimizer"],
            scheduler=cfg["scheduler"],
            cls_criterion=cls_criterion,
            temp=cfg["temperature"],
            alpha=cfg["alpha"],
            rl=cfg["rl"],
            nc=nc,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            milestone=cfg["milestone"],
            gamma=cfg["gamma"],
            kd=cfg["kd"],
            decay_stragegy=cfg["decay_stragegy"],
        )
    elif cfg["kd"] == "mlpdkd":
        cls_criterion, kd_criterion = helper.check_and_get_loss(cfg["kd"], nc=nc)
        datamodule.prepare_data()
        distiller = MLPDKDDistiller(
            student=stu_model,
            teacher=tea_model,
            kd_criterion=kd_criterion,
            optimizer=cfg["optimizer"],
            scheduler=cfg["scheduler"],
            cls_criterion=cls_criterion,
            temp=cfg["temperature"],
            alpha=cfg["alpha"],
            rl=cfg["rl"],
            nc=nc,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            milestone=cfg["milestone"],
            gamma=cfg["gamma"],
            kd=cfg["kd"],
            decay_stragegy=cfg["decay_stragegy"],
        )

    elif cfg["kd"] == "manual":
        cls_criterion, kd_criterion = helper.check_and_get_loss(
            "manual_kd",
            manual_temperature_min=cfg["manual_temperature_min"],
            manual_temperature_max=cfg["manual_temperature_max"],
        )
        distiller = MKDDistiller(
            student=stu_model,
            teacher=tea_model,
            kd_criterion=kd_criterion,
            optimizer=cfg.optimizer,
            scheduler=cfg["scheduler"],
            cls_criterion=cls_criterion,
            temp=cfg["temperature"],
            alpha=cfg["alpha"],
            rl=cfg["rl"],
            nc=nc,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            milestone=cfg["milestone"],
            gamma=cfg["gamma"],
            kd=cfg["kd"],
        )
    elif cfg["kd"] == "dkd":
        cls_criterion, kd_criterion = helper.check_and_get_loss(
            "dkd",
            temperature=cfg["temperature"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            warmup_epoch=cfg["warmup_epoch"],
        )
        distiller = DKDDistiller(
            student=stu_model,
            teacher=tea_model,
            kd_criterion=kd_criterion,
            optimizer=cfg["optimizer"],
            scheduler=cfg["scheduler"],
            cls_criterion=cls_criterion,
            temp=cfg["temperature"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            rl=cfg["rl"],
            nc=nc,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            milestone=cfg["milestone"],
            gamma=cfg["gamma"],
            kd=cfg["kd"],
        )
    else:
        cls_criterion, kd_criterion = helper.check_and_get_loss(
            "kd",
        )
        distiller = BaseDistiller(
            student=stu_model,
            teacher=tea_model,
            kd_criterion=kd_criterion,
            optimizer=cfg["optimizer"],
            scheduler=cfg["scheduler"],
            cls_criterion=cls_criterion,
            temp=cfg["temperature"],
            alpha=cfg["alpha"],
            rl=cfg["rl"],
            nc=nc,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            milestone=cfg["milestone"],
            gamma=cfg["gamma"],
            kd=cfg["kd"],
        )

    # logger
    print("==> load logger")
    logger = None
    helper.check_folder_exists_or_create(cfg["outputs"])
    if cfg["wandb"]:
        logger = loggers.WandbLogger(
            project="distillation",
            name=(
                cfg["dataset"] + " " + cfg["kd"] if not cfg["rl"] else "rl " + cfg["kd"]
            )
            + " "
            + cfg["tea_model"]
            + " "
            + cfg["stu_model"]
            + " distiller",
            save_dir=cfg["outputs"],
            config=cfg,
            tags=tags if tags else None,
        )

    else:
        logger = loggers.TensorBoardLogger(
            save_dir=cfg["outputs"],
            name=("rl" if cfg["rl"] else cfg["kd"])
            + " "
            + cfg["tea_model"]
            + " "
            + cfg["stu_model"]
            + " distiller",
        )
    # logger.addHandler(logging.FileHandler(os.path.join(cfg["outputs"], "log.txt")))

    # callbacks
    lr_monitor = Callbacks.LearningRateMonitor(logging_interval="step")
    # early_stopping = Callbacks.EarlyStopping(
    # monitor='best_acc', patience=30, verbose=False, mode='max')
    checkpoint_saver = Callbacks.ModelCheckpoint(
        monitor="best_acc",
        filename="distiller-{epoch:02d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
        verbose=False,
    )

    # train
    # devices = list(cfg['devices'])
    # if len(devices) == 1:
    #     devices = devices[0]
    # else:
    #     devices = list(cfg['devices'])

    trainer = pl.Trainer(
        devices=cfg["devices"],
        accelerator="auto",
        strategy="ddp", 
        max_epochs=cfg["num_epochs"],
        callbacks=[
            lr_monitor,
            checkpoint_saver,
        ],
        logger=logger,
        sync_batchnorm=True,
        benchmark=True,
        reload_dataloaders_every_n_epochs=False
        if cfg["dataset"] == "imagenet"
        else True,
        # strategy="ddp_find_unused_parameters_true",
    )
    if cfg["resume"] != "":
        trainer.fit(distiller, datamodule=datamodule, ckpt_path=cfg["resume"])
    else:
        trainer.fit(
            distiller,
            datamodule=datamodule,
        )

    trainer.validate(distiller, datamodule=datamodule)


if __name__ == "__main__":
    main()
