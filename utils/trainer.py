from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.utils import shuffle
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rl.ppo import Actor, Critic, gaussian_likelihood
from utils.rl.agents.utils.ppo_utils import ReplayBufferPPO
import numpy as np
from torchmetrics.functional.classification import multiclass_accuracy

from tqdm import tqdm

from utils.rl.utils import CriticalStateDetector, RewardRedistribution, lossfunction_rew

from .loss import KL_Loss
import pandas as pd


class BaseDistiller(pl.LightningModule):
    def __init__(
        self,
        student,
        teacher,
        kd_criterion,
        cls_criterion,
        optimizer,
        scheduler,
        **kwargs,
    ):
        super(BaseDistiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.criterion = [kd_criterion, cls_criterion]
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.rl = kwargs["rl"]
        self.temp = kwargs["temp"]
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]

        self.lr = kwargs["lr"]
        self.weight_decay = kwargs["weight_decay"]
        self.milestone = kwargs["milestone"]
        self.gamma = kwargs["gamma"]
        self.val_acc = 0.0
        self.kd = kwargs["kd"]

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=kwargs["nc"]
        )
        self.top5_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=kwargs["nc"], top_k=5
        )

        self.best_acc = 0.0
        self.top5_best_acc = 0.0

        self.teacher.eval()

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        student_outputs = self.student(inputs)[0]
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)[0]

        cls_loss = self.criterion[1](student_outputs, targets)
        kl_loss = self.criterion[0](student_outputs, teacher_outputs, self.temp)

        losses = cls_loss + kl_loss
        self.log_dict(
            {
                "train_loss": losses,
                "train_cls_loss": cls_loss,
                "train_kl_loss": kl_loss,
            },
            sync_dist=True,
        )

        return {"loss": losses, "cls_loss": cls_loss, "kl_loss": kl_loss}

    def validation_step(self, batch, batch_idx):
        inputs, targets, index = batch
        outputs = self.student(inputs)
        if type(outputs) == tuple:
            outputs = outputs[0]
        acc = self.accuracy(outputs, targets)
        top5_acc = self.top5_accuracy(
            outputs,
            targets,
        )
        loss = self.criterion[1](outputs, targets)

        self.log_dict(
            {"val_acc": acc, "val_loss": loss, "top5_val_acc": top5_acc}, sync_dist=True
        )

        return {"val_acc": acc, "val_loss": loss}

    def configure_optimizers(
        self,
    ):
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.student.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestone, gamma=0.1
            )
            return [optimizer], [scheduler]
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestone, gamma=0.1
            )
            return [optimizer], [scheduler]
        elif self.optimizers == "AdamW":
            optimizer = torch.optim.AdamW(
                self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestone, gamma=0.1
            )
            return [optimizer], [scheduler]

    def on_validation_epoch_end(
        self,
    ):
        val_acc = self.accuracy.compute()
        top5_val_acc = self.top5_accuracy.compute()
        self.accuracy.reset()
        self.top5_accuracy.reset()
        self.val_acc = val_acc
        self.top5_val_acc = top5_val_acc
        if self.val_acc > self.best_acc:
            self.best_acc = self.val_acc
        if self.top5_val_acc > self.top5_best_acc:
            self.top5_best_acc = self.top5_val_acc
        self.log_dict(
            {
                "epoch": self.current_epoch + 1,
                "best_acc": self.best_acc,
                "best_acc_top5": self.top5_best_acc,
            },
            sync_dist=True,
        )
        return val_acc


class DKDDistiller(BaseDistiller):
    def __init__(
        self,
        student,
        teacher,
        kd_criterion,
        cls_criterion,
        optimizer,
        scheduler,
        **kwargs,
    ):
        super().__init__(
            student,
            teacher,
            kd_criterion,
            cls_criterion,
            optimizer,
            scheduler,
            **kwargs,
        )
        self.dkd_alpha = kwargs["alpha"]
        self.dkd_beta = kwargs["beta"]
        self.criterion = [kd_criterion, cls_criterion]

    def training_step(self, batch, batch_idx):
        input, target, _ = batch
        student_outputs = self.student(input)[0]
        with torch.no_grad():
            teacher_outputs = self.teacher(input)[0]

        cls_loss = self.criterion[1](student_outputs, target)
        kl_loss = self.criterion[0](
            student_outputs,
            teacher_outputs,
            target,
            epoch=self.current_epoch + 1,
            T=torch.ones((target.shape[0], 1)).to(self.device).float() * 8.0,
        )

        losses = cls_loss + kl_loss
        self.log_dict(
            {
                "train_loss": losses,
                "train_cls_loss": cls_loss,
                "train_kl_loss": kl_loss,
            },
            sync_dist=True,
        )

        return {"loss": losses, "cls_loss": cls_loss, "kl_loss": kl_loss}


class ITKDDistiller(BaseDistiller):
    """
    KD+RL
    KD部分，使用RL的输出，当为蒸馏温度。

    蒸馏温度在1到10之间

    """

    def __init__(
        self,
        student,
        teacher,
        kd_criterion,
        cls_criterion,
        optimizer,
        scheduler,
        **kwargs,
    ):
        super().__init__(
            student,
            teacher,
            kd_criterion,
            cls_criterion,
            optimizer,
            scheduler,
            **kwargs,
        )

        self.criterion = [
            kd_criterion,
            cls_criterion,
        ]

        self.datamodule = kwargs["datamodule"]

        self.nc = kwargs["nc"]

        self.batch_size = kwargs["batch_size"]

        self.automatic_optimization = False

        self.c_epoch, self.a_epoch = kwargs["c_epoch"], kwargs["a_epoch"]
        self.clipping_ratio = kwargs["clipping_ratio"]

        self.obs_dim = 3
        self.actor = Actor(self.obs_dim, 1, 1, 1).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)
        self.max_action = 1.0
        self.is_update = 64
        self.replay_buffer = ReplayBufferPPO(
            obs_dim=self.obs_dim,
            discrete_action_dim=1,
            parameter_action_dim=1,
            size=self.is_update,
        )

        self.reward_redistribution = RewardRedistribution(
            n_positions=self.obs_dim, n_actions=1, n_lstm=16
        )  # LSTM params
        self.critical_state_detector = CriticalStateDetector(
            n_positions=self.obs_dim, n_lstm=16
        )

        self.max_grad = 2.0

        self.warmup_epoch = kwargs["warmup_epoch"]
        self.rl_epoch = kwargs["rl_epoch"]

        # self.last_stu_top1 = None
        self.temperature = kwargs["temperature"]
        self.rl_balance = (
            kwargs["rl_balance"] if kwargs["rl_balance"] is not None else 1
        )

        # lstm epoch
        self._update = 10

        self.aug_epoch = kwargs["aug_epoch"]

        self.entropy_record = None

        self.kd = kwargs["kd"]
        if kwargs["kd"] in ["crd", "similarity", "pkt", "vid", "srrl"]:
            self.criterion.append(KL_Loss())

        if kwargs["kd"] == "srrl":
            self.criterion.append(nn.MSELoss())

        self.acc = []
        self.batch_update = False

        self.dataset = kwargs["dataset"]

    def on_train_start(self) -> None:
        self.entropy = np.zeros((len(self.datamodule.train_set), 1))

        self.state, self.t_action, self.parameter_logp_t = None, None, None
        # self.last_stu_top1 = torch.zeros((len(self.datamodule.train_set), )).to(self.device)

        for i in range(len(self.criterion)):
            if isinstance(self.criterion[i], nn.Module):
                self.criterion[i] = self.criterion[i].to(self.device)
            elif isinstance(self.criterion[i], list):
                for j in range(len(self.criterion[i])):
                    self.criterion[i][j] = self.criterion[i][j].to(self.device)

        self.all_T = np.zeros((len(self.datamodule.train_set), 1))

    def forward(self, x):
        student_outputs, stu_feats = self.student(x)
        with torch.no_grad():
            teacher_outputs, tea_feats = self.teacher(x)

        return student_outputs, teacher_outputs, stu_feats, tea_feats

    def training_step(self, batch, batch_idx):
        opt, _, _, _, _ = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        # deploy agent
        if self.kd == "crd":
            inputs, targets, index, contrast_idx = batch
        else:
            inputs, targets, index = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.student.train()
        self.teacher.eval()
        student_outputs, teacher_outputs, stu_feats, tea_feats = self(inputs)
        # self.last_stu_top1[index] = torch.max(
        #     F.softmax(student_outputs.detach(), dim=1), dim=1
        # )[0]

        _index = index.detach().cpu().numpy()

        with torch.no_grad():
            probs = F.softmax(student_outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs), dim=1)
            temp = entropy.detach().cpu().numpy()
            self.entropy[_index, 0] = temp

        cls_loss = self.criterion[1](student_outputs, targets)

        acc = multiclass_accuracy(student_outputs, targets, self.nc)
        """
        RL部分开始
        """
        if (
            (self.current_epoch + 1) >= self.warmup_epoch
            and ((self.current_epoch + 1) % self.rl_epoch == 0)
            and self.batch_update
        ):
            # print("rl training start")
            """
            train RL
            1. get T
            2. collect T
            3. collect acc
            4. train RL every two batch
            """
            if len(self.acc) == 0:
                raise NotImplementedError
            else:
                (_, act_opt, cri_opt, _, _) = self.optimizers()

                # get reward
                """
                差或者比值
                """
                reward = self.acc[-1] - self.acc[-2]
                # 初期acc差距大不代表效果好，而是kd进展巨大
                # 随便找个函数拟合一下
                # reward = (
                #     torch.sigmoid(torch.tensor(self.current_epoch / 80))
                #     .detach()
                #     .cpu()
                #     .numpy()
                #     .item()
                #     * reward
                # )

                # 鼓励探索
                """
                周期是300, 2pi/150 = T
                (cos(k \pi) + 1) / 2 + 0.5
                """
                if self.dataset == "cifar100":
                    if (self.current_epoch + 1) < 150:
                        reward = (
                            (
                                (
                                    torch.cos(
                                        torch.tensor(self.current_epoch + 1)
                                        * 1
                                        / 75
                                        * torch.pi
                                    )
                                    + 1
                                )
                                / 2
                                + 1
                            )
                            * 10
                            * reward
                        )
                    elif (self.current_epoch + 1) < 210:
                        reward = (
                            (
                                (
                                    torch.cos(
                                        torch.tensor(self.current_epoch + 1)
                                        * 1
                                        / 30
                                        * torch.pi
                                    )
                                    + 1
                                )
                                / 2
                                + 1
                            )
                            * 5
                            * reward
                        )
                    elif (self.current_epoch + 1) < 240:
                        reward = (
                            (
                                (
                                    torch.cos(
                                        torch.tensor(self.current_epoch + 1)
                                        * 1
                                        / 30
                                        * torch.pi
                                    )
                                    + 1
                                )
                                / 2
                                + 1
                            )
                            * 2
                            * reward
                        )
                elif self.dataset == "imagenet":
                    if (self.current_epoch + 1) < 30:
                        reward = (
                            (
                                (
                                    torch.cos(
                                        torch.tensor(self.current_epoch + 1)
                                        * 1
                                        / 75
                                        * torch.pi
                                    )
                                    + 1
                                )
                                / 2
                                + 1
                            )
                            * 10
                            * reward
                        )
                    elif (self.current_epoch + 1) < 60:
                        reward = (
                            (
                                (
                                    torch.cos(
                                        torch.tensor(self.current_epoch + 1)
                                        * 1
                                        / 30
                                        * torch.pi
                                    )
                                    + 1
                                )
                                / 2
                                + 1
                            )
                            * 5
                            * reward
                        )
                    elif (self.current_epoch + 1) < 90:
                        reward = (
                            (
                                (
                                    torch.cos(
                                        torch.tensor(self.current_epoch + 1)
                                        * 1
                                        / 30
                                        * torch.pi
                                    )
                                    + 1
                                )
                                / 2
                                + 1
                            )
                            * 2
                            * reward
                        )
                # if (self.current_epoch + 1) < 150:
                #     reward = 10 * reward
                # elif (self.current_epoch + 1) < 180:
                #     reward = 5 * reward
                # elif (self.current_epoch + 1) < 210:
                #     reward = 2 * reward
                # else:
                #     raise NotImplementedError
                # reward = reward * 10 if (self.current_epoch + 1 < 80) else reward * 5

                self.replay_buffer.add(
                    obs=self.state.detach().cpu().numpy(),
                    discrete_action=0.0,
                    parameter_action=self.t_action,
                    rew=reward,
                    val=self.v_t,
                    discrete_logp=0.0,
                    parameter_logp=self.parameter_logp_t,
                )

                self.replay_buffer.finish_path(self.last_vals)

                (
                    obs_buf,
                    discrete_act_buf,
                    parameter_act_buf,
                    adv_buf,
                    ret_buf,
                    discrete_logp_buf,
                    parameter_logp_buf,
                ) = self.replay_buffer.get()

                self.train_det(obs_buf, rewards=ret_buf, actions=parameter_act_buf)
                self.train_rew(
                    obs_buf,
                    discrete_act_buf,
                    parameter_act_buf,
                    adv_buf,
                    ret_buf,
                    discrete_logp_buf,
                    parameter_logp_buf,
                )
                masks = self.get_masks(obs_buf)
                masks = masks.detach().cpu().numpy()

                rew = self.get_prediction(obs_buf, parameter_act_buf, ret_buf)
                self.replay_buffer.obs_buf = obs_buf * masks

                self.replay_buffer.rew_buf = rew
                self.replay_buffer.finish_path(self.last_vals)

                self.update_agent(act_opt, cri_opt)

                self.replay_buffer.reset()
                self.batch_update = False

        if (self.current_epoch + 1) >= self.warmup_epoch:
            self.actor.eval(), self.critic.eval()

            # 得到observations
            obs = self.get_observations(
                student_outputs, teacher_outputs, targets, index=index
            )
            total_steps = 0
            actions = []

            # explore the world
            self.actor.train()
            while total_steps < student_outputs.shape[0]:
                state = obs[total_steps, :]
                (
                    t_action,
                    _,
                    parameter_logp_t,
                ) = self.select_action(state)

                actions.append(np.array([0.0, t_action], dtype=np.float32))

                if (self.current_epoch + 1) % self.rl_epoch == 0:
                    # get value from critic function
                    v_t = self.get_value(state)

                    if (total_steps + 1) % self.is_update == 0:
                        # update
                        state = state.reshape(1, -1).float().to(self.device)
                        with torch.no_grad():
                            value = self.critic(state)
                        self.last_vals = value.cpu().numpy().squeeze(0)
                        self.state, self.t_action, self.parameter_logp_t = (
                            state,
                            t_action,
                            parameter_logp_t,
                        )
                        self.v_t = v_t
                        self.batch_update = True
                    else:
                        self.replay_buffer.add(
                            obs=state.detach().cpu().numpy(),
                            discrete_action=0.0,
                            parameter_action=t_action,
                            rew=0.0,
                            val=v_t,
                            discrete_logp=0.0,
                            parameter_logp=parameter_logp_t,
                        )
                total_steps += 1

            actions = torch.from_numpy(np.array(actions)).to(
                self.device
            )  # [timesteps_per_batch, 2]
            T = actions[:, 1].reshape(-1, 1)

        else:
            T = (
                torch.ones((targets.shape[0], 1)).to(self.device).float()
                * self.temperature
            )
            self.replay_buffer.reset()

        # Append T values from each epoch to the list

        self.all_T[_index, -1] = T.detach().cpu().numpy().squeeze(1)

        if self.kd == "dkd":
            kl_loss = self.criterion[0](
                student_outputs,
                teacher_outputs,
                targets,
                epoch=self.current_epoch + 1,
                T=T.to(self.device),
            )
            losses = kl_loss + cls_loss

        elif self.kd == "srrl":
            cls_t = self.teacher.get_feat_modules()[-1].to(self.device)
            trans_feat_s, pred_feat_s = self.criterion[0](stu_feats["feats"][-1], cls_t)

            kl_loss_div = self.criterion[2](
                stu_logit=student_outputs,
                tea_logit=teacher_outputs,
                T=T.to(self.device),
            )
            kl_loss = self.criterion[-1](
                trans_feat_s,
                tea_feats["feats"][-1] if self.dataset == "cifar100" else tea_feats[-1],
            ) + self.criterion[-1](pred_feat_s, teacher_outputs)

            losses = (
                cls_loss * self.gamma + kl_loss * self.beta + self.alpha * kl_loss_div
            )
        elif self.kd == "vid":
            g_s = stu_feats["feats"][1:-1]
            g_t = (
                tea_feats["feats"][1:-1]
                if self.dataset == "cifar100"
                else tea_feats[1:-1]
            )
            loss_group = [
                c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, self.criterion[0])
            ]
            kl_loss = sum(loss_group) * 1 + self.criterion[2](
                student_outputs, teacher_outputs, T=T
            )

            kl_loss_div = self.criterion[2](
                stu_logit=student_outputs,
                tea_logit=teacher_outputs,
                T=T.to(self.device),
            )

            losses = (
                cls_loss * self.gamma + kl_loss * self.beta + self.alpha * kl_loss_div
            )
        elif self.kd == "pkt":
            f_s = stu_feats["feats"][-1]
            f_t = (
                tea_feats["feats"][-1] if self.dataset == "cifar100" else tea_feats[-1]
            )
            kl_loss = self.criterion[0](f_s, f_t)
            kl_loss_div = self.criterion[2](
                stu_logit=student_outputs,
                tea_logit=teacher_outputs,
                T=T.to(self.device),
            )
            losses = (
                cls_loss * self.gamma + kl_loss + self.beta + self.alpha * kl_loss_div
            )
        elif self.kd == "similarity":
            g_s = [stu_feats["feats"][-2]]
            g_t = [
                tea_feats["feats"][-2] if self.dataset == "cifar100" else tea_feats[-2]
            ]
            kl_loss = sum(self.criterion[0](g_s, g_t))

            kl_loss_div = self.criterion[2](
                stu_logit=student_outputs,
                tea_logit=teacher_outputs,
                T=T.to(self.device),
            )
            losses = (
                cls_loss * self.gamma + kl_loss * self.beta + kl_loss_div * self.alpha
            )
        elif self.kd == "crd":
            f_s = stu_feats["feats"][-1]
            f_t = (
                tea_feats["feats"][-1] if self.dataset == "cifar100" else tea_feats[-1]
            )
            kl_loss = self.criterion[0](f_s, f_t, index, contrast_idx)
            kl_loss_div = self.criterion[2](
                stu_logit=student_outputs,
                tea_logit=teacher_outputs,
                T=T.to(self.device),
            )
            losses = (
                cls_loss * self.gamma + kl_loss * self.beta + kl_loss_div * self.alpha
            )
        elif self.kd == "kd":
            kl_loss = self.criterion[0](
                stu_logit=student_outputs, tea_logit=teacher_outputs, T=T
            )
            losses = cls_loss * self.gamma + kl_loss * self.alpha
        else:
            raise NotImplementedError

        self.manual_backward(losses)
        opt.step()

        # step every N epochs
        if self.trainer.is_last_batch:
            sch.step()

        self.log_dict(
            {
                "train_loss": losses,
                "train_cls_loss": cls_loss,
                "train_kl_loss": kl_loss,
                "mean of T": T.detach().cpu().mean(),
                "median of T": T.detach().cpu().median(),
            },
            sync_dist=True,
        )
        self.acc.append(acc.detach().cpu().numpy().item())
        if len(self.acc) > 3:
            self.acc = self.acc[-3:]
        return {"loss": losses, "cls_loss": cls_loss, "kl_loss": kl_loss}

    def on_fit_start(self) -> None:
        self.threshold = 0.7

    def on_train_epoch_end(
        self,
    ):
        self.replay_buffer.reset()

            
        # export to csv
        if (self.current_epoch + 1) % self.rl_epoch == 0:
            df = pd.DataFrame(self.all_T)
            df.to_csv(f"all_T_{self.current_epoch + 1}.csv", index=False)
            
            # flush
            self.all_T = np.zeros((len(self.datamodule.train_set), 1))
        else:
            self.all_T = np.concatenate(
            [self.all_T, np.zeros((len(self.datamodule.train_set), 1))], axis=1
        )
        
        

        # self.batch_update=False
        if self.entropy_record is None:
            self.entropy_record = self.entropy
        else:
            self.entropy_record = np.concatenate(
                (self.entropy_record, self.entropy.reshape((len(self.entropy), 1))),
                axis=1,
            )

        if (
            self.aug_epoch != 0
            and (self.current_epoch + 2) > self.warmup_epoch
            and ((self.current_epoch + 2) % self.aug_epoch == 0)
        ):
            """
            purification
            """
            remain_num = int(self.threshold * len(self.datamodule.train_set))

            _entropy = np.clip(
                self.entropy_record[:, 1:], 0.0, a_max=self.entropy_record.max()
            )
            _entropy = np.sum(_entropy, axis=1)
            _sum = np.where(self.entropy_record[:1:] > 0.0, 1.0, 0.0)
            _sum = np.sum(_sum, axis=1)
            ET = np.true_divide(_sum, 40.0)
            ET = np.power(ET, 0.03)
            _entropy = np.true_divide(_entropy, _sum)
            scores = ET * _entropy
            indice = scores.argsort()[::-1]
            pre_len = len(self.datamodule.train_set)
            remain = indice[:remain_num]

            # head
            lost_len_start = int(pre_len * 0.8)
            lost_len_end = lost_len_start + int(pre_len * 0.1)
            lost = indice[lost_len_start:lost_len_end]
            # lost = indice[remain_num:pre_len]
            # lost = indice[]
            # tail
            remain_len_start = int(pre_len * 0.1)
            remain_len_end = remain_len_start + int(pre_len * 0.1)

            assert (remain_len_end - remain_len_start) == (
                lost_len_end - lost_len_start
            ), "length of remain and lost is not equal"
            mix_remain = indice[remain_len_start:remain_len_end]
            # mix_remain = remain[-len(lost) :]
            # new tail test
            # mix_remain = self.remain[: len(self.lost)]
            print("begin mixup")

            self.datamodule.train_set.mixup(_entropy, mix_remain, lost)
            if self.dataset == "cifar100":
                self.datamodule.train_set.set_aug(True)
            print("end mixup")

        else:
            if self.dataset == "cifar100":
                self.datamodule.train_set.set_aug(False)

        # rl training
        if False and (
            self.current_epoch == self.warmup_epoch
            or (self.current_epoch != 0 and self.current_epoch % self.rl_epoch == 0)
        ):
            """
            1. 根据score，将数据分为两部分，好的和坏的
            2. 根据学生网络和教师网络的logit，判断mixup的比例
            """

            # rl training
            # train the agent
            (_, act_opt, cri_opt, _, _) = self.optimizers()
            print(f"epoch: {self.current_epoch} ==> rl starts training\n")
            self.student.eval(), self.teacher.eval()
            self.actor.train(), self.critic.train()

            assert self.datamodule.train_set is not None, "train_agent_set is None"
            agent_dataloader = torch.utils.data.DataLoader(
                self.datamodule.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
            )
            loop = tqdm(
                enumerate(agent_dataloader), leave=False, total=len(agent_dataloader)
            )
            loop.set_description(f"rl epoch: {self.current_epoch}: ")
            for i, rl_batch in loop:
                # loop.set_description(f'epoch: {self.current_epoch}: ')
                inputs, targets, index = rl_batch
                if inputs.shape[0] != self.batch_size:
                    # 重复补全
                    continue
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.no_grad():
                    student_outputs = self.student(inputs)[0]
                    teacher_outputs = self.teacher(inputs)[0]

                rewards = torch.zeros((student_outputs.shape[0], 1))
                """
                有待商榷。具体是前面全是0最后才是精度还是前面是前几个图片的精度, 不好说。这得看效果。
                """
                self.train_agent(
                    student_outputs,
                    teacher_outputs,
                    targets,
                    rewards.float(),
                    act_opt=act_opt,
                    cri_opt=cri_opt,
                    index=index,
                )
            loop.close()

    def get_masks(self, obs):
        """
        输入：obs
        输出：mask
        """

        observations = torch.tensor(obs).to(self.device)
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
        # Reset gradients
        # Get outputs for network
        with torch.no_grad():
            masks = self.critical_state_detector(
                observations=observations,
            )

        return masks.squeeze(1)

    def get_prediction(self, obs_buf, parameter_act_buf, returns):
        """
        输入：replay buffer, 即包含states，actions，rewards
        输出：预测的reward
        """
        assert len(obs_buf) == len(parameter_act_buf), "length of buffer is not equal"

        observations, actions = obs_buf, parameter_act_buf

        # Get samples
        observations, actions = (
            torch.tensor(observations).to(self.device),
            torch.tensor(actions).to(self.device),
        )

        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(1)

        # Reset gradients
        # Get outputs for network
        with torch.no_grad():
            predictions = self.reward_redistribution(
                observations=observations,
                actions=actions,
            )
        redistributed_reward = (predictions[1:] - predictions[:-1]).detach().cpu()
        redistributed_reward = np.concatenate(
            [returns[:1], redistributed_reward.reshape(-1)]
        )
        predicted_returns = redistributed_reward.sum()

        prediction_error = returns.reshape(-1, 1) - predicted_returns

        # Distribute correction for prediction error equally over all sequence positions
        redistributed_reward += (
            prediction_error.reshape(-1) / redistributed_reward.shape[0]
        )

        redistributed_reward = (predictions[1:] - predictions[:-1]).detach().cpu()
        redistributed_reward = np.concatenate(
            [returns[:1], redistributed_reward.reshape(-1)]
        )
        predicted_returns = redistributed_reward.sum()

        prediction_error = returns.reshape(-1, 1) - predicted_returns

        # Distribute correction for prediction error equally over all sequence positions
        redistributed_reward += (
            prediction_error.reshape(-1) / redistributed_reward.shape[0]
        )

        return torch.tensor(redistributed_reward)

    def train_rew(
        self,
        obs_buf,
        discrete_act_buf,
        parameter_act_buf,
        adv_buf,
        ret_buf,
        discrete_logp_buf,
        parameter_logp_buf,
    ):
        """
        输入：replay buffer, 即包含states，actions，rewards


        """
        _, _, _, ret_opt, _ = self.optimizers()

        ind = 1

        assert (
            len(obs_buf)
            == len(discrete_act_buf)
            == len(parameter_act_buf)
            == len(adv_buf)
            == len(ret_buf)
            == len(discrete_logp_buf)
            == len(parameter_logp_buf)
        ), "length of buffer is not equal"

        outer_progressbar = tqdm(
            None, total=self._update, desc="rew training", leave=False
        )

        observations, actions, rewards = (
            obs_buf,
            parameter_act_buf,
            # np.concatenate([discrete_act_buf, parameter_act_buf], axis=-1),
            ret_buf,
        )

        # 课程学习的loss
        running_loss = 100.0 - self.current_epoch if self.current_epoch < 100 else 0.0

        while ind < self._update:
            ret_opt.zero_grad()

            # Get samples

            observations, actions, rewards = (
                torch.tensor(observations).to(self.device),
                torch.tensor(actions).to(self.device),
                torch.tensor(rewards).to(self.device),
            )

            if len(observations.shape) == 2:
                observations = observations.unsqueeze(1)
            if len(actions.shape) == 2:
                actions = actions.unsqueeze(1)

            # Reset gradients
            ret_opt.zero_grad()

            # Get outputs for network
            predictions = self.reward_redistribution(
                observations=observations,
                actions=actions,
            )

            # Calculate loss, do backward pass, and update
            loss = lossfunction_rew(predictions[..., 0], rewards)
            loss.backward()
            running_loss = running_loss * 0.99 + loss * 0.01
            ret_opt.step()
            outer_progressbar.update(1)
            outer_progressbar.set_description(
                "LSTM training (loss=%.4f)" % (running_loss)
            )
            ind += 1

        outer_progressbar.close()

    def get_observations(
        self,
        stu_logit: torch.Tensor,
        tea_logit: torch.Tensor,
        gt: torch.Tensor,
        index: torch.Tensor,
    ):
        stu_prob, tea_prob = F.softmax(stu_logit, dim=1), F.softmax(tea_logit, dim=1)
        stu_top1 = torch.max(stu_prob, dim=1)[0]
        stu_top2 = torch.kthvalue(stu_prob, 2, dim=1)[0]
        tea_top1 = torch.max(tea_prob, dim=1)[0]
        tea_top2 = torch.kthvalue(tea_prob, 2, dim=1)[0]

        # last_epoch_top1 = self.last_stu_top1[gt]

        # if self.last_stu_top1 is None:
        #     self.last_stu_top1 = stu_top1
        # obs = (stu_top1 / self.last_stu_top1[: stu_top1.shape[0]]).reshape(-1, 1).unsqueeze(0)
        # obs = stu_top1.reshape(-1, 1).unsqueeze(0)
        # obs = tea_top1.reshape(-1, 1).unsqueeze(0)
        if self.obs_dim == 3:
            obs = torch.cat(
                [
                    stu_top1.reshape(-1, 1),
                    tea_top1.reshape(-1, 1),
                    (stu_top1 - stu_top2).reshape(-1, 1),
                    # (tea_top1 - tea_top2).reshape(-1, 1),
                    # (stu_top1 / last_epoch_top1).reshape(-1, 1),
                ],
                dim=1,
            )
        elif self.obs_dim == 2:
            obs = torch.cat(
                [
                    stu_top1.reshape(-1, 1),
                    tea_top1.reshape(-1, 1),
                    # (stu_top1 - stu_top2).reshape(-1, 1),
                    # (tea_top1 - tea_top2).reshape(-1, 1),
                    # (stu_top1 / last_epoch_top1).reshape(-1, 1),
                ],
                dim=1,
            )

        elif self.obs_dim == 4:
            obs = torch.cat(
                [
                    stu_top1.reshape(-1, 1),
                    tea_top1.reshape(-1, 1),
                    (stu_top1 - stu_top2).reshape(-1, 1),
                    (tea_top1 - tea_top2).reshape(-1, 1),
                ],
                dim=1,
            )
        else:
            raise NotImplementedError

        return obs

    def train_agent(
        self,
        stu_logit: torch.Tensor,
        tea_logit: torch.Tensor,
        gt: torch.Tensor,
        rewards: torch.Tensor,
        act_opt,
        cri_opt,
        index,
    ) -> np.array:
        # 得到observations
        obs = self.get_observations(stu_logit, tea_logit, gt, index=index)

        total_steps = 0

        actions = []
        state = None

        # explore the world

        kl_loss = KL_Loss()
        cls_loss = nn.CrossEntropyLoss()
        while total_steps < stu_logit.shape[0]:
            state = obs[total_steps, :]
            (
                parameter_action,
                _,
                parameter_logp_t,
            ) = self.select_action(state)

            # discrete_logp_t = np.max(prob_discrete_action)
            v_t = self.get_value(state)

            if total_steps + 1 == stu_logit.shape[0]:
                rewards[-1] = -torch.log(
                    kl_loss(stu_logit, tea_logit, parameter_action)
                    + cls_loss(stu_logit, gt)
                )
            else:
                rewards[total_steps] = 0.0

            self.replay_buffer.add(
                obs=state.detach().cpu().numpy(),
                discrete_action=0.0,
                parameter_action=parameter_action,
                rew=rewards[total_steps],
                val=v_t,
                discrete_logp=0.0,
                parameter_logp=parameter_logp_t,
            )
            # print(discrete_action, parameter_action)
            actions.append(np.array([0.0, parameter_action], dtype=np.float32))

            if (total_steps + 1) % self.is_update == 0:
                # update
                state = state.reshape(1, -1).float().to(self.device)
                with torch.no_grad():
                    value = self.critic(state)
                last_vals = value.cpu().numpy().squeeze(0)
                self.replay_buffer.finish_path(last_vals)

                (
                    obs_buf,
                    discrete_act_buf,
                    parameter_act_buf,
                    adv_buf,
                    ret_buf,
                    discrete_logp_buf,
                    parameter_logp_buf,
                ) = self.replay_buffer.get()

                self.train_det(
                    obs_buf,
                    rewards=ret_buf,
                    actions=parameter_act_buf,
                )
                masks = self.get_masks(obs_buf)
                masks = masks.detach().cpu().numpy()

                self.train_rew(
                    obs_buf,
                    discrete_act_buf,
                    parameter_act_buf,
                    adv_buf,
                    ret_buf,
                    discrete_logp_buf,
                    parameter_logp_buf,
                )

                rew = self.get_prediction(obs_buf, parameter_act_buf, ret_buf)
                self.replay_buffer.obs_buf = obs_buf * masks

                self.replay_buffer.rew_buf = rew.detach().cpu().numpy()
                self.replay_buffer.finish_path(last_vals)

                self.update_agent(act_opt, cri_opt)
                self.replay_buffer.reset()

            total_steps += 1

        """
        deal with actions
        [augmentation, T]
        """
        actions = torch.from_numpy(np.array(actions))  # [timesteps_per_batch, 2]
        return actions

    def train_det(self, obs, rewards, actions):
        _, _, _, _, det_opt = self.optimizers()
        outer_progressbar = tqdm(
            None, total=self._update, desc="DET training", leave=False
        )

        ind = 0
        while ind < self._update:
            det_opt.zero_grad()
            # Get samples
            observations, rewards = (
                torch.tensor(obs).to(self.device),
                torch.tensor(rewards).to(self.device),
            )

            if len(observations.shape) == 2:
                observations = observations.unsqueeze(1)
            # Reset gradients
            det_opt.zero_grad()

            # Get outputs for network
            masks = self.critical_state_detector(
                observations=observations,
            )

            masks = masks.squeeze(1)

            # Calculate loss, do backward pass, and update
            loss_1 = torch.linalg.norm(masks, ord=1, dim=-1).sum()
            # masked_obs = masks * observations.squeeze(1)
            # masked_obs = masked_obs.unsqueeze(1)
            # predictions=self.get_prediction(masked_obs, actions)
            masked_obs = masks * observations.mean(dim=-1)
            loss_2 = F.cross_entropy(masked_obs, rewards.unsqueeze(-1))
            # loss_2 = F.cross_entropy(predictions, rewards.unsqueeze(-1))

            loss = loss_1 * 5e-3 + loss_2 * 1.0
            loss.backward()
            det_opt.step()
            outer_progressbar.update(1)
            outer_progressbar.set_description("LSTM training (loss=%.4f)" % (loss))
            ind += 1

        outer_progressbar.close()

    def get_action(
        self,
        stu_logit: torch.Tensor,
        tea_logit: torch.Tensor,
        gt: torch.Tensor,
        index: torch.tensor,
    ):
        # 得到observations
        obs = self.get_observations(stu_logit, tea_logit, gt, index=index)
        total_steps = 0
        actions = []

        # explore the world
        self.actor.eval()
        while total_steps < stu_logit.shape[0]:
            state = obs[total_steps, :]
            with torch.no_grad():
                (
                    # prob_discrete_action,
                    # q_action,
                    t_action,
                    _,
                    parameter_logp_t,
                ) = self.select_action(state)

                actions.append(np.array([0.0, t_action], dtype=np.float32))
            total_steps += 1

        actions = torch.from_numpy(np.array(actions))  # [timesteps_per_batch, 2]
        return actions

    def get_value(self, state):
        state = torch.FloatTensor(state.detach().cpu().reshape(1, -1)).to(self.device)
        # state = state.reshape(1, -1).float()
        with torch.no_grad():
            value = self.critic(state)
        return value.cpu().data.numpy().squeeze(0)

    def select_action(self, state: torch.Tensor):
        state = state.to(self.device)
        with torch.no_grad():
            mu, std, log_std = self.actor(state)
        # dist = distributions.Normal(mu, std)
        # pi = dist.sample()
        # 离散 softmax 然后在分布中选择

        # 连续 均值方差分
        # dist = distributions.Normal(mu, std)
        # pi = dist.sample()

        noise = torch.FloatTensor(np.random.normal(0, 1, size=std.size())).to(
            self.device
        )
        pi = mu + noise * std

        """
        cos vs sigmoid
        """
        # pi = pi.clamp(-4, 4)
        # pi = 1 +(torch.sin(torch.pi*0.125 * pi) + 1) / 2 * (10-1)
        # pi = pi.clamp(-4,4)
        # print(f'pi: {pi}, noise: {noise}')
        pi = torch.sigmoid(pi) * 10  # 0~10

        parameter_action = pi * self.max_action

        logp = gaussian_likelihood(pi, mu, log_std)

        return (
            # q_action.cpu().numpy().flatten(),
            # q_action.cpu().item(),
            parameter_action.cpu().item(),
            pi.cpu().item(),
            logp.cpu().data.numpy().flatten(),
        )

        # return prob_discrete_action, discrete_action, parameter_action, raw_act, parameter_logp_t

    def update_v(self, x, y, batch_size, cri_opt):
        """Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            batch_size
            cri_opt: critic optimizer
        """
        num_batches = max(x.shape[0] // batch_size, 1)
        batch_size = x.shape[0] // num_batches
        x_train, y_train = shuffle(x, y)

        losses = 0
        for j in range(num_batches):
            start = j * batch_size
            end = (j + 1) * batch_size
            b_x = torch.FloatTensor(x_train[start:end]).to(self.device)
            b_y = torch.FloatTensor(y_train[start:end].reshape(-1, 1)).to(self.device)

            v_loss = F.mse_loss(self.critic(b_x), b_y)
            cri_opt.zero_grad()
            self.manual_backward(v_loss)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
            cri_opt.step()

            losses += v_loss.cpu().data.numpy()

        return losses / num_batches

    def update_agent(self, act_opt, cri_opt):
        (
            obs_buf,
            discrete_act_buf,
            parameter_act_buf,
            adv_buf,
            ret_buf,
            discrete_logp_buf,
            parameter_logp_buf,
        ) = self.replay_buffer.get()

        # with open('obs_buf.txt', 'a') as f:
        # f.write(f'epoch: {self.current_epoch+1}, obs_buf={obs_buf}, discrete_act_buf={discrete_act_buf}, parameter_act_buf={parameter_act_buf}, adv_buf={adv_buf}, ret_buf={ret_buf}, discrete_logp_buf={discrete_logp_buf}, parameter_logp_buf={parameter_logp_buf}\n')

        c_loss_list, a_loss_list, q_a_loss_list, t_a_loss_list = [], [], [], []
        for _ in range(self.c_epoch):
            c_loss = self.update_v(
                obs_buf, ret_buf, batch_size=self.is_update, cri_opt=cri_opt
            )
            c_loss_list.append(c_loss)

        obss = torch.FloatTensor(obs_buf).to(self.device)

        # discrete_act_buf = torch.FloatTensor(discrete_act_buf).to(self.device)
        parameter_act_buf = torch.FloatTensor(parameter_act_buf).to(self.device)

        advs = torch.FloatTensor(adv_buf).view(-1, 1).to(self.device)
        # discrete_logp_olds = (
        #     torch.FloatTensor(discrete_logp_buf).view(-1, 1).to(self.device)
        # )
        parameter_logp_olds = (
            torch.FloatTensor(parameter_logp_buf).view(-1, 1).to(self.device)
        )

        for _ in range(self.a_epoch):
            # q_action, mu, std, parameter_log_std = self.actor(obss)
            mu, std, parameter_log_std = self.actor(obss)

            # 图片质量
            # q_logp_t = q_action.gather(1, discrete_act_buf.long())
            # q_ratio = torch.exp(q_logp_t - discrete_logp_olds)

            # q_L1 = q_ratio * advs
            # q_L2 = (
            #     torch.clamp(q_ratio, 1 - self.clipping_ratio, 1 + self.clipping_ratio)
            #     * advs
            # )
            # q_a_loss = -torch.min(q_L1, q_L2).mean()
            # q_a_loss_list.append(q_a_loss.cpu().data.numpy())

            t_logp = gaussian_likelihood(parameter_act_buf, mu, parameter_log_std)

            t_ratio = torch.exp(t_logp - parameter_logp_olds)

            t_L1 = t_ratio * advs
            t_L2 = (
                torch.clamp(t_ratio, 1 - self.clipping_ratio, 1 + self.clipping_ratio)
                * advs
            )
            t_a_loss = -torch.min(t_L1, t_L2).mean()
            t_a_loss_list.append(t_a_loss.cpu().data.numpy())
            # loss 离散部分和连续部分相加
            a_loss = t_a_loss
            # a_loss = q_a_loss + t_a_loss

            # with open('update.txt', 'a') as f:
            #     f.write(
            #         f'a_loss={a_loss}, advs={advs}, discrete_L2={discrete_L2}, discrete_L1={discrete_L1}, discrete_a_loss={discrete_a_loss}, parameter_a_loss={parameter_a_loss}, a_loss={a_loss}, discrete_ratio={discrete_ratio}, parameter_ratio={parameter_ratio}\n')

            act_opt.zero_grad()
            self.manual_backward(a_loss)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
            act_opt.step()

            a_loss_list.append(a_loss.cpu().data.numpy())

        losses = {
            "c_loss": sum(c_loss_list) / self.c_epoch,
            "a_loss": sum(a_loss_list) / self.a_epoch,
            # "q_a_loss": sum(q_a_loss_list) / self.a_epoch,
            "t_a_loss": sum(t_a_loss_list) / self.a_epoch,
            "mean of rewards": ret_buf.mean(),
        }

        self.log_dict(losses, sync_dist=True)

    def configure_optimizers(
        self,
    ):
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.student.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestone, gamma=0.1
            )

            actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
            critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
            ret_optimizer = torch.optim.Adam(
                self.reward_redistribution.parameters(), lr=1e-3
            )
            state_optimizer = torch.optim.Adam(
                self.critical_state_detector.parameters(), lr=1e-3
            )
            return [
                optimizer,
                actor_optimizer,
                critic_optimizer,
                ret_optimizer,
                state_optimizer,
            ], [scheduler]

        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestone, gamma=0.1
            )
            return [optimizer, actor_optimizer, critic_optimizer], [scheduler]
        elif self.optimizers == "AdamW":
            optimizer = torch.optim.AdamW(
                self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestone, gamma=0.1
            )
            return [optimizer, actor_optimizer, critic_optimizer], [scheduler]


class CTKDDistiller(BaseDistiller):
    def __init__(
        self,
        student,
        teacher,
        kd_criterion,
        cls_criterion,
        optimizer,
        scheduler,
        **kwargs,
    ):
        super().__init__(
            student,
            teacher,
            kd_criterion,
            cls_criterion,
            optimizer,
            scheduler,
            **kwargs,
        )

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)


class MKDDistiller(BaseDistiller):
    def __init__(
        self,
        student,
        teacher,
        kd_criterion,
        cls_criterion,
        optimizer,
        scheduler,
        **kwargs,
    ):
        super().__init__(
            student,
            teacher,
            kd_criterion,
            cls_criterion,
            optimizer,
            scheduler,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        input, target, _ = batch
        student_outputs = self.student(input)[0]
        with torch.no_grad():
            teacher_outputs = self.teacher(input)[0]

        cls_loss = self.criterion[1](student_outputs, target)
        kl_loss = 0.0
        kl_loss, _ = self.criterion[0](student_outputs, teacher_outputs, target)

        losses = cls_loss + kl_loss
        self.log_dict(
            {
                "train_loss": losses,
                "train_cls_loss": cls_loss,
                "train_kl_loss": kl_loss,
            },
            sync_dist=True,
        )

        return {"loss": losses, "cls_loss": cls_loss, "kl_loss": kl_loss}


class MLPDistiller(BaseDistiller):
    def __init__(
        self,
        student,
        teacher,
        kd_criterion,
        cls_criterion,
        optimizer,
        scheduler,
        **kwargs,
    ):
        super().__init__(
            student,
            teacher,
            kd_criterion,
            cls_criterion,
            optimizer,
            scheduler,
            **kwargs,
        )

        self.criterion = [kd_criterion.to(self.device), cls_criterion]
        self.decay_stragegy = kwargs["decay_stragegy"]

    def get_decay_value(
        self,
    ):
        max_l = 0
        min_l = -1
        max_loop = 100
        if self.decay_stragegy == "cosine":
            if self.current_epoch < max_loop:
                decay_value = (
                    min_l
                    + (max_l - min_l)
                    * (1 + np.cos(np.pi * self.current_epoch / max_loop))
                    / 2
                )
            else:
                decay_value = min_l
            return decay_value
        elif self.decay_stragegy == "linear":
            if self.current_epoch < max_loop:
                decay_value = min_l + (max_l - min_l) * (
                    1 - self.current_epoch / max_loop
                )
            else:
                decay_value = min_l
            return decay_value
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        input, target, _ = batch
        student_outputs = self.student(input)
        if type(student_outputs) == tuple:
            student_outputs = student_outputs[0]
        with torch.no_grad():
            teacher_outputs = self.teacher(input)
            if type(teacher_outputs) == tuple:
                teacher_outputs = teacher_outputs[0]

        decay_value = self.get_decay_value()
        cls_loss = self.criterion[1](student_outputs, target)
        kl_loss, T = self.criterion[0](
            tea_logit=teacher_outputs,
            stu_logit=student_outputs,
            gt=target,
            device=self.device,
            decay_value=decay_value,
        )

        # warmup kl_loss
        # kl_loss = min((self.current_epoch+1) / 20, 1.0) * kl_loss

        losses = cls_loss + kl_loss
        self.log_dict(
            {
                "train_loss": losses,
                "train_cls_loss": cls_loss,
                "train_kl_loss": kl_loss,
                "mean of T": T.mean(),
                "median of T": T.median(),
            },
            sync_dist=True,
        )

        return {"loss": losses, "cls_loss": cls_loss, "kl_loss": kl_loss}


class MLPDKDDistiller(MLPDistiller):
    def __init__(
        self,
        student,
        teacher,
        kd_criterion,
        cls_criterion,
        optimizer,
        scheduler,
        **kwargs,
    ):
        super().__init__(
            student,
            teacher,
            kd_criterion,
            cls_criterion,
            optimizer,
            scheduler,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        input, target, _ = batch
        student_outputs = self.student(input)
        if type(student_outputs) == tuple:
            student_outputs = student_outputs[0]
        with torch.no_grad():
            teacher_outputs = self.teacher(input)
            if type(teacher_outputs) == tuple:
                teacher_outputs = teacher_outputs[0]

        decay_value = self.get_decay_value()
        cls_loss = self.criterion[1](student_outputs, target)
        kl_loss, T = self.criterion[0](
            tea_logit=teacher_outputs,
            stu_logit=student_outputs,
            gt=target,
            device=self.device,
            decay_value=decay_value,
        )

        # warmup kl_loss
        # kl_loss = min((self.current_epoch+1) / 20, 1.0) * kl_loss

        losses = cls_loss + kl_loss
        self.log_dict(
            {
                "train_loss": losses,
                "train_cls_loss": cls_loss,
                "train_kl_loss": kl_loss,
                "mean of T": T.mean(),
                "median of T": T.median(),
            },
            sync_dist=True,
        )

        return {"loss": losses, "cls_loss": cls_loss, "kl_loss": kl_loss}
