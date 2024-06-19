import torch
import torch.nn as nn
import torch.nn.functional as F


def self_dkd_loss(
    logits_student,
    logits_teacher,
    target,
    alpha,
    beta,
    temperature,
    epoch,
    warmup_epoch,
):
    tckd_T = 1.0
    nckd_T = temperature * F.sigmoid(torch.Tensor([epoch - warmup_epoch * 3])).item()
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / tckd_T, dim=1)
    pred_teacher = F.softmax(logits_teacher / tckd_T, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (tckd_T**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(logits_teacher / nckd_T - 1000.0 * gt_mask, dim=1)
    log_pred_student_part2 = F.log_softmax(
        logits_student / nckd_T - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (nckd_T**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    # T = T.reshape(-1,1)
    # kd_loss = F.kl_div(F.log_softmax(stu_logit/T, dim=1), F.softmax(tea_logit/T, dim=1), reduction='none').sum(dim=1)
    # kd_loss = kd_loss*T*T
    # return kd_loss.mean()

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)

    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(dim=1)
        * (temperature**2)
    ).mean()

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="none").sum(
            dim=1
        )
        * (temperature**2)
    ).mean()
    return alpha * tckd_loss + beta * nckd_loss


def mlp_dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(
        dim=1
    ) * (temperature**2)

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = F.kl_div(
        log_pred_student_part2, pred_teacher_part2, reduction="none"
    ).sum(dim=1) * (temperature**2)

    return alpha * tckd_loss.mean() + beta * nckd_loss.mean()


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
