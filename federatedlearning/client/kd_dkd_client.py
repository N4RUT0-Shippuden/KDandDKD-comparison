import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from federatedlearning.model.teacher import get_teacher
from federatedlearning.model.student import get_student


def kd_loss_fn(student_logits, teacher_logits, temperature):
    q = F.softmax(teacher_logits / temperature, dim=1)
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(log_p, q, reduction="batchmean") * temperature * temperature
    return loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def _cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdim=True)
    t2 = (t * mask2).sum(dim=1, keepdim=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss_fn(student_logits, teacher_logits, targets, temperature, t_weight, n_weight):
    batch_size = targets.shape[0]
    gt_mask = _get_gt_mask(student_logits, targets)
    other_mask = _get_other_mask(student_logits, targets)
    pred_student = F.softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
    pred_student = _cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = _cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    loss_tckd = (
        F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        * (temperature**2)
        / batch_size
    )
    pred_teacher_part2 = F.softmax(
        teacher_logits / temperature - 1000.0 * gt_mask,
        dim=1,
    )
    log_pred_student_part2 = F.log_softmax(
        student_logits / temperature - 1000.0 * gt_mask,
        dim=1,
    )
    loss_nckd = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="sum")
        * (temperature**2)
        / batch_size
    )
    return t_weight * loss_tckd + n_weight * loss_nckd


class KDDKDClient:
    def __init__(
        self,
        client_id,
        train_dataset,
        num_classes=100,
        method="KD",
        batch_size=64,
        lr=0.05,
        temperature=4.0,
        tckd_weight=1.0,
        nckd_weight=8.0,
        kd_weight=1.0,
        dkd_weight=1.0,
        global_rounds=200,
        warmup_rounds=20,
        device=None,
        num_workers=2,
    ):
        self.client_id = client_id
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.method = str(method).upper()
        self.teacher = get_teacher(num_classes=num_classes).to(self.device)
        self.student = get_student(num_classes=num_classes).to(self.device)
        self.temperature = temperature
        self.tckd_weight = tckd_weight
        self.nckd_weight = nckd_weight
        self.batch_size = batch_size
        self.lr = lr
        self.kd_weight = kd_weight
        self.dkd_weight = dkd_weight
        self.global_rounds = int(global_rounds)
        self.warmup_rounds = warmup_rounds
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.student_optimizer = torch.optim.SGD(
            self.student.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        self.teacher_optimizer = torch.optim.SGD(
            self.teacher.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

    def set_global_student_params(self, global_state_dict):
        self.student.load_state_dict(global_state_dict, strict=True)

    def get_student_params(self):
        return self.student.state_dict()

    def get_teacher_params(self):
        return self.teacher.state_dict()

    def _compute_dkd_coeff(self, global_round):
        r = max(1, int(global_round))
        if self.warmup_rounds <= 0:
            return 1.0
        ratio = float(min(r, self.warmup_rounds)) / float(self.warmup_rounds)
        return ratio

    def local_train_one_epoch(self, global_round):
        self.teacher.train()
        self.student.train()
        dkd_coeff = self._compute_dkd_coeff(global_round)
        total_student_loss = 0.0
        total_teacher_loss = 0.0
        total_samples = 0
        total_correct1 = 0
        total_correct5 = 0
        total_teacher_correct1 = 0
        total_teacher_correct5 = 0
        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            teacher_logits = self.teacher(images)
            student_logits = self.student(images)
            if self.method == "KD":
                student_kd = kd_loss_fn(
                    student_logits,
                    teacher_logits,
                    self.temperature,
                )
                teacher_kd = kd_loss_fn(
                    teacher_logits,
                    student_logits,
                    self.temperature,
                )
                student_loss = self.kd_weight * student_kd
                teacher_loss = self.kd_weight * teacher_kd
            elif self.method == "DKD":
                student_dkd = dkd_loss_fn(
                    student_logits,
                    teacher_logits,
                    targets,
                    self.temperature,
                    self.tckd_weight,
                    self.nckd_weight,
                )
                teacher_dkd = dkd_loss_fn(
                    teacher_logits,
                    student_logits,
                    targets,
                    self.temperature,
                    self.tckd_weight,
                    self.nckd_weight,
                )
                student_loss = self.dkd_weight * dkd_coeff * student_dkd
                teacher_loss = self.dkd_weight * dkd_coeff * teacher_dkd
            else:
                student_kd = kd_loss_fn(
                    student_logits,
                    teacher_logits,
                    self.temperature,
                )
                teacher_kd = kd_loss_fn(
                    teacher_logits,
                    student_logits,
                    self.temperature,
                )
                student_loss = self.kd_weight * student_kd
                teacher_loss = self.kd_weight * teacher_kd
            batch_size = targets.size(0)
            total_student_loss += student_loss.item() * batch_size
            total_teacher_loss += teacher_loss.item() * batch_size
            total_samples += batch_size
            _, predicted = student_logits.max(1)
            _, top5_pred = student_logits.topk(5, 1, True, True)
            correct_top1 = predicted.eq(targets)
            correct_top5 = top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred))
            total_correct1 += correct_top1.sum().item()
            total_correct5 += correct_top5.sum().item()
            _, teacher_pred = teacher_logits.max(1)
            _, teacher_top5_pred = teacher_logits.topk(5, 1, True, True)
            teacher_correct_top1 = teacher_pred.eq(targets)
            teacher_correct_top5 = teacher_top5_pred.eq(
                targets.view(-1, 1).expand_as(teacher_top5_pred),
            )
            total_teacher_correct1 += teacher_correct_top1.sum().item()
            total_teacher_correct5 += teacher_correct_top5.sum().item()
            self.student_optimizer.zero_grad()
            self.teacher_optimizer.zero_grad()
            student_loss.backward(retain_graph=True)
            teacher_loss.backward()
            self.student_optimizer.step()
            self.teacher_optimizer.step()
        if total_samples > 0:
            avg_student_loss = total_student_loss / total_samples
            avg_teacher_loss = total_teacher_loss / total_samples
            top1 = float(total_correct1) / float(total_samples)
            top5 = float(total_correct5) / float(total_samples)
            teacher_top1 = float(total_teacher_correct1) / float(total_samples)
            teacher_top5 = float(total_teacher_correct5) / float(total_samples)
        else:
            avg_student_loss = 0.0
            avg_teacher_loss = 0.0
            top1 = 0.0
            top5 = 0.0
            teacher_top1 = 0.0
            teacher_top5 = 0.0
        stats = {
            "student_loss": avg_student_loss,
            "teacher_loss": avg_teacher_loss,
            "student_top1": top1,
            "student_top5": top5,
            "teacher_top1": teacher_top1,
            "teacher_top5": teacher_top5,
        }
        return self.get_student_params(), self.get_teacher_params(), stats
