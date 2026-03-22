import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model.teacher import get_teacher
from model.student import get_student


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_dataset(dataset_name, data_root):
    if dataset_name == "cifar10":
        num_classes = 10
        dataset_class = datasets.CIFAR10
    else:
        num_classes = 100
        dataset_class = datasets.CIFAR100
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.247, 0.243, 0.261],
            ),
        ]
    )
    dataset = dataset_class(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    return dataset, num_classes


def kd_loss_fn(student_logits, teacher_logits, temperature):
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(log_p, q, reduction="batchmean") * temperature * temperature
    return loss


def dkd_loss_fn(student_logits, teacher_logits, targets, temperature, t_weight, n_weight):
    batch_size = targets.shape[0]
    gt_mask = _get_gt_mask(student_logits, targets)
    other_mask = _get_other_mask(student_logits, targets)
    pred_student = F.softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
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


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdim=True)
    t2 = (t * mask2).sum(dim=1, keepdim=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def train_one_epoch_distill(
    student,
    teacher,
    loader,
    optimizer,
    ce_criterion,
    method,
    temperature,
    alpha,
    dkd_t_weight,
    dkd_n_weight,
    device,
    epoch,
    warmup,
):
    student.train()
    teacher.eval()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        if method == "KD":
            loss_kd = kd_loss_fn(student_logits, teacher_logits, temperature)
            loss = alpha * loss_kd
        else:
            loss_dkd = dkd_loss_fn(
                student_logits,
                teacher_logits,
                targets,
                temperature,
                dkd_t_weight,
                dkd_n_weight,
            )
            if warmup > 0:
                dkd_weight = min(float(epoch) / float(warmup), 1.0)
            else:
                dkd_weight = 1.0
            loss = dkd_weight * loss_dkd
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = student_logits.max(1)
        _, top5_pred = student_logits.topk(5, 1, True, True)
        correct_top1 = predicted.eq(targets)
        correct_top5 = top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred))
        total += targets.size(0)
        correct1 += correct_top1.sum().item()
        correct5 += correct_top5.sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct1 / total
    epoch_top5 = correct5 / total
    return epoch_loss, epoch_acc, epoch_top5


def evaluate_student(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_top1 = predicted.eq(targets)
            correct_top5 = top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred))
            total += targets.size(0)
            correct1 += correct_top1.sum().item()
            correct5 += correct_top5.sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct1 / total
    epoch_top5 = correct5 / total
    return epoch_loss, epoch_acc, epoch_top5


def run_single_experiment(
    dataset_name,
    ratio,
    method,
    dataset,
    test_loader,
    indices,
    num_classes,
    args,
    device,
    log,
):
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    teacher = get_teacher(num_classes=num_classes)
    student = get_student(num_classes=num_classes)
    teacher_ckpt_path = os.path.join(
        args.teacher_ckpt_dir,
        f"teacher_{dataset_name}_best.pth",
    )
    checkpoint = torch.load(teacher_ckpt_path, map_location="cpu")
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher = teacher.to(device)
    student = student.to(device)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
        gamma=0.1,
    )
    train_losses = []
    accs = []
    train_top5_accs = []
    test_losses = []
    test_accs = []
    test_top5_accs = []
    best_acc = 0.0
    no_improve_epochs = 0
    log(
        f"Start distill dataset={dataset_name} ratio={ratio:.1f} "
        f"method={method} samples={len(indices)} epochs={args.epochs}",
    )
    for epoch in range(1, args.epochs + 1):
        epoch_loss, epoch_acc, epoch_top5 = train_one_epoch_distill(
            student,
            teacher,
            loader,
            optimizer,
            ce_criterion,
            method,
            args.temperature,
            args.alpha,
            args.dkd_t_weight,
            args.dkd_n_weight,
            device,
            epoch,
            args.warmup,
        )
        test_loss, test_acc, test_top5 = evaluate_student(
            student,
            test_loader,
            ce_criterion,
            device,
        )
        scheduler.step()
        train_losses.append(epoch_loss)
        accs.append(epoch_acc)
        train_top5_accs.append(epoch_top5)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_top5_accs.append(test_top5)
        if test_acc > best_acc:
            best_acc = test_acc
            no_improve_epochs = 0
            log(
                f"[{dataset_name}] ratio={ratio:.1f} method={method} "
                f"new best test_acc={best_acc:.4f} at epoch={epoch}",
            )
        else:
            no_improve_epochs += 1
        if args.patience > 0 and no_improve_epochs >= args.patience:
            log(
                f"[{dataset_name}] ratio={ratio:.1f} method={method} "
                f"early stopping at epoch={epoch} no_improve_epochs={no_improve_epochs}",
            )
            break
        log(
            f"[{dataset_name}] ratio={ratio:.1f} method={method} "
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f} train_top5={epoch_top5:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_top5={test_top5:.4f}",
        )
    os.makedirs(args.save_dir, exist_ok=True)
    history_dir = os.path.join(args.save_dir, "distill_history")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(
        history_dir,
        f"{dataset_name}_ratio{ratio:.1f}_{method}.npz",
    )
    np.savez_compressed(
        history_path,
        train_losses=np.array(train_losses, dtype=np.float32),
        accs=np.array(accs, dtype=np.float32),
        train_top5_accs=np.array(train_top5_accs, dtype=np.float32),
        test_losses=np.array(test_losses, dtype=np.float32),
        test_accs=np.array(test_accs, dtype=np.float32),
        test_top5_accs=np.array(test_top5_accs, dtype=np.float32),
    )
    log(
        f"History saved to {history_path}",
    )


def run_all_experiments(args):
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)
    ratios = [i / 10.0 for i in range(10, 0, -1)]
    methods = ["KD", "DKD"]
    datasets_to_run = []
    if args.run_cifar10:
        datasets_to_run.append("cifar10")
    if args.run_cifar100:
        datasets_to_run.append("cifar100")
    if not datasets_to_run:
        datasets_to_run.append("cifar10")

    log_path = args.log_file
    if not log_path:
        datasets_str = "-".join(datasets_to_run)
        methods_str = "-".join(methods)
        log_filename = f"distill_{datasets_str}_{methods_str}.log"
        log_path = os.path.join("logs", log_filename)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    args.log_file = log_path

    def log(message):
        print(message)
        if log_file is not None:
            log_file.write(str(message) + "\n")
            log_file.flush()

    log("Hyperparameters:")
    for key, value in sorted(vars(args).items()):
        log(f"{key}={value}")

    log(
        f"Start distillation experiments seed={args.seed} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} "
        f"temperature={args.temperature} alpha={args.alpha} beta={args.beta}",
    )
    log(
        f"datasets={datasets_to_run} ratios={ratios} methods={methods}",
    )
    for dataset_name in datasets_to_run:
        dataset, num_classes = get_base_dataset(
            dataset_name,
            args.data_root,
        )
        num_samples = len(dataset)
        if dataset_name == "cifar10":
            dataset_class = datasets.CIFAR10
        else:
            dataset_class = datasets.CIFAR100
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.247, 0.243, 0.261],
                ),
            ]
        )
        test_dataset = dataset_class(
            root=args.data_root,
            train=False,
            download=True,
            transform=transform_test,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        all_indices = np.arange(num_samples)
        rng.shuffle(all_indices)
        for ratio in ratios:
            count = max(1, int(num_samples * ratio))
            selected_indices = all_indices[:count]
            for method in methods:
                run_single_experiment(
                    dataset_name,
                    ratio,
                    method,
                    dataset,
                    test_loader,
                    selected_indices,
                    num_classes,
                    args,
                    device,
                    log,
                )
    log("All distillation experiments finished")
    log(f"Logs written to {args.log_file}")
    if log_file is not None:
        log_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="distill",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--teacher-ckpt-dir",
        type=str,
        default="./checkpoints",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--dkd-t-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--dkd-n-weight",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--run-cifar10",
        action="store_true",
    )
    parser.add_argument(
        "--run-cifar100",
        action="store_true",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_experiments(args)

