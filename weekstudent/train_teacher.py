import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.teacher import get_teacher


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(dataset_name, data_root, batch_size):
    if dataset_name == "cifar10":
        num_classes = 10
        dataset_class = datasets.CIFAR10
    else:
        num_classes = 100
        dataset_class = datasets.CIFAR100
    transform_train = transforms.Compose(
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
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.247, 0.243, 0.261],
            ),
        ]
    )
    train_dataset = dataset_class(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = dataset_class(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader, num_classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
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


def evaluate(model, loader, criterion, device):
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


def train_teacher(args):
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = args.log_file
    if not log_path:
        log_filename = f"teacher_{args.dataset}_{args.experiment}.log"
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

    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset,
        args.data_root,
        args.batch_size,
    )
    model = get_teacher(num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
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
    train_accs = []
    train_top5_accs = []
    val_losses = []
    val_accs = []
    val_top5_accs = []
    best_acc = 0.0
    no_improve_epochs = 0
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir,
        f"teacher_{args.dataset}_best.pth",
    )
    log(
        f"Start teacher training seed={args.seed} "
        f"dataset={args.dataset} experiment={args.experiment} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}",
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_top5 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc, val_top5 = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
        scheduler.step()
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_top5_accs.append(train_top5)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_top5_accs.append(val_top5)
        log(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_top5={train_top5:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f}",
        )
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                },
                save_path,
            )
            log(
                f"New best val_acc={best_acc:.4f} "
                f"model saved to {save_path}",
            )
        else:
            no_improve_epochs += 1
        if args.patience > 0 and no_improve_epochs >= args.patience:
            log(
                f"Early stopping at epoch {epoch} "
                f"no_improve_epochs={no_improve_epochs}",
            )
            break
    history_path = os.path.join(
        args.save_dir,
        f"teacher_{args.dataset}_history.npz",
    )
    np.savez_compressed(
        history_path,
        train_losses=np.array(train_losses, dtype=np.float32),
        train_accs=np.array(train_accs, dtype=np.float32),
        train_top5_accs=np.array(train_top5_accs, dtype=np.float32),
        val_losses=np.array(val_losses, dtype=np.float32),
        val_accs=np.array(val_accs, dtype=np.float32),
        val_top5_accs=np.array(val_top5_accs, dtype=np.float32),
    )
    log(f"Training history saved to {history_path}")
    log("Teacher training finished")
    log(f"Logs written to {args.log_file}")
    if log_file is not None:
        log_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="train_teachermodel",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
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
        "--save-dir",
        type=str,
        default="./checkpoints",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
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
    train_teacher(args)
