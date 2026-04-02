import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model.teacher import get_teacher


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(data_root):
    num_classes = 100
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
    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )
    return train_dataset, test_dataset, num_classes


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
        total += targets.size(0)
        correct1 += predicted.eq(targets).sum().item()
        correct5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()
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
            total += targets.size(0)
            correct1 += predicted.eq(targets).sum().item()
            correct5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct1 / total
    epoch_top5 = correct5 / total
    return epoch_loss, epoch_acc, epoch_top5


def _make_ratios(start, stop, step):
    ratios = []
    cur = start
    while cur >= stop - 1e-12:
        ratios.append(round(cur, 1))
        cur -= step
    ratios = [r for r in ratios if r > 0]
    return ratios


def train_teacher_cifar100_allclasses(args):
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(args.seed)

    log_path = args.log_file
    if not log_path:
        log_path = os.path.join("logs", f"teacher_cifar100_allclasses_{args.experiment}.log")
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

    train_dataset, test_dataset, num_classes = get_datasets(args.data_root)
    num_samples = len(train_dataset)
    all_indices = np.arange(num_samples)
    rng.shuffle(all_indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ratios = _make_ratios(args.start_ratio, args.min_ratio, args.ratio_step)
    os.makedirs(args.save_dir, exist_ok=True)
    log(f"Ratios to train: {ratios}")

    history_dir = os.path.join(args.save_dir, "teacher_history")
    os.makedirs(history_dir, exist_ok=True)

    for ratio in ratios:
        count = max(1, int(math.floor(num_samples * ratio)))
        selected_indices = all_indices[:count]
        subset = Subset(train_dataset, selected_indices)
        train_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = get_teacher(num_classes=num_classes).to(device)
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

        best_acc = 0.0
        best_epoch = 0
        train_losses = []
        train_accs = []
        train_top5_accs = []
        test_losses = []
        test_accs = []
        test_top5_accs = []
        save_path = os.path.join(
            args.save_dir,
            f"teacher_cifar100_ratio{ratio:.1f}_best.pth",
        )
        history_path = os.path.join(
            history_dir,
            f"teacher_cifar100_ratio{ratio:.1f}.npz",
        )

        log(
            f"Start training teacher on CIFAR100 all classes ratio={ratio:.1f} "
            f"samples={count}/{num_samples} epochs={args.epochs} seed={args.seed}",
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
            test_losses.append(val_loss)
            test_accs.append(val_acc)
            test_top5_accs.append(val_top5)
            log(
                f"[ratio={ratio:.1f}] epoch={epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_top5={train_top5:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f}",
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_acc": best_acc,
                        "dataset": "cifar100",
                        "num_classes": num_classes,
                        "ratio": float(ratio),
                        "num_train_samples": int(count),
                        "seed": int(args.seed),
                    },
                    save_path,
                )
                log(
                    f"[ratio={ratio:.1f}] new best val_acc={best_acc:.4f} "
                    f"saved to {save_path}",
                )

        np.savez_compressed(
            history_path,
            train_losses=np.array(train_losses, dtype=np.float32),
            train_accs=np.array(train_accs, dtype=np.float32),
            train_top5_accs=np.array(train_top5_accs, dtype=np.float32),
            test_losses=np.array(test_losses, dtype=np.float32),
            test_accs=np.array(test_accs, dtype=np.float32),
            test_top5_accs=np.array(test_top5_accs, dtype=np.float32),
            ratio=np.float32(ratio),
            num_train_samples=np.int64(count),
            seed=np.int64(args.seed),
        )
        log(f"[ratio={ratio:.1f}] history saved to {history_path}")
        log(
            f"Finished ratio={ratio:.1f} best_acc={best_acc:.4f} best_epoch={best_epoch} "
            f"checkpoint={save_path}",
        )

    log("All teacher trainings finished")
    log(f"Logs written to {args.log_file}")
    if log_file is not None:
        log_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="train_teachers",
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
        "--start-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ratio-step",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
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
    train_teacher_cifar100_allclasses(args)
