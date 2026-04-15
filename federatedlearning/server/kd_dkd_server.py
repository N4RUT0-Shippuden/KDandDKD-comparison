import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from federatedlearning.client.kd_dkd_client import KDDKDClient
from federatedlearning.model.student import get_student
try:
    import wandb
except ImportError:
    wandb = None


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, label_map):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_map = dict(label_map)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx):
        x, y = self.dataset[int(self.indices[int(idx)])]
        return x, int(self.label_map[int(y)])


def _parse_classes(classes_str):
    parts = [p.strip() for p in classes_str.split(",") if p.strip() != ""]
    class_ids = [int(p) for p in parts]
    unique = sorted(set(class_ids))
    if len(class_ids) != len(unique):
        raise ValueError(f"classes contains duplicates: {classes_str}")
    if len(class_ids) != 10:
        raise ValueError(f"classes must contain exactly 10 ids, got {len(class_ids)}")
    for c in class_ids:
        if c < 0 or c > 99:
            raise ValueError(f"class id out of range [0, 99]: {c}")
    return class_ids


def _classes_tag(class_ids):
    return "classes" + "-".join(str(c) for c in class_ids)


def _filter_and_remap(dataset, class_ids):
    class_set = set(int(c) for c in class_ids)
    indices = [i for i, y in enumerate(dataset.targets) if int(y) in class_set]
    label_map = {int(c): idx for idx, c in enumerate(class_ids)}
    return np.asarray(indices, dtype=np.int64), label_map


def get_cifar100_allclasses_datasets(data_root):
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
    return train_dataset, test_dataset, num_classes, None, "allclasses"


def get_cifar100_tenclasses_datasets(data_root, classes_str):
    class_ids = _parse_classes(classes_str)
    tag = _classes_tag(class_ids)
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
    base_train = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    base_test = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )
    train_indices, label_map = _filter_and_remap(base_train, class_ids)
    test_indices, _ = _filter_and_remap(base_test, class_ids)
    train_subset = RemappedSubset(base_train, train_indices, label_map)
    test_subset = RemappedSubset(base_test, test_indices, label_map)
    num_classes = 10
    return train_subset, test_subset, num_classes, class_ids, tag


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            _, top5_pred = outputs.topk(5, 1, True, True)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            correct_top5 += top5_pred.eq(
                targets.view(-1, 1).expand_as(top5_pred),
            ).sum().item()
    if total == 0:
        return 0.0, 0.0
    top1 = correct / total
    top5 = correct_top5 / total
    return top1, top5


def aggregate_state_dicts(state_dicts, weights=None):
    if not state_dicts:
        return {}
    num_models = len(state_dicts)
    weights = [1.0 / float(num_models)] * num_models
    base_state = state_dicts[0]
    global_state = {}
    for name, tensor in base_state.items():
        t = tensor.clone()
        if t.is_floating_point():
            t.zero_()
        global_state[name] = t
    for idx, (w, client_state) in enumerate(zip(weights, state_dicts)):
        for name, server_tensor in global_state.items():
            client_tensor = client_state[name]
            if server_tensor.is_floating_point():
                server_tensor.add_(client_tensor.to(server_tensor.dtype), alpha=w)
            elif name.endswith("num_batches_tracked"):
                if idx == 0:
                    server_tensor.copy_(client_tensor)
                else:
                    server_tensor.copy_(torch.maximum(server_tensor, client_tensor))
    return global_state


def run_federated_kd_dkd(args):
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_wandb = bool(getattr(args, "use_wandb", False)) and wandb is not None
    if use_wandb:
        project = getattr(args, "wandb_project", "niid-kd-dkd")
        entity = getattr(args, "wandb_entity", "") or None
        run_name = getattr(args, "wandb_run_name", "") or (
            f"fed_{args.method}_{args.mode}_seed{args.seed}"
        )
        wandb.init(project=project, entity=entity, name=run_name, config=vars(args))
    if args.mode == "all":
        train_dataset, test_dataset, num_classes, class_ids, tag = (
            get_cifar100_allclasses_datasets(args.data_root)
        )
    else:
        train_dataset, test_dataset, num_classes, class_ids, tag = (
            get_cifar100_tenclasses_datasets(args.data_root, args.classes)
        )
    rng = np.random.RandomState(args.seed)
    num_samples = len(train_dataset)
    all_indices = np.arange(num_samples)
    rng.shuffle(all_indices)
    ratios = [i / 10.0 for i in range(10, 0, -1)]
    if args.num_clients != len(ratios):
        raise ValueError(
            f"num_clients must be {len(ratios)} to match ratios, got {args.num_clients}",
        )
    client_subsets = []
    for ratio in ratios:
        count = max(1, int(math.floor(num_samples * ratio)))
        selected_indices = all_indices[:count]
        subset = Subset(train_dataset, selected_indices)
        client_subsets.append(subset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    clients = []
    for client_id in range(args.num_clients):
        client_dataset = client_subsets[client_id]
        client = KDDKDClient(
            client_id=client_id,
            train_dataset=client_dataset,
            num_classes=num_classes,
            method=args.method,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            tckd_weight=args.tckd_weight,
            nckd_weight=args.nckd_weight,
            kd_weight=args.kd_weight,
            dkd_weight=args.dkd_weight,
            global_rounds=args.global_rounds,
            warmup_rounds=args.warmup_rounds,
            device=device,
            num_workers=args.num_workers,
        )
        clients.append(client)
    global_student = get_student(num_classes=num_classes).to(device)
    global_state = global_student.state_dict()
    for round_idx in range(1, args.global_rounds + 1):
        print(
            f"===== Global Round {round_idx}/{args.global_rounds} "
            f"mode={args.mode} tag={tag} =====",
        )
        for client in clients:
            client.set_global_student_params(global_state)
        student_states = []
        for client in clients:
            for _ in range(args.local_epochs):
                student_state, _, train_stats = client.local_train_one_epoch(
                    global_round=round_idx,
                )
            teacher_top1, teacher_top5 = evaluate(
                client.teacher,
                test_loader,
                device,
            )
            student_top1, student_top5 = evaluate(
                client.student,
                test_loader,
                device,
            )
            round_i = round_idx - 1
            client_i = client.client_id
            print(
                "[TRAIN] Round:{:3d} Client:{:2d} "
                "StudentLoss:{:.4f} TeacherLoss:{:.4f} "
                "StudentTop1:{:.2f}% StudentTop5:{:.2f}% "
                "TeacherTop1:{:.2f}% TeacherTop5:{:.2f}%".format(
                    round_idx,
                    client.client_id,
                    train_stats["student_loss"],
                    train_stats["teacher_loss"],
                    train_stats["student_top1"] * 100.0,
                    train_stats["student_top5"] * 100.0,
                    train_stats["teacher_top1"] * 100.0,
                    train_stats["teacher_top5"] * 100.0,
                ),
            )
            print(
                "[EVAL ] Round:{:3d} Client:{:2d} "
                "TeacherTop1:{:.2f}% TeacherTop5:{:.2f}% "
                "StudentTop1:{:.2f}% StudentTop5:{:.2f}%".format(
                    round_idx,
                    client.client_id,
                    teacher_top1 * 100.0,
                    teacher_top5 * 100.0,
                    student_top1 * 100.0,
                    student_top5 * 100.0,
                ),
            )
            if use_wandb:
                metrics = {
                    f"client_{client_i}/train/student_loss": train_stats[
                        "student_loss"
                    ],
                    f"client_{client_i}/train/teacher_loss": train_stats[
                        "teacher_loss"
                    ],
                    f"client_{client_i}/train/student_top1": train_stats[
                        "student_top1"
                    ],
                    f"client_{client_i}/train/student_top5": train_stats[
                        "student_top5"
                    ],
                    f"client_{client_i}/train/teacher_top1": train_stats[
                        "teacher_top1"
                    ],
                    f"client_{client_i}/train/teacher_top5": train_stats[
                        "teacher_top5"
                    ],
                    f"client_{client_i}/eval/teacher_top1": teacher_top1,
                    f"client_{client_i}/eval/teacher_top5": teacher_top5,
                    f"client_{client_i}/eval/student_top1": student_top1,
                    f"client_{client_i}/eval/student_top5": student_top5,
                }
                wandb.log(metrics, step=round_idx)
            student_states.append(student_state)
        global_state = aggregate_state_dicts(student_states)
