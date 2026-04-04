import os
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_scientific_style():
    mpl.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.8,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


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


def load_test_accs(npz_path):
    data = np.load(npz_path)
    if "test_accs" not in data.files:
        return None
    return np.array(data["test_accs"], dtype=np.float32)


def plot_student_cifar100_tenclasses(history_root, output_dir, classes_str):
    set_scientific_style()
    os.makedirs(output_dir, exist_ok=True)

    class_ids = _parse_classes(classes_str)
    tag = _classes_tag(class_ids)
    history_dir = os.path.join(
        history_root,
        f"distill_cifar100_tenclasses_{tag}_history",
    )

    ratios = [i / 10.0 for i in range(10, 0, -1)]
    for ratio in ratios:
        kd_path = os.path.join(
            history_dir,
            f"cifar100_tenclasses_{tag}_teacher_ratio{ratio:.1f}_KD.npz",
        )
        dkd_path = os.path.join(
            history_dir,
            f"cifar100_tenclasses_{tag}_teacher_ratio{ratio:.1f}_DKD.npz",
        )
        if not os.path.isfile(kd_path) or not os.path.isfile(dkd_path):
            continue
        kd_accs = load_test_accs(kd_path)
        dkd_accs = load_test_accs(dkd_path)
        if kd_accs is None or dkd_accs is None:
            continue
        epochs_kd = np.arange(1, len(kd_accs) + 1)
        epochs_dkd = np.arange(1, len(dkd_accs) + 1)
        fig, ax = plt.subplots()
        ax.plot(epochs_kd, kd_accs, label="KD")
        ax.plot(epochs_dkd, dkd_accs, label="DKD")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Top-1 Accuracy")
        ax.set_title(
            f"CIFAR100 Student (Ten Classes: {tag}) - Ratio {ratio:.1f}",
        )
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(
            output_dir,
            f"cifar100_tenclasses_{tag}_ratio{ratio:.1f}_student_top1.png",
        )
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-root",
        type=str,
        default="./checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots/student_cifar100_tenclasses",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_student_cifar100_tenclasses(
        args.history_root,
        args.output_dir,
        args.classes,
    )

