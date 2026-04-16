import os
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_scientific_style():
    mpl.rcParams.update(
        {
            "figure.figsize": (6, 4),
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


def load_test_accs(npz_path):
    data = np.load(npz_path)
    if "test_accs" not in data.files:
        return None
    return np.array(data["test_accs"], dtype=np.float32)


def plot_student_cifar10(history_dir, output_dir):
    set_scientific_style()
    os.makedirs(output_dir, exist_ok=True)
    ratios = [i / 10.0 for i in range(10, 0, -1)]
    for ratio in ratios:
        kd_path = os.path.join(
            history_dir,
            f"cifar10_ratio{ratio:.1f}_KD.npz",
        )
        dkd_path = os.path.join(
            history_dir,
            f"cifar10_ratio{ratio:.1f}_DKD.npz",
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
        ax.set_ylabel("Test Top-1 Accuracy")
        ax.set_title(f"CIFAR10 Student - Ratio {ratio:.1f}")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(
            output_dir,
            f"cifar10_ratio{ratio:.1f}_student_top1.png",
        )
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-dir",
        type=str,
        default="./checkpoints/distill_history",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots/student_cifar10",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_student_cifar10(args.history_dir, args.output_dir)

