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


def load_val_accs(npz_path):
    data = np.load(npz_path)
    # 在train_teacher_cifar100_allclasses.py里保存的是test_accs，对应val_acc
    if "test_accs" in data.files:
        return np.array(data["test_accs"], dtype=np.float32)
    if "val_accs" in data.files:
        return np.array(data["val_accs"], dtype=np.float32)
    return None


def plot_teacher_cifar100_allclasses(history_dir, output_dir):
    set_scientific_style()
    os.makedirs(output_dir, exist_ok=True)

    ratios = [i / 10.0 for i in range(10, 0, -1)]
    ratio_to_accs = {}
    for ratio in ratios:
        path = os.path.join(
            history_dir,
            f"teacher_cifar100_ratio{ratio:.1f}.npz",
        )
        if not os.path.isfile(path):
            continue
        accs = load_val_accs(path)
        if accs is None:
            continue
        ratio_to_accs[ratio] = accs

    if not ratio_to_accs:
        return

    fig, ax = plt.subplots()
    for ratio, accs in sorted(ratio_to_accs.items(), reverse=True):
        epochs = np.arange(1, len(accs) + 1)
        ax.plot(epochs, accs, label=f"ratio={ratio:.1f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("CIFAR100 Teacher (All Classes) - Top-1 for Different Ratios")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(output_dir, "teacher_cifar100_allclasses_Top-1_accs.png")
    fig.savefig(out_path, dpi=300)  
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-dir",
        type=str,
        default="./checkpoints/teacher_history",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots/teacher_cifar100_allclasses",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_teacher_cifar100_allclasses(args.history_dir, args.output_dir)

