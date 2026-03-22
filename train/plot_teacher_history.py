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


def plot_single_history(npz_path, output_dir):
    set_scientific_style()
    data = np.load(npz_path)
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(npz_path))[0]
    metric_keys = []
    for key in data.files:
        values = data[key]
        if not isinstance(values, np.ndarray):
            continue
        if values.ndim != 1:
            continue
        key_lower = key.lower()
        if "loss" in key_lower or "acc" in key_lower:
            metric_keys.append(key)
    for key in metric_keys:
        values = data[key]
        epochs = np.arange(1, len(values) + 1)
        fig, ax = plt.subplots()
        ax.plot(epochs, values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(f"{basename} - {key}")
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{basename}_{key}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def plot_history(target_path, output_dir):
    if os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for name in files:
                if not name.lower().endswith(".npz"):
                    continue
                npz_path = os.path.join(root, name)
                plot_single_history(npz_path, output_dir)
    else:
        plot_single_history(target_path, output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_history(args.path, args.output_dir)

