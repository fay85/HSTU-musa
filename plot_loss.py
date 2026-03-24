"""Plot HSTU training loss curves. Supports single or dual (MUSA vs MUSA) comparison."""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def read_loss_log(path):
    steps, losses = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return np.array(steps), np.array(losses)


def moving_average(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser(description="Plot HSTU loss curves")
    parser.add_argument("--musa", type=str, default="loss_musa.csv",
                        help="MUSA loss log CSV")
    parser.add_argument("--musa", type=str, default=None,
                        help="MUSA loss log CSV (optional, for comparison)")
    parser.add_argument("--smooth", type=int, default=10,
                        help="Moving average window size")
    parser.add_argument("--output", type=str, default="loss_curve.png",
                        help="Output image path")
    args = parser.parse_args()

    has_musa = args.musa is not None

    fig, axes = plt.subplots(
        1, 2 if has_musa else 1,
        figsize=(14 if has_musa else 8, 5),
        squeeze=False,
    )

    # --- MUSA loss ---
    musa_steps, musa_losses = read_loss_log(args.musa)
    musa_smooth = moving_average(musa_losses, args.smooth)
    smooth_steps = musa_steps[args.smooth - 1:]

    ax = axes[0, 0]
    ax.plot(musa_steps, musa_losses, alpha=0.25, color="#2196F3", linewidth=0.5)
    ax.plot(smooth_steps, musa_smooth, color="#2196F3", linewidth=1.5, label="MUSA")

    if has_musa:
        musa_steps, musa_losses = read_loss_log(args.musa)
        musa_smooth = moving_average(musa_losses, args.smooth)
        musa_smooth_steps = musa_steps[args.smooth - 1:]
        ax.plot(musa_steps, musa_losses, alpha=0.25, color="#FF5722", linewidth=0.5)
        ax.plot(musa_smooth_steps, musa_smooth, color="#FF5722", linewidth=1.5, label="MUSA")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("HSTU Training Loss (ML-20M)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    stats_text = (
        f"MUSA: start={musa_losses[0]:.4f}  "
        f"end={musa_losses[-1]:.4f}  "
        f"avg={musa_losses.mean():.4f}"
    )
    if has_musa:
        stats_text += (
            f"\nMUSA: start={musa_losses[0]:.4f}  "
            f"end={musa_losses[-1]:.4f}  "
            f"avg={musa_losses.mean():.4f}"
        )
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # --- Difference plot (only if both logs provided) ---
    if has_musa:
        ax2 = axes[0, 1]
        min_len = min(len(musa_losses), len(musa_losses))
        diff = musa_losses[:min_len] - musa_losses[:min_len]
        diff_smooth = moving_average(diff, args.smooth)
        diff_smooth_steps = musa_steps[args.smooth - 1 : args.smooth - 1 + len(diff_smooth)]

        ax2.plot(musa_steps[:min_len], diff, alpha=0.25, color="#9C27B0", linewidth=0.5)
        ax2.plot(diff_smooth_steps, diff_smooth, color="#9C27B0", linewidth=1.5)
        ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss Difference (MUSA - MUSA)")
        ax2.set_title("MUSA vs MUSA Divergence")
        ax2.grid(True, alpha=0.3)

        abs_diff = np.abs(diff)
        ax2.text(
            0.98, 0.98,
            f"Mean |diff|: {abs_diff.mean():.6f}\n"
            f"Max |diff|: {abs_diff.max():.6f}\n"
            f"Steps compared: {min_len}",
            transform=ax2.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")

    try:
        plt.show(block=False)
        plt.pause(0.1)
    except Exception:
        pass
    plt.close()


if __name__ == "__main__":
    main()
