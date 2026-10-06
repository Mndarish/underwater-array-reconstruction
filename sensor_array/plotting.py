"""Result figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import MAX_SPACING, N_SENSORS


def _finish(fig, save_path, show):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_results(sensor_xy_final, sensor_xy_init, track_xy, tx_xy,
                 loss_history, save_path, show=False):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle("Undersea Sensor Array Estimation", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(track_xy[:, 0], track_xy[:, 1], "k--", lw=2, label="Ship track", alpha=0.7, zorder=3)
    ax.plot(sensor_xy_init[:, 0], sensor_xy_init[:, 1], "b-", lw=1, alpha=0.3, label="Track init")
    ax.plot(sensor_xy_final[:, 0], sensor_xy_final[:, 1], "r-", lw=1.8, label="Estimated array", alpha=0.9)
    ax.scatter(tx_xy[:, 0], tx_xy[:, 1], marker="*", s=300, c="limegreen",
               edgecolors="darkgreen", lw=0.8, zorder=5, label="Transmitters")
    ax.scatter(*sensor_xy_final[0], c="darkred", s=100, zorder=6, marker="o", label="Sensor 1")
    ax.scatter(*sensor_xy_final[-1], c="maroon", s=100, zorder=6, marker="s", label=f"Sensor {N_SENSORS}")
    for i, (tx, ty) in enumerate(tx_xy):
        ax.annotate(f"Tx{i + 1}", (tx, ty), fontsize=7, xytext=(5, 5),
                    textcoords="offset points", color="darkgreen")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("Array Shape vs Ship Track")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    dev = np.sqrt(((sensor_xy_final - sensor_xy_init) ** 2).sum(axis=1))
    idx = np.arange(1, N_SENSORS + 1)
    ax2.plot(idx, dev, color="steelblue", lw=0.8)
    ax2.axhline(dev.mean(), color="red", linestyle="--", lw=1.5, label=f"Mean: {dev.mean():.2f} m")
    ax2.fill_between(idx, 0, dev, alpha=0.15, color="steelblue")
    ax2.set_xlabel("Sensor index"); ax2.set_ylabel("Displacement from track init (m)")
    ax2.set_title("Per-sensor Deviation from Track Prior")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.semilogy(loss_history, color="steelblue", lw=1.5)
    ax3.set_xlabel("Iteration"); ax3.set_ylabel("Loss (log scale)")
    ax3.set_title("Optimisation Convergence"); ax3.grid(True, alpha=0.3)

    _finish(fig, save_path, show)


def plot_spacings(sensor_xy_final, save_path, show=False):
    spacings = np.sqrt((np.diff(sensor_xy_final, axis=0) ** 2).sum(axis=1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Cable Spacing Distribution", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.hist(spacings, bins=100, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(MAX_SPACING, color="red", linestyle="--", lw=2, label=f"Max: {MAX_SPACING} m")
    ax.set_xlabel("Spacing (m)"); ax.set_ylabel("Count")
    ax.set_title(f"Full range  [min={spacings.min():.4f}, max={spacings.max():.5f}]")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    bulk = spacings[spacings > 1.010]
    ax2.hist(bulk, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax2.axvline(MAX_SPACING, color="red", linestyle="--", lw=2, label=f"Max: {MAX_SPACING} m")
    ax2.set_xlabel("Spacing (m)"); ax2.set_ylabel("Count")
    ax2.set_title(f"Zoomed: > 1.010 m  (n={len(bulk)})")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    tail = spacings[spacings < 1.010]
    if len(tail) > 0:
        ax3.hist(tail, bins=30, color="coral", edgecolor="white", alpha=0.85)
        ax3.set_title(f"Short spacings < 1.010 m  (n={len(tail)}, cable bends)")
    else:
        ax3.text(0.5, 0.5, "No short spacings", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Short spacings < 1.010 m")
    ax3.set_xlabel("Spacing (m)"); ax3.set_ylabel("Count"); ax3.grid(True, alpha=0.3)

    _finish(fig, save_path, show)
    print(f"  Spacing - mean: {spacings.mean():.5f} m, max: {spacings.max():.5f} m, "
          f"violations: {(spacings > MAX_SPACING).sum()}")
