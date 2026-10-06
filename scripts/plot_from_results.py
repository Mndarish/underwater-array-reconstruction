"""Regenerate figures from results/results.csv alone (no raw data needed)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensor_array.config import MAX_SPACING  # noqa: E402
from sensor_array.plotting import plot_spacings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "results.csv")
xy = df[["x", "y"]].values
fig_dir = ROOT / "docs" / "figures"

plot_spacings(xy, fig_dir / "spacing_histogram_from_csv.png")

fig, ax = plt.subplots(figsize=(6, 9))
sc = ax.scatter(xy[:, 0], xy[:, 1], c=df["sensor"], s=3, cmap="viridis")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Estimated array geometry (1926 sensors)")
fig.colorbar(sc, ax=ax, label="Sensor index", shrink=0.6)
fig.tight_layout()
fig.savefig(fig_dir / "array_geometry.png", dpi=150, bbox_inches="tight")
print("Saved array_geometry.png")
