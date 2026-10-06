"""Writing the results file (sensor, x, y)."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_SENSORS


def save_results(sensor_xy_final, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "sensor": np.arange(1, N_SENSORS + 1),
        "x": sensor_xy_final[:, 0],
        "y": sensor_xy_final[:, 1],
    })
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return df
