"""Loading and pre-processing of the three input CSV files."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_TX


def load_data(data_dir):
    """Load track, transmitter locations and arrival timings.

    Returns
    -------
    track_xy : (T, 2)   ship GPS track
    tx_xy    : (6, 2)   transmitter positions
    toa_obs  : (N, 12)  raw times of arrival (2 firings x 6 transmitters)
    tdoa_obs : (N, 6)   averaged, median-referenced TDOA
    """
    data_dir = Path(data_dir)
    track = pd.read_csv(data_dir / "track.csv")
    tx = pd.read_csv(data_dir / "transmissions.csv")
    timings = pd.read_csv(data_dir / "timings.csv")

    track_xy = track[["x", "y"]].values.astype(np.float64)
    tx_xy = tx[["x", "y"]].values.astype(np.float64)

    toa_cols = [f"t{k}{p}" for k in range(1, N_TX + 1) for p in (1, 2)]
    toa_obs = timings[toa_cols].values.astype(np.float64)

    # Each transmitter fires twice from the same location: average the pair
    # to cut noise by sqrt(2) -> 6 cleaner observations per sensor.
    toa_avg = np.stack(
        [0.5 * (toa_obs[:, 2 * k] + toa_obs[:, 2 * k + 1]) for k in range(N_TX)],
        axis=1,
    )

    # Subtract a per-column reference to form TDOA; this cancels the unknown
    # firing-time offset t0 without having to optimise it explicitly.
    toa_ref = np.median(toa_avg, axis=0, keepdims=True)
    tdoa_obs = toa_avg - toa_ref

    print(f"  {len(track_xy)} track pts | {len(tx_xy)} transmitters | "
          f"raw TOA {toa_obs.shape} -> averaged TDOA {tdoa_obs.shape}")
    return track_xy, tx_xy, toa_obs, tdoa_obs
