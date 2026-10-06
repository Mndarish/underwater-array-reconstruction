"""Initial sensor layout, interpolated along the ship's GPS track."""
import numpy as np

from .config import MAX_SPACING, N_SENSORS


def init_from_track(track_xy, n_sensors=N_SENSORS):
    diffs = np.diff(track_xy, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    track_len = arc[-1]
    cable_max = (n_sensors - 1) * MAX_SPACING

    # Cap at the physical cable length: using the full track length would give
    # initial steps already above MAX_SPACING before optimisation begins.
    init_arc = min(track_len, cable_max)
    sensor_arcs = np.linspace(0.0, init_arc, n_sensors)

    x_init = np.interp(sensor_arcs, arc, track_xy[:, 0])
    y_init = np.interp(sensor_arcs, arc, track_xy[:, 1])
    sensor_xy_init = np.stack([x_init, y_init], axis=1)

    norms = np.sqrt(np.diff(x_init) ** 2 + np.diff(y_init) ** 2)
    print(f"  Track: {track_len:.1f} m | Cable max: {cable_max:.1f} m | "
          f"Spacing mean: {norms.mean():.5f} m, max: {norms.max():.5f} m")
    return sensor_xy_init
