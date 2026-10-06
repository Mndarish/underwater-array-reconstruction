"""Cable parameterisation: (anchor, segment lengths, segment angles).

Optimising lengths + angles instead of raw (x, y) preserves cable
connectivity, and a sigmoid on the lengths enforces the max-spacing
constraint *by construction* while staying differentiable.
"""
import numpy as np

from .config import C_SOUND, DZ, MAX_SPACING, N_SENSORS


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))   # clip avoids overflow


def to_positions(anchor, log_lengths, angles):
    """Map unconstrained parameters to sensor positions (N, 2) and lengths (N-1,)."""
    lengths = MAX_SPACING * sigmoid(log_lengths)          # always in (0, MAX_SPACING)
    steps = np.stack([lengths * np.cos(angles),
                      lengths * np.sin(angles)], axis=1)
    positions = np.empty((N_SENSORS, 2))
    positions[0] = anchor
    positions[1:] = anchor + np.cumsum(steps, axis=0)
    return positions, lengths


def unpack(x):
    """Split the flat optimiser vector into named parameter groups."""
    n = N_SENSORS - 1
    anchor = x[:2]
    log_lengths = x[2:2 + n]
    angles = x[2 + n:2 + 2 * n]
    log_t0s = x[2 + 2 * n:]
    return anchor, log_lengths, angles, log_t0s


def init_params(sensor_xy_init, tx_xy):
    steps = np.diff(sensor_xy_init, axis=0)
    norms = np.clip(np.sqrt((steps ** 2).sum(axis=1)), 1e-6, MAX_SPACING * 0.9999)

    v = norms / MAX_SPACING
    log_lengths = np.log(v / (1.0 - v + 1e-10))          # logit
    angles = np.arctan2(steps[:, 1], steps[:, 0])

    dist0 = np.sqrt((tx_xy[:, 0] - sensor_xy_init[0, 0]) ** 2 +
                    (tx_xy[:, 1] - sensor_xy_init[0, 1]) ** 2 + DZ ** 2)
    log_t0s = np.log(dist0 / C_SOUND + 1e-6)             # log-space keeps t0 > 0

    return np.concatenate([sensor_xy_init[0], log_lengths, angles, log_t0s])
