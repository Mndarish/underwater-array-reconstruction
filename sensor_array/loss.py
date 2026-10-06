"""Objective: weighted Huber TDOA misfit + weak ship-track prior."""
import numpy as np

from .config import C_SOUND, DZ, HUBER_DELTA, LAMBDA_TRACK
from .parameterisation import to_positions, unpack


def huber(r, delta=HUBER_DELTA):
    """Quadratic for |r| <= delta, linear beyond -- limits outlier influence."""
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r ** 2, delta * (abs_r - 0.5 * delta))


def build_loss(tx_xy, tdoa_obs, weights, sensor_xy_init, lambda_track=LAMBDA_TRACK):
    """Return loss(x) closure.

    lambda_track = 1e-7: at 1e-4 the prior gradient was ~100x the acoustic
    gradient and froze the optimisation; at 1e-7 the data dominates and
    sensors move 2-7 m, which is physically realistic.
    """
    def loss(x):
        anchor, log_lengths, angles, log_t0s = unpack(x)
        t0s = np.exp(log_t0s)

        sxy, _ = to_positions(anchor, log_lengths, angles)

        dx = tx_xy[:, 0][None, :] - sxy[:, 0:1]            # (N, 6)
        dy = tx_xy[:, 1][None, :] - sxy[:, 1:2]
        dist = np.sqrt(dx ** 2 + dy ** 2 + DZ ** 2)

        toa_pred = dist / C_SOUND + t0s[None, :]
        tdoa_pred = toa_pred - toa_pred.mean(axis=0, keepdims=True)

        residuals = tdoa_pred - tdoa_obs
        data_loss = (weights[:, None] * huber(residuals)).mean()

        track_loss = ((sxy - sensor_xy_init) ** 2).sum(axis=1).mean()
        return data_loss + lambda_track * track_loss

    return loss
