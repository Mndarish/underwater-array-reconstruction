"""Per-sensor reliability weights from repeat-transmission consistency."""
import numpy as np

from .config import C_SOUND, N_SENSORS, N_TX


def compute_sensor_weights(toa_obs):
    # For transmitter k, t_k1[i] - t_k2[i] should be the same for every sensor
    # (identical geometry, only t0 differs). Deviation from the median is
    # therefore a direct measure of timing noise at sensor i.
    noise_score = np.zeros(N_SENSORS)
    for k in range(N_TX):
        diff = toa_obs[:, 2 * k] - toa_obs[:, 2 * k + 1]
        noise_score += np.abs(diff - np.median(diff))
    noise_score /= N_TX

    # Gaussian decay; floored at 0.05 so noisy sensors are down-weighted, not dropped.
    sigma = np.percentile(noise_score, 50)
    weights = np.clip(np.exp(-0.5 * (noise_score / sigma) ** 2), 0.05, 1.0)

    print(f"  Noise: mean={noise_score.mean():.5f} s "
          f"({noise_score.mean() * C_SOUND:.1f} m equivalent) | "
          f"reliable (w>0.5): {(weights > 0.5).sum()}/{N_SENSORS}")
    return weights
