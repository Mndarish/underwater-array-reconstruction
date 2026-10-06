"""L-BFGS-B driver."""
import numpy as np
from scipy.optimize import minimize

from .config import MAX_SPACING, MAXFUN, MAXITER
from .parameterisation import to_positions, unpack


def optimise(loss_fn, x0, sensor_xy_init, maxiter=MAXITER, maxfun=MAXFUN):
    print(f"  Parameters: {len(x0)} | Initial loss: {loss_fn(x0):.8f}")
    loss_history = []

    def callback(xk):
        loss_history.append(loss_fn(xk))
        if len(loss_history) % 100 == 0:
            anchor, ll, ang, _ = unpack(xk)
            sxy, lens = to_positions(anchor, ll, ang)
            dev = np.sqrt(((sxy - sensor_xy_init) ** 2).sum(axis=1)).mean()
            print(f"    iter {len(loss_history):5d} | loss={loss_history[-1]:.6e} | "
                  f"mean_dev={dev:.3f} m | max_spacing={lens.max():.5f} m")

    result = minimize(
        loss_fn, x0, method="L-BFGS-B", callback=callback,
        options={"maxiter": maxiter, "maxfun": maxfun, "ftol": 1e-15, "gtol": 1e-12},
    )

    print(f"\n  {'Converged' if result.success else 'Stopped'}: {result.message}")
    print(f"  Final loss: {result.fun:.8f}")

    anchor_f, ll_f, ang_f, _ = unpack(result.x)
    sensor_xy_final, spacings = to_positions(anchor_f, ll_f, ang_f)
    devs = np.sqrt(((sensor_xy_final - sensor_xy_init) ** 2).sum(axis=1))
    print(f"  Spacing   - mean: {spacings.mean():.5f} m, max: {spacings.max():.5f} m, "
          f"violations (>{MAX_SPACING}): {(spacings > MAX_SPACING).sum()}")
    print(f"  Deviation - mean: {devs.mean():.3f} m, max: {devs.max():.3f} m")
    return sensor_xy_final, loss_history, result
