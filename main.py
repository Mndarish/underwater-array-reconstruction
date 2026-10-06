"""Run the full array-estimation pipeline.

    python main.py --data-dir data --out-dir results
"""
import argparse
from pathlib import Path

import matplotlib
from sensor_array import (build_loss, compute_sensor_weights, init_from_track,
                          init_params, load_data, optimise)
from sensor_array.config import LAMBDA_TRACK, MAXFUN, MAXITER
from sensor_array.io import save_results
from sensor_array.plotting import plot_results, plot_spacings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data",
                   help="folder with track.csv, transmissions.csv, timings.csv")
    p.add_argument("--out-dir", default="results", help="where results.csv is written")
    p.add_argument("--fig-dir", default="docs/figures", help="where figures are written")
    p.add_argument("--lambda-track", type=float, default=LAMBDA_TRACK)
    p.add_argument("--maxiter", type=int, default=MAXITER)
    p.add_argument("--maxfun", type=int, default=MAXFUN)
    p.add_argument("--show", action="store_true", help="display plots interactively")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    sep = "-" * 62

    print(sep); print("1. Loading data")
    track_xy, tx_xy, toa_obs, tdoa_obs = load_data(args.data_dir)

    print(sep); print("2. Initialising from ship track")
    sensor_xy_init = init_from_track(track_xy)

    print(sep); print("3. Computing sensor reliability weights")
    weights = compute_sensor_weights(toa_obs)

    print(sep); print("4. Building loss function")
    loss_fn = build_loss(tx_xy, tdoa_obs, weights, sensor_xy_init, args.lambda_track)

    print(sep); print("5. Initialising parameters")
    x0 = init_params(sensor_xy_init, tx_xy)

    print(sep); print("6. Optimising (L-BFGS-B)")
    sensor_xy_final, loss_history, _ = optimise(
        loss_fn, x0, sensor_xy_init, maxiter=args.maxiter, maxfun=args.maxfun)

    print(sep); print("7. Plotting")
    plot_results(sensor_xy_final, sensor_xy_init, track_xy, tx_xy, loss_history,
                 fig_dir / "array_estimation_result.png", show=args.show)
    plot_spacings(sensor_xy_final, fig_dir / "spacing_histogram.png", show=args.show)

    print(sep); print("8. Saving")
    save_results(sensor_xy_final, out_dir / "results.csv")
    print(sep); print("Done.")


if __name__ == "__main__":
    main()
