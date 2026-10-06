<h1 align="center">Undersea Sensor Array Estimation</h1>
<p align="center">
Recovering the shape of a 1,926-sensor seabed cable from acoustic arrival times,<br>
using a differentiable physics model and gradient-based optimisation.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/optimiser-L--BFGS--B-orange" alt="optimiser">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
</p>

<p align="center"><img src="docs/figures/array_estimation_result.png" width="100%"></p>

## Problem

A cable of 1,926 hydrophones was laid on the seabed along a ship's GPS track, but currents
shifted it. Six transmitters at known positions each fired twice; every sensor recorded the
arrival times. The firing times are unknown, so only *time differences* of arrival (TDOA) are usable.
**Goal:** estimate the (x, y) position of every sensor, subject to a maximum cable spacing of 1.0213 m.

## Method

| Step | Idea |
|------|------|
| **Propagation model** | `d = sqrt((x-X)² + (y-Y)² + 18²)`, `τ = d / 1540 m/s` |
| **TDOA** | Average the two firings per transmitter (noise ÷ √2), subtract a per-transmitter reference to cancel the unknown firing time |
| **Initialisation** | Interpolate sensors evenly along the ship track, capped at the physical cable length |
| **Cable parameterisation** | Optimise an anchor + segment *lengths* + *angles* instead of raw (x, y). `length = 1.0213 · σ(·)` enforces the spacing limit **by construction** and stays differentiable |
| **Robust loss** | Huber loss on TDOA residuals, so bad sensors don't dominate |
| **Sensor weights** | The two firings should differ by a constant across sensors; deviation = noise. Noisy sensors get lower (floored) weight |
| **Track prior** | Very weak (λ = 1e-7) pull toward the ship track. At 1e-4 it dominated the acoustic gradient ~100× and froze the optimisation |
| **Optimiser** | L-BFGS-B over ~3,850 parameters |

Full write-up: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Results

| Metric | Value |
|--------|-------|
| Constraint violations (spacing > 1.0213 m) | **0** |
| Mean / max inter-sensor spacing | 1.0196 m / 1.0212 m |
| Mean deviation from ship track | ≈ 3 m |
| Short spacings (cable bends) | 35 of 1,925 |

<p align="center"><img src="docs/figures/spacing_histogram.png" width="100%"></p>

Almost the whole cable sits at the spacing limit (taut on the seabed); the short spacings
occur at bends. The estimate stays within a few metres of the track, consistent with ocean-current drift.
The final coordinates are in [`results/results.csv`](results/results.csv) (`sensor, x, y`).

## Project structure

```
.
├── main.py                    # CLI entry point
├── sensor_array/
│   ├── config.py              # constants & defaults
│   ├── data.py                # CSV loading, TOA averaging, TDOA
│   ├── initialisation.py      # track-based initial layout
│   ├── weights.py             # per-sensor reliability weights
│   ├── parameterisation.py    # sigmoid length + angle cable model
│   ├── loss.py                # Huber TDOA loss + track prior
│   ├── optimise.py            # L-BFGS-B driver
│   ├── plotting.py            # figures
│   └── io.py                  # results.csv writer
├── scripts/plot_from_results.py   # figures from results.csv alone
├── data/                      # put input CSVs here (git-ignored)
├── results/results.csv
└── docs/                      # methodology write-up + figures
```

## Usage

```bash
pip install -r requirements.txt

# put track.csv, transmissions.csv, timings.csv in data/
python main.py --data-dir data --out-dir results --fig-dir docs/figures
```

Runs in a few minutes. Options: `--lambda-track`, `--maxiter`, `--maxfun`, `--show`.

## License

MIT
