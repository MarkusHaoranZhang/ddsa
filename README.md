# Diagnosis-Driven Structural Adaptation

Companion code for *Diagnosis-Driven Structural Adaptation: A
Closed-Loop Architecture for Resilient Distributed Optimization*
(Lining Xing et al., 2025).

> **Read first**: [`STATUS.md`](STATUS.md) lists every paper section
> and what code path implements it. [`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md)
> lists where the code's current numbers differ from the paper's
> headline numbers and why.
>
> **Note on the high-fidelity track**: Section 5.1.1 of the paper uses
> NASA 42 (open source under NOSA). This repository ships a
> self-contained Python stand-in (`hf_simulator.py`) covering the same
> physics (J2 + bearing friction + SRP + gravity gradient + residual
> drag near perigee). The engine accepts a `simulator_factory`, so a
> wrapper around the real binary plugs in without touching the rest of
> the pipeline.

## Quick start

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev,learning,plot]
pytest                                       :: 20 / 20 tests
ruff check .                                 :: clean
python scripts/reproduce.py --quick --seed 0 :: < 1 minute, smoke run
```

The smoke run produces a complete `results/seed0/*.json` set and all
13 paper figures in `figures/`. For the full run drop `--quick` (a
few minutes on a modern laptop):

```cmd
python scripts/reproduce.py --seed 0
```

## Reference numbers (seed = 0)

A clean run of `python scripts/reproduce.py --seed 0` on the locked
dependency set should land within statistical noise of the table
below. If your numbers do not, treat that as a reproduction failure
and check `meta.json` against `requirements-lock.txt` first.

### §5.4.1 Comparative (single-fault actuator degradation, n_runs = 30)

| Method      | Utilisation | Health MAE | Kendall τ | Detection delay |
|-------------|-------------|------------|-----------|-----------------|
| Oracle      | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | ~50            |
| Proposed    | 0.66 ± 0.04 | 0.12 ± 0.01 | 0.59 ± 0.05 | 0              |
| FDI-Reconf  | 0.66 ± 0.04 | 0.04 ± 0.01 | 1.00 ± 0.00 | ~240           |
| D-S Fusion  | 0.65 ± 0.04 | 0.04 ± 0.01 | 1.00 ± 0.00 | ~270           |
| Robust DO   | 0.52 ± 0.03 | 0.31 ± 0.00 | 0.00 ± 0.00 | 0              |
| Byzantine   | 0.16 ± 0.04 | 0.09 ± 0.02 | 0.00 ± 0.00 | —              |

### §5.3 Ablation (n_runs = 30)

| Variant                 | Utilisation | Health MAE | Kendall τ   |
|-------------------------|-------------|------------|-------------|
| Full framework          | 0.65 ± 0.04 | 0.12 ± 0.01 | 0.59 ± 0.05 |
| Variant A (D-S in loop) | 0.60 ± 0.05 | 0.16 ± 0.02 | 0.50 ± 0.06 |
| Variant E (no adapt)    | 0.16 ± 0.04 | 0.09 ± 0.02 | 0.00 ± 0.00 |

The full table (six variants and per-metric standard deviation) is in
[`STATUS.md`](STATUS.md). Where the absolute numbers diverge from the
paper's headline numbers, the offset is documented in
[`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md).

## Hello, world

```python
from dds_adapt.runner import ExperimentRunner

runner = ExperimentRunner()                            # 8 satellites, ring + chords topology
profile, _ = runner.degradation_profile(n_steps=200)   # one-fault decay profile
metrics, log = runner.run_proposed(profile, seed=0)
print(metrics)
# {'global_cost': ..., 'utilization': ..., 'health_mae': ..., 'kendall_tau': ..., ...}
```

`log` is an `EngineLog` with per-tick health estimates, optimiser
targets, and consensus traces. See `scripts/make_figures.py` for how
to turn one into a plot.

## What is in the box

```
src/dds_adapt/
  config.py             topology + global parameters (Section 5.1.1 values)
  utils.py              Sinkhorn-Knopp + adapt_mixing_matrix (Eq. 5)
  simulator.py          planar double-integrator with health-modulated thrust
  residual.py           residual energy + GDM training corpus
  diagnostic.py         RPSGM + RPSR + OPT (Section 4.2)
  optimizer.py          DIGing with global gradient + W swapping
  cost.py               formation cost: tracking + coupling + safe-anchor
  baselines.py          Robust DO / FDI / Byzantine / D-S Dempster / Oracle
  metrics.py            three-layer, nine-metric system
  engine.py             closed-loop runtime: sim → diag → adapt → DIGing → control
  scenarios.py          Scenario 1, Scenario 2, topology, concurrent profiles
  learning_baseline.py  §5.5.4 sklearn MLP baseline
  runner.py             study drivers (comparative / ablation / extended)
  cli.py                ``dds-run`` entry point
scripts/
  reproduce.py          one-shot driver, dumps results/<seed>/ + figures/
  make_figures.py       13-figure renderer
tests/                  20 unit + integration tests
```

## CLI

```cmd
dds-run --experiments comparative   :: §5.4.1
dds-run --experiments ablation      :: §5.3
dds-run --experiments scenario2     :: §5.4.2
dds-run --experiments topology      :: §5.5.1
dds-run --experiments concurrent    :: §5.5.2
dds-run --experiments scale         :: §5.5.3
dds-run --experiments learning      :: §5.5.4
dds-run --experiments all
```

Common flags:

```cmd
--n-runs N        independent runs per condition (default 30)
--n-steps T       trajectory length in simulation steps (default 500)
--n-satellites N  formation size (default 8)
--seed S          global RNG seed
```

## Reproduce as a paper reviewer

```cmd
python scripts/reproduce.py --seed 0
```

writes:

```
results/seed0/
  meta.json          git commit, Python + dependency versions, config
  comparative.json   §5.4.1
  scenario2.json     §5.4.2
  ablation.json      §5.3
  topology.json      §5.5.1
  concurrent.json    §5.5.2
  scale.json         §5.5.3
  learning.json      §5.5.4
figures/
  fig_architecture.pdf
  fig_convergence_rate.pdf
  fig_convergence_time.pdf
  fig_health_sensitivity.pdf
  fig_gamma_sensitivity.pdf
  fig_scenario1_cost.pdf
  fig_scenario1_constraint.pdf
  fig_scenario2_cost.pdf
  fig_scenario2_variance.pdf
  fig_topology_robustness.pdf
  fig_multi_fault.pdf
  fig_scalability.pdf
  fig_high_fidelity.pdf
```

`meta.json` is what makes a specific number reproducible: it records
the git commit and the resolved dependency versions used to produce
the JSON files alongside it.

## Tests

```cmd
pytest          :: 20 tests covering every module + closed loop
ruff check .    :: lint
```

## Limitations

See `STATUS.md` and `KNOWN_DISCREPANCIES.md`. The biggest items:

* The NASA 42 high-fidelity track ships a Python stand-in; the real
  binary is open source (NOSA) and can be swapped in via the engine's
  `simulator_factory` parameter.
* Utilisation is normalised by a `(cost_no_adapt − cost_oracle)` band
  per diagnosis tick; the paper's exact denominator is not fully
  specified in §5.1.3, so absolute numbers differ while method
  ordering is preserved.
* The learning baseline uses scikit-learn's `MLPRegressor` so the
  public companion code installs without PyTorch / GPU.

## Citation

Please cite the paper (BibTeX entry to be added on publication) and
record the git commit hash from `results/<seed>/meta.json` so future
runs can be compared against the same code revision.

## License

MIT.
