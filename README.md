# Diagnosis-Driven Structural Adaptation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20434143.svg)](https://doi.org/10.5281/zenodo.20434143)

Companion code for *Diagnosis-Driven Structural Adaptation: A
Closed-Loop Architecture for Elastic Degradation in Satellite Formation Control*
(Haoran Zhang, Lining Xing et al., 2025).

> **Read first**: [`STATUS.md`](STATUS.md) lists every paper section
> and what code path implements it. [`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md)
> lists where the code's current numbers differ from the paper's
> headline numbers and why.
>
> **Note on the high-fidelity track**: Section 5.1.1 of the paper uses
> NASA 42 (open source under NOSA). This repository ships a
> self-contained Python stand-in (`hf_simulator.py`) covering the same
> physics (J2 + bearing friction + SRP + gravity gradient + residual
> drag near perigee) at the paper's eight-satellite GTO scale. The engine accepts a `simulator_factory`, so a
> wrapper around the real binary plugs in without touching the rest of
> the pipeline.

## Quick start

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev,learning,plot]
pytest                                       :: 58 tests, ~3 min
ruff check .                                 :: lint
mypy src/ddsa                                :: type-check
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
dependency set reproduces the tables below. Utilisation magnitudes now
match the manuscript's Table 3 (comparative summary), Tables 1-2
(ablations) and Table 4 (high-fidelity); the band
denominator, the steady-state window, the per-track ceiling, and the
baseline safe-hold constants are documented in
[`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md) and `config.py`.

### §5.4.1 Comparative (single-fault actuator degradation, n_runs = 30)

| Method      | Utilisation | Health MAE | Kendall τ | Detection delay |
|-------------|-------------|------------|-----------|-----------------|
| Oracle      | 0.85        | 0.00       | 1.00      | ~25             |
| Proposed    | 0.78        | 0.06       | 1.00      | 0               |
| Robust DO   | 0.31        | —          | —         | —               |
| FDI-Reconf  | 0.54        | —          | —         | ~225            |
| D-S Fusion  | 0.60        | —          | —         | ~243            |
| Byzantine   | 0.59        | —          | —         | —               |

Ordering: Oracle ≥ Proposed > D-S / Byzantine / FDI > Robust DO; D-S
detection delayed relative to FDI; MAE / τ NaN for methods that do not
estimate continuous health. τ comes from the severity-ladder benchmark
(GDM severity regression), not from the online OPT estimate.

### §5.3 Ablation (n_runs = 30)

| Variant                 | Utilisation |
|-------------------------|-------------|
| Full framework          | 0.78        |
| Variant A (D-S in loop) | 0.66        |
| Variant B (Average)     | 0.57        |
| Variant C (No Sinkhorn) | 0.78        |
| Variant D (binary)      | 0.52        |
| Variant E (no adapt)    | 0.00        |

The full per-metric standard deviation is in
[`results/seed0/ablation.json`](results/seed0/ablation.json).

## Hello, world

```python
from ddsa.runner import ExperimentRunner

runner = ExperimentRunner(verbose=True)                # ~2 s to train the GDM
profile, _ = runner.degradation_profile(n_steps=200)   # one-fault decay profile
metrics, log = runner.run_proposed(profile, seed=0)
print(metrics)
# {'global_cost': ..., 'utilization': ..., 'health_mae': ..., 'kendall_tau': ..., ...}
```

`log` is an `EngineLog` with per-tick health estimates, optimiser
targets, and consensus traces. See `scripts/make_figures.py` for how
to turn one into a plot.

## Debugging a comparative number

Once you have a concrete number you want to explain, the most direct
path is to inspect the `EngineLog` produced by a single trajectory:

```python
from ddsa.runner import ExperimentRunner

runner = ExperimentRunner(verbose=True)
profile, onset = runner.degradation_profile(n_steps=500, eta=0.002, seed=0)
metrics, log = runner.run_proposed(profile, seed=0)

for k, h_hat in enumerate(log.health_est):           # per-diagnosis-tick estimate
    print(f"tick {log.diag_steps[k]:>4}  h_hat = {h_hat.round(2)}")

# log.optimiser_targets[k]  - DIGing's per-agent target at tick k
# log.consensus_history     - raw DIGing trace (for fig_convergence_*)
# log.detection_tick        - first tick where any h_hat dropped below 0.9
# log.iters_to_consensus[k] - how many DIGing iters were needed at tick k
```

The utilisation number printed in the comparative table is the
time-averaged form built by `ExperimentRunner._utilisation_timeseries`
on top of an Oracle / no-adaptation cost band; reading that method
explains the absolute scale, and `KNOWN_DISCREPANCIES.md` discusses
why this scale differs from the paper's headline numbers.

## What is in the box

Top-level layout: `src/ddsa/` (the package), `scripts/`
(`reproduce.py`, `make_figures.py`), `tests/`. For a paper-section to
code-path map see [`STATUS.md`](STATUS.md).

## Figure index

Each figure answers a specific question. If a hyperparameter sweep
matters to your decision, the figure here is the cheapest way to see
its effect; rerun it with `python scripts/make_figures.py --seed 0`.

| Question                                                        | Figure                            | Paper §  |
|-----------------------------------------------------------------|-----------------------------------|----------|
| What is the closed-loop architecture?                           | `fig_architecture`                | Fig. 1   |
| How does ρ (health-variation rate) affect convergence?          | `fig_convergence_rate / time`     | §5.2.1   |
| How does diagnostic noise σ affect cost?                        | `fig_health_sensitivity`          | §5.2.2   |
| How does γ trade off safety vs exploitation?                    | `fig_gamma_sensitivity`           | §5.2.3   |
| Scenario 1 (single-fault actuator): cost / constraint over time | `fig_scenario1_cost / constraint` | §5.4.1   |
| Scenario 2 (communication loss): cost / variance over time      | `fig_scenario2_cost / variance`   | §5.4.2   |
| How does topology damage degrade constraint satisfaction?       | `fig_topology_robustness`         | §5.5.1   |
| Two concurrent faults: estimate vs ground truth                 | `fig_multi_fault`                 | §5.5.2   |
| Wall time vs formation size N                                   | `fig_scalability`                 | §5.5.3   |
| Diagnostic accuracy on the high-fidelity GTO stand-in           | `fig_high_fidelity`               | §5.1.1   |

## CLI

```cmd
ddsa --experiments comparative   :: §5.4.1
ddsa --experiments ablation      :: §5.3
ddsa --experiments scenario2     :: §5.4.2
ddsa --experiments topology      :: §5.5.1
ddsa --experiments concurrent    :: §5.5.2
ddsa --experiments scale         :: §5.5.3
ddsa --experiments learning      :: §5.5.4
ddsa --experiments all
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
  meta.json              git commit, Python + dependency versions, config
  comparative.json       §5.4.1
  scenario2.json         §5.4.2
  ablation.json          §5.3
  topology.json          §5.5.1
  concurrent.json        §5.5.2
  scale.json             §5.5.3
  learning.json          §5.5.4
  high_fidelity.json     §5.1.1 high-fidelity track diagnostic
  rho_calibration.json   §5.2.1 ρ_max sweep
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
pytest          :: 58 tests covering every module, the closed loop, the CLI, scenario boundaries, and ordering-inequality regression of the comparative + ablation studies
ruff check .    :: lint
mypy src/ddsa   :: type-check
```

## Limitations

See [`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md) for the full
ledger. The biggest items:

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
