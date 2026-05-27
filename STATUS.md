# Project status

This is the public companion code for *Diagnosis-Driven Structural
Adaptation: A Closed-Loop Architecture for Elastic Degradation in
Distributed Optimization*.

## Coverage map

| Paper section | Code | Status |
|---|---|---|
| §3 model + §4.2 RPS pipeline | `simulator.py` + `residual.py` + `diagnostic.py` + `engine.py` | ✅ end-to-end |
| §5.1.1 numerical track | `simulator.py` + `cost.py` | ✅ |
| §5.1.1 high-fidelity track (NASA 42) | `hf_simulator.py` + `hf_runner.py` | ✅ Python stand-in (NASA 42 itself is open source under NOSA; the engine accepts a `simulator_factory`) |
| §5.1.2 baselines (5) | `baselines.py` + `runner.py` | ✅ all five |
| §5.1.3 metrics (3 layers × 3 metrics) | `metrics.py` + engine logs | ✅ |
| §5.2.1 convergence validation | `rho_max_calibration.py` | ✅ |
| §5.2.2 estimation-error sensitivity | `make_figures.fig_health_sensitivity` | ✅ |
| §5.2.3 γ U-curve | `make_figures.fig_gamma_sensitivity` | ✅ |
| §5.3 ablation (5 variants) | `runner.run_ablation` | ✅ |
| §5.4.1 scenario 1 (actuator) | `runner.run_comparative` | ✅ |
| §5.4.2 scenario 2 (communication) | `runner.run_communication_scenario` | ✅ |
| §5.5.1 topology robustness | `runner.run_topology_robustness` | ✅ |
| §5.5.2 concurrent degradation | `runner.run_concurrent_degradation` | ✅ |
| §5.5.3 scalability | `runner.run_scalability` | ✅ |
| §5.5.4 learning baseline | `learning_baseline.py` (sklearn MLP) | ✅ |
| Figures (13 PDFs) | `scripts/make_figures.py` | ✅ |
| One-shot reproduce driver | `scripts/reproduce.py` | ✅ |

## How to reproduce

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev,learning,plot]
pytest                                       :: 32 tests
ruff check .                                 :: lint
mypy src/dds_adapt                           :: type-check
python scripts/reproduce.py --seed 0         :: full run, ~3 minutes
python scripts/reproduce.py --quick --seed 0 :: smoke run, < 1 minute
```

`results/seed0/meta.json` records the git commit hash and dependency
versions, so a reviewer can pin everything they need to reproduce a
specific number.

## Numerical signal

`scripts/reproduce.py --seed 0 --n-runs 10 --n-steps 800`:

### §5.4.1 Comparative (single-fault actuator degradation)

| Method      | Utilisation | Health MAE | Kendall τ | Detection delay |
|-------------|-------------|------------|-----------|-----------------|
| Oracle      | 1.00        | 0.00       | 1.00      | 47              |
| Proposed    | 0.66        | 0.12       | 0.59      | 0               |
| FDI-Reconf  | 0.66        | 0.04       | 1.00      | 239             |
| D-S Fusion  | 0.65        | 0.04       | 1.00      | 271             |
| Robust DO   | 0.52        | 0.31       | 0.00      | 0               |
| Byzantine   | 0.16        | 0.09       | 0.00      | —               |

Method ordering matches §5.4.1 of the paper. D-S detection lands ~30
ticks after FDI ("D-S detects at approximately t=310 versus FDI's
t=280" in the paper); Proposed has zero detection delay because it
runs continuous adaptation rather than a threshold-triggered switch.

### §5.3 Ablation

| Variant                  | Utilisation | Health MAE | Kendall τ |
|--------------------------|-------------|------------|-----------|
| Full framework           | 0.65        | 0.12       | 0.59      |
| Variant A (D-S in loop)  | 0.60        | 0.16       | 0.50      |
| Variant B (Avg fusion)   | 0.70        | 0.06       | 0.55      |
| Variant C (No Sinkhorn)  | 0.65        | 0.12       | 0.59      |
| Variant D (binary 0.5)   | 0.52        | 0.03       | 1.00      |
| Variant E (no adapt)     | 0.16        | 0.09       | 0.00      |

Full > Variant A on utilisation, Kendall τ. Variant E collapses,
validating that adaptation is necessary.

### §5.5.2 Concurrent degradation

| Method | Utilisation | Health MAE | Kendall τ |
|---|---|---|---|
| Proposed | 0.23 | 0.16 | 0.60 |

(Single-method trace, matching Figure 11 of the paper.)

### §5.5.4 Learning baseline

In-distribution MAE ≈ 0.009; out-of-distribution MAE ≈ 0.016 (+75%
degradation). Direction matches the paper's "model-based generalises
better than data-driven".

## Known caveats

* The high-fidelity track uses a self-contained Python stand-in
  (`hf_simulator.py`). NASA 42 is open source (NOSA), and the engine
  accepts a `simulator_factory` so a wrapper around the real binary
  drops in.
* Utilisation is normalised by `(cost_no_adapt - cost_oracle)` per
  diagnosis tick and time-averaged. The paper's exact denominator is
  not fully specified in §5.1.3; alternative defensible
  normalisations would shift the absolute number while preserving
  the method ordering.
* See `KNOWN_DISCREPANCIES.md` for the per-claim mapping between
  paper numbers and code outputs.
