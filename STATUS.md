# Project status

This is the public companion code for *Diagnosis-Driven Structural
Adaptation: A Closed-Loop Architecture for Elastic Degradation in
Satellite Formation Control*.

For installation, quick-start commands, reference numbers, and the
figure index, see [`README.md`](README.md). This file is kept lean and
serves a single purpose: a paper-section to code-path map so a reader
can jump from a §-reference in the manuscript to the file that
implements it.

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

Where the code's numbers diverge from the paper's headline numbers,
see [`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md) for the
per-claim mapping.
