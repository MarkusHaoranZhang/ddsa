# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] – 2025-05-27

Initial public release accompanying the paper *Diagnosis-Driven
Structural Adaptation: A Closed-Loop Architecture for Elastic
Degradation in Distributed Optimization*.

### Added

* End-to-end closed-loop runtime (`engine.py`) wiring together:
  the planar-formation simulator, the residual generator, the RPS
  diagnostic module (RPSGM + RPSR + OPT), the Sinkhorn-attenuated
  mixing-matrix adaptation, the DIGing distributed optimiser, and
  the control feedback into the simulator.
* Five baselines: Robust DO, FDI-Reconf, Byzantine-Resilient,
  D-S Fusion (threshold-based reconfiguration variant for §5.4 and
  closed-loop variant for §5.3), and Oracle.
* Three-layer, nine-metric evaluation system (`metrics.py`).
* High-fidelity GTO stand-in (`hf_simulator.py` + `hf_runner.py`)
  modelling J2, reaction-wheel bearing friction, solar radiation
  pressure, gravity gradient, and residual atmospheric drag near
  perigee.
* `rho_max_calibration.py` for the §5.2.1 ρ-sweep plus back-solved
  C constant from the empirical critical rate.
* `learning_baseline.py` (sklearn MLP) for §5.5.4.
* `scripts/reproduce.py`: one-shot driver that produces every JSON
  + every figure with a single command.
* `scripts/make_figures.py`: 13 vector PDFs covering the paper's
  figures plus a high-fidelity diagnostic plot.
* 20 pytest tests covering each module and the closed-loop pipeline.
* GitHub Actions CI (`.github/workflows/ci.yml`) running pytest
  and ruff on Ubuntu and Windows for Python 3.10 / 3.12.
* `STATUS.md` and `KNOWN_DISCREPANCIES.md` documenting per-claim
  mapping between paper assertions and code outputs.

### Notes for reviewers

* `requirements-lock.txt` records the exact versions used to
  produce the numerical signal reported in `STATUS.md`. CI tests
  against fresher versions; the lock is for byte-identical
  reproduction only.
* The high-fidelity track ships a Python stand-in. NASA 42 itself
  is open source under the NASA Open Source Agreement (NOSA), and
  `engine.run_closed_loop` accepts a `simulator_factory` so a
  wrapper around the real binary can be plugged in without
  touching the rest of the pipeline.
