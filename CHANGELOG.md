# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The repository is in pre-release state; no version tag has been cut yet.
The list below is the planned content of the first release (`v0.1.0`).

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
* 58 pytest tests: per-module unit tests, closed-loop integration,
  CLI dispatch coverage, scenario-builder boundary tests, and
  ordering-inequality regression tests for the §5.4.1 comparative
  study and §5.3 ablation (Oracle ≥ Proposed > all binary baselines;
  Full > Variant A; Variant E ≈ 0). Line coverage is 93% overall;
  uncovered surface is concentrated in `learning_baseline.py`
  (sklearn-only path, exercised by `reproduce.py` not pytest).
* `mypy --strict` clean across all 18 source files (the only
  disabled check is `no-any-return`, an artefact of numpy's
  partial generic stubs rather than a project-side issue).
* GitHub Actions CI (`.github/workflows/ci.yml`) running ruff +
  mypy + pytest + a `reproduce.py --quick` end-to-end smoke on
  Ubuntu and Windows for Python 3.10 / 3.12.
* `STATUS.md` and `KNOWN_DISCREPANCIES.md` documenting per-claim
  mapping between paper assertions and code outputs.

### Notes for reviewers

* `requirements-lock.txt` records the exact versions used to
  produce the reference numbers in `README.md` and the JSON files
  under `results/seed0/`. CI tests against fresher versions; the
  lock is for byte-identical reproduction only.
* The high-fidelity track ships a Python stand-in. NASA 42 itself
  is open source under the NASA Open Source Agreement (NOSA), and
  `engine.run_closed_loop` accepts a `simulator_factory` so a
  wrapper around the real binary can be plugged in without
  touching the rest of the pipeline.
