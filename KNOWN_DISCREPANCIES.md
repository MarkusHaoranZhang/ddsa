# Known discrepancies between code output and paper numbers

Per-claim mapping of paper assertions to what the code produces on
`python scripts/reproduce.py --seed 0 --n-runs 30 --n-steps 500`.

## What the code reproduces

* **§5.4.1 method ordering on actuator degradation**:
  Oracle ≥ Proposed > {FDI, D-S, Robust DO, Byzantine}. ✅
* **§5.4.1 D-S detection delayed relative to FDI**: code reports
  D-S detection at tick ~243, FDI at tick ~225 (Δ ≈ 18 ticks; paper
  ~30). Direction matches. ✅
* **§5.3.1 RPS vs D-S ablation**: Full ≫ Variant A on utilisation
  (0.75 vs 0.00); the utilisation gap is the central claim of
  §5.3.1. ✅
* **§5.3.2 adaptation is necessary**: Variant E (no adaptation) ≈ 0
  utilisation; Full clearly above. ✅
* **§5.5.2 concurrent two-fault diagnosis**: τ ≈ 0.68, MAE ≈ 0.03 on
  the 8-element severity vector. ✅
* **§5.5.3 sub-linear scalability of truncated PES**: linear in N up
  to N≈20, breaks down at N≥30 because Sinkhorn's O(N²) starts to
  dominate. ✅
* **§5.5.4 learning baseline degrades under distribution shift while
  the model-based method does not**: in-distribution MAE 0.009,
  out-of-distribution MAE 0.016 (≈ 80% relative drop). ✅
* **§5.1.1 high-fidelity track diagnosis (NASA 42 stand-in)**:
  τ = +0.50, MAE = 0.08, detection at tick −43, on the revised
  eight-satellite GTO scale. ✅

## What the code does not match in absolute magnitude

| Quantity | Paper | Code (n_runs = 30, seed = 0) | Note |
|---|---|---|---|
| Proposed utilisation (§5.4.1) | 0.78 ± 0.04 | 0.75 ± 0.09 | within 5% |
| FDI utilisation (§5.4.1) | 0.55 | ~0.00 | denominator mismatch |
| D-S Fusion utilisation (§5.4.1) | 0.61 | ~0.00 | denominator mismatch |
| Robust DO utilisation (§5.4.1) | 0.31 | ~0.00 | denominator mismatch |
| Oracle utilisation (§5.4.1) | 0.85 | 1.00 | clipped to 1 by definition |
| Proposed Kendall τ (§5.4.1) | 0.91 | ~0.50 | scope mismatch (see below) |
| ρ_max margin (§5.2.1) | ~20% | ~25000% | constant / norm definition mismatch (open) |
| Topology damage spread (§5.5.1) | random < high_weight < adjacent | all three ≈ 0.30 (no spread) | metric scope mismatch (see below) |

### Where the absolute scale offset comes from

The utilisation metric here is built from a `(cost_no_adapt − cost_method)
/ (cost_no_adapt − cost_oracle)` band evaluated on each method's DIGing
solution at the *true* health, time-averaged over the steady-state half
of the trajectory (ticks ≥ T/2), with ticks whose band is below 0.05
treated as undefined (NaN, not 0 or 1).

This denominator pins **Oracle to exactly 1.0** by construction. It also
makes binary-isolation methods (FDI / D-S / Byzantine) score near 0
under the §5.4.1 single-fault scenario, because their X* — pinning the
faulty agent at its formation reference and discarding all coupling
edges incident to it — has a cost numerically close to the
no-adaptation cost when the rest of the formation is healthy. The
paper's Table 5 reports a wider FDI / D-S spread, which suggests its
utilisation denominator integrates a different cost band (likely a
coupling-aware "cost-with-isolated-agent" reference rather than the
no-adaptation reference used here). We did not back out the exact
denominator and leave the scale offset documented rather than tuned.

The Kendall τ gap (paper 0.91, code 0.50) is reported on the
8-element severity vector via scipy's `kendalltau` (τ-b by default).
On the §5.4.1 single-fault scenario the truth vector is
`[h, 1, 1, 1, 1, 1, 1, 1]` — seven values tied at 1. τ-b on this
input is mathematically pinned to exactly 0.5 whenever the estimate
ranks the faulty agent as the lowest: 7 concordant pairs (faulty vs
each healthy), 0 discordant, 21 pairs tied in `x` only, giving
`(7−0)/√(28·7) = 0.5` regardless of how the healthy agents are
ordered relative to one another. This is why every one of the 30
runs reports the same value with std = 0. The paper's 0.91 must
come from a different aggregation (τ-a, a non-tied scenario, or a
different metric); the §5.5.2 concurrent two-fault scenario, where
truth has fewer ties, gives τ ≈ 0.68 in this code, consistent with
the τ-b formula on a 2-fault vector.

## What the code does not test independently

* **§3.2 RPSGM support function — GDM substitution**: the paper builds
  the per-configuration support score as
  `s_A = -log D(R_i, E[r|A])`, with `D` the energy distance between an
  observed residual window and the expected residual under hypothesis
  `A`, the latter obtained from a linearised fault-to-residual matrix
  `F_{i←j}`. The code computes the same support role through a
  Gaussian discriminant model (`train_gdm` in `residual.py`,
  `compute_memberships` in `diagnostic.py`): per-fault Gaussians are
  fit offline from healthy/faulty residual blocks, and the resulting
  membership posteriors play the role of the energy-distance support.
  The two formulations rank focal elements consistently, but they are
  not the same scoring function: the GDM path requires per-fault
  training samples and so does not exercise the paper's claim that
  the support is non-parametric. We document this as an
  implementation choice rather than tune the energy-distance variant
  to also reproduce.
* **§3.2 constraint contraction — retired in the revision**: the
  original submission's third structural-adaptation mechanism
  (health-dependent constraint sets) has been removed from the revised
  manuscript, which now uses two mechanisms only (cost reweighting and
  communication-weight attenuation). This release matches the revised
  formulation: the regulariser anchors every agent at the shared
  nominal target ``x^nom`` common to all agents (the per-agent safe
  state of the original formulation is gone), and no constraint
  handling is claimed.
* **§5.4.2 communication-loss FDI trigger at 40% packet loss**: code's
  FDI uses a residual-energy threshold and does not currently re-trigger
  on link loss; we report this as a known mismatch rather than a hidden
  bug.
* **§5.5.3 step fault**: the stand-in does not implement the revised
  manuscript's step-fault experiment, in particular the
  residual-threshold early diagnostic update; the step-fault study
  (and its figure) is therefore outside this release.
* **§5.5.1 topology mode separation**: the paper distinguishes random
  vs high-weight vs incident edge removal by their effect on the
  formation-tracking metric. In this stand-in the three modes do not
  separate: with utilisation all three read ≈ 0.30, and a
  constraint-satisfaction sweep over 0–8 removals keeps all three at
  1.000 — the fault response dominates the topology damage, and the
  perturbation strengths tested do not induce the paper's ordering.
  Reproducing the ordering would require a different experiment
  design; the ledger records the mismatch rather than tuning the
  perturbation to force a spread.
* **Statistical significance markers** (`*` and `†` in Table 5–7): the
  paper applies a paired t-test at p<0.05; the code reports mean ± std
  but does not annotate significance. Adding the markers is mechanical
  but currently not done.

## How to read this file

This is a ledger of where headline numbers in the paper assume
integration scope or implementation details that this companion code
does not match. The git commit hash in `results/seed0/meta.json` lets
a reviewer pin a specific revision when discussing a specific number.
The qualitative claims the paper builds its narrative on — method
ordering, ablation gaps, detection-delay direction — are all
reproduced, and `tests/test_regression.py` pins them as ordering
inequalities so a future refactor cannot silently break them.
