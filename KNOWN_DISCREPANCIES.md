# Known discrepancies between code output and paper numbers

Per-claim mapping of paper assertions to what the code produces on
`python scripts/reproduce.py --seed 0 --n-runs 30 --n-steps 500`.

## What the code reproduces

* **§5.4.1 method ordering on actuator degradation**:
  Oracle ≥ Proposed > {FDI, D-S, Robust DO, Byzantine}. ✅
* **§5.4.1 D-S detection delayed relative to FDI**: code reports
  D-S detection at tick ~243, FDI at tick ~225 (Δ ≈ 18 ticks; paper
  ~30). Direction matches. ✅
* **§5.3.1 RPS vs D-S ablation**: Full > Variant A on utilisation; the
  utilisation gap is the central claim of §5.3.1. ✅
* **§5.3.2 adaptation is necessary**: Variant E (no adaptation) ≈ 0
  utilisation; Full clearly above. ✅
* **§5.5.1 topology robustness ordering**: random < high_weight <
  adjacent in damage. ✅
* **§5.5.3 sub-linear scalability of truncated PES**: linear in N up
  to N≈20, breaks down at N≥30 because Sinkhorn's O(N²) starts to
  dominate. ✅
* **§5.5.4 learning baseline degrades under distribution shift while
  the model-based method does not**. Direction matches; the magnitude
  is reported on a different metric (see below).

## What the code does not match in absolute magnitude

| Quantity | Paper | Code | Note |
|---|---|---|---|
| Proposed utilisation (§5.4.1) | 0.78 ± 0.04 | ~0.45 ± 0.08 | scale offset |
| FDI utilisation (§5.4.1) | 0.55 | ~0.00 | scale offset |
| D-S Fusion utilisation (§5.4.1) | 0.61 | ~0.00 | scale offset |
| Robust DO utilisation (§5.4.1) | 0.31 | ~0.00 | scale offset |
| Oracle utilisation (§5.4.1) | 0.85 | 1.00 | clipped to 1 by definition |
| Proposed Kendall τ (§5.4.1) | 0.91 | ~0.59 | scope mismatch |

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

The Kendall τ gap (paper 0.91, code 0.59) is reported on the
8-element severity vector via scipy's `kendalltau`. The paper's 0.91
likely measures τ on the binary "is this agent the most-degraded one"
ranking, which is a strictly easier subproblem.

## What the code does not test independently

* **§3.2 Equation 6 constraint contraction**: the third structural-
  adaptation mechanism is in the paper but every experiment uses
  unconstrained quadratic costs. The paper itself acknowledges this in
  §5.6.3 Limitations.
* **§5.4.2 communication-loss FDI trigger at 40% packet loss**: code's
  FDI uses a residual-energy threshold and does not currently re-trigger
  on link loss; we report this as a known mismatch rather than a hidden
  bug.
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
