# Known discrepancies between code output and paper numbers

Per-claim mapping of paper assertions to what the code produces on
`python scripts/reproduce.py --seed 0 --n-runs 10 --n-steps 800`.

## Headline claims that hold

* **Method ordering on actuator degradation (§5.4.1)**: Oracle is the
  upper bound; FDI/Byzantine sit clearly below adaptive methods on
  utilisation; Robust DO degrades when degradation crosses its margin.
  ✅
* **D-S detection delayed relative to FDI (§5.4.1)**: code reports
  D-S detection at tick 271, FDI at tick 239 (Δ ≈ 32 ticks; paper says
  Δ ≈ 30 ticks). ✅
* **RPS vs D-S in ablation (§5.3.1)**: Full > Variant A on utilisation
  (0.65 vs 0.60) and Kendall τ (0.59 vs 0.50), with comparable MAE.
  This matches the paper's central claim ("the value of RPS fusion
  lies not in producing more accurate health estimates, but in
  preserving the priority ordering"). ✅
* **Adaptation is necessary (§5.3.2)**: Variant E (no adaptation)
  utilisation 0.16 vs Full 0.65. ✅
* **FDI in §5.4.2 isolates unnecessarily under communication
  degradation**: code shows FDI utilisation drop from 0.66 (scenario
  1) to 0.29 (scenario 2). ✅
* **Topology robustness ordering (§5.5.1)**: random < high-weight <
  adjacent in damage. ✅
* **Sub-linear scalability of truncated PES (§5.5.3)**: ~0.1 s at
  N = 5 → ~0.7 s at N = 30. ✅
* **Learning baseline OOD degradation (§5.5.4)**: code +75%, paper
  +36%. Direction matches. ✅

## Headline claims with magnitude offset

| Quantity | Paper | Code |
|---|---|---|
| Proposed utilisation (§5.4.1) | 0.78 ± 0.04 | 0.66 ± 0.04 |
| Oracle utilisation (§5.4.1) | 0.85 ± 0.03 | 1.00 ± 0.00 |
| Proposed Kendall τ (§5.4.1) | 0.91 | 0.59 |
| FDI utilisation (§5.4.1) | 0.55 | 0.66 |
| D-S Fusion utilisation (§5.4.1) | 0.61 | 0.65 |
| Variant A utilisation gap (§5.3.1) | 15% | 8% |

These numbers sit on a `(cost_no_adapt − cost_method) / (cost_no_adapt
− cost_oracle)` band averaged over diagnosis ticks. The paper's
§5.1.3 prose defines utilisation by analogy without pinning the
denominator; this implementation chose a method-independent
no-adaptation/oracle cost band, which fixes Oracle at 1 by
construction. A different denominator would shift Proposed and FDI
proportionally; the *ordering* between methods is what the §5.3.1
ablation establishes, and that ordering is reproduced.

The Kendall τ gap (paper 0.91, code 0.59) is the second persistent
offset. We use scipy's tau-b on the severity vector. The paper's 0.91
in single-fault scenarios may be measured on the Satellite-A-vs-rest
binary ranking rather than the full 8-agent severity ordering;
`metrics.compute_metrics` could expose both variants if desired.

## Claims with no exact paper number

* **§5.1.1 high-fidelity track**. The paper uses NASA 42; this code
  uses a Python stand-in covering the same physics (J2 + bearing
  friction + SRP + gravity gradient + residual drag near perigee).
  Diagnostic accuracy on the GTO stand-in: MAE ≈ 0.4, τ ≈ 0.3.
* **§5.2.1 ρ_max calibration**. The paper claims a ~20% empirical
  margin above the theoretical bound. The code reports the back-
  solved C constant from the empirical critical rate so a reviewer
  can plug it into Eq. 12 with their own choice of prior σ^M and
  L_h. We did not tune the prior post-hoc to recover the 20%.

## How to read this file

This is a ledger of where headline numbers in the paper assume
integration scope or implementation details that this companion code
does not match. The git commit hash in `results/seed0/meta.json` lets
a reviewer pin a specific revision when discussing a specific number.
