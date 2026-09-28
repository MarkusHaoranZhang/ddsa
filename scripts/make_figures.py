"""Generate every figure referenced in the paper.

Each figure is saved as a vector PDF in ``figures/``. The script is
deliberately deterministic given a seed: it does not call into anything
that could change a random state outside its own scope.

Run:

    python scripts/make_figures.py

Optional arguments:

    --seed N         RNG seed (default 0)
    --out DIR        output directory (default figures/)
    --quick          smaller sweeps so the script finishes in seconds
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ddsa.config import Config
from ddsa.cost import formation_cost_global
from ddsa.engine import run_closed_loop
from ddsa.hf_runner import run_hf_diagnostic_experiment
from ddsa.runner import ExperimentRunner
from ddsa.scenarios import (
    actuator_degradation_profile,
    communication_loss_w_sequence,
    step_fault_profile,
    topology_removal_sequence,
)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


# ----------------------------------------------------- helpers
def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / name
    fig.savefig(fp, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fp}")


def _proposed_log(runner: ExperimentRunner, profile: np.ndarray, seed: int):
    _, log = runner.run_proposed(profile, seed)
    return log


# ----------------------------------------------------- 1. architecture
def fig_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    blocks = [
        ("Physical\nMAS", 0.05),
        ("Diagnostic\nmodule\n(RPS)", 0.27),
        ("Health\nestimate", 0.46),
        ("Structure\nadaptation\n(W, cost)", 0.65),
        ("DIGing\nsolver", 0.84),
    ]
    for label, x in blocks:
        ax.add_patch(
            plt.Rectangle((x - 0.06, 0.4), 0.12, 0.2, fc="#dde", ec="black", lw=0.8)
        )
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9)
    for x_from, x_to in zip(
        [0.11, 0.33, 0.52, 0.71], [0.21, 0.40, 0.59, 0.78], strict=True
    ):
        ax.annotate(
            "",
            xy=(x_to, 0.5),
            xytext=(x_from, 0.5),
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )
    # closing arrow
    ax.annotate(
        "",
        xy=(0.05, 0.4),
        xytext=(0.90, 0.4),
        arrowprops=dict(
            arrowstyle="->", lw=1.2, connectionstyle="arc3,rad=-0.35"
        ),
    )
    ax.text(
        0.5, 0.18, "control / actuation", ha="center", fontsize=9, style="italic"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.suptitle("Diagnosis-driven structural adaptation: closed loop", y=0.98)
    _save(fig, out_dir, "fig_architecture.pdf")


# ----------------------------------------------------- 2 + 3. convergence
def fig_convergence(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    # Sweep in units of the analytic bound rho_max (Theorem 1, C = 1 + 1/sqrt(N)),
    # matching the manuscript's rho/rho_max axis: sub-critical rates settle,
    # super-critical rates cannot track the moving optimum.
    mu_bar = min(Config.MU, runner.gamma)
    L_bar = max(Config.L_SMOOTH, runner.gamma)
    kappa = L_bar / mu_bar
    rho_max = runner.lambda2 * mu_bar / ((1.0 + 1.0 / np.sqrt(runner.n)) * kappa)
    ratios = (
        np.array([0.1, 0.5, 0.9, 1.0, 1.1, 1.3, 2.0, 5.0])
        if not quick
        else np.array([0.5, 0.9, 1.0, 1.3])
    )
    n_steps = 500 if not quick else 200
    histories: dict[float, list[float]] = {}
    convergence_times: list[float] = []

    for ratio in ratios:
        eta = float(ratio) * rho_max
        profile, _ = runner.degradation_profile(n_steps, eta=eta, seed=seed)
        log = _proposed_log(runner, profile, seed)
        hist = np.asarray(log.consensus_history, dtype=float)
        histories[float(ratio)] = hist.tolist()
        # convergence time: first iteration where error < 1.05 * final
        # tail mean. Rates at or beyond the bound cannot settle; report
        # NaN for them so the divergence boundary stays visible.
        if ratio <= 1.0 and hist.size >= 50:
            tail = np.mean(hist[-30:])
            crossings = np.where(hist <= 1.05 * tail)[0]
            convergence_times.append(
                float(crossings[0]) if crossings.size else float(hist.size)
            )
        else:
            convergence_times.append(float("nan"))

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for ratio, hist in histories.items():
        ax.plot(hist, label=rf"$\rho = {ratio:.1f}\rho_{{\max}}$", lw=1.2)
    ax.set_xlabel("DIGing iteration")
    ax.set_ylabel("consensus error")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Convergence under varying health-variation rate")
    _save(fig, out_dir, "fig_convergence_rate.pdf")

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(ratios, convergence_times, "o-", lw=1.2, label="iterations to settle")
    ax.axvline(1.0, color="red", ls="--", lw=1.0, label=r"$\rho_{\max}$ (Eq. 12)")
    if not quick:
        empirical_ratio = 0.0375 / max(rho_max, 1e-12)
        ax.axvline(
            empirical_ratio, color="black", ls=":", lw=1.0,
            label=fr"$\rho^*$ (empirical, +{100.0 * (empirical_ratio - 1.0):.0f}%)",
        )
    ax.set_xlabel(r"$\rho / \rho_{\max}$")
    ax.set_ylabel("iterations to settle")
    ax.set_title("Convergence time vs health-variation rate")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "fig_convergence_time.pdf")


# ----------------------------------------------------- 4. health sensitivity
def fig_health_sensitivity(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    sigmas = np.linspace(0.02, 0.25, 8) if not quick else np.linspace(0.05, 0.25, 4)
    means: list[float] = []
    stds: list[float] = []
    n_steps = 300
    n_reps = 3 if not quick else 2

    # Reference costs per repetition: the noise-free achieved cost and
    # the no-adaptation cost. The reported quantity is the normalised
    # optimality gap (0 = noise-free performance, 1 = no adaptation),
    # matching the manuscript's "optimality gap" axis.
    refs: list[tuple[float, float]] = []
    for run_idx in range(n_reps):
        local_seed = seed + run_idx * 17
        profile, _ = runner.degradation_profile(n_steps, seed=local_seed)
        m0, _ = runner._run_engine_method(profile, seed + run_idx)
        _, na_log = runner._run_no_adaptation(profile, seed + run_idx)
        na_cost = runner.metrics_from_log(na_log, profile)["global_cost"]
        _ = m0
        refs.append((m0["global_cost"], na_cost))

    for sigma in sigmas:
        gaps = []
        for run_idx in range(n_reps):
            local_seed = seed + run_idx * 17
            rng = np.random.default_rng(local_seed)
            profile, _ = runner.degradation_profile(n_steps, seed=local_seed)
            noisy = np.clip(profile + rng.normal(0, sigma, profile.shape), 0, 1)
            log = run_closed_loop(
                health_profile=profile,
                diagnostic=runner.diagnostic,
                W_base=runner.W,
                desired_positions=runner.desired_positions,
                edges=runner.edges,
                n_diag_intervals=10,
                iters_per_diag=200,
                alpha=runner.alpha,
                gamma=runner.gamma,
                beta=runner.beta,
                seed=seed + run_idx,
                health_estimate_override=noisy,
            )
            metric = runner.metrics_from_log(log, profile)
            c0, na = refs[run_idx]
            denom = max(na - c0, 1e-9)
            gaps.append(float((metric["global_cost"] - c0) / denom))
        means.append(float(np.mean(gaps)))
        stds.append(float(np.std(gaps)))

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.errorbar(sigmas, means, yerr=stds, fmt="o-", capsize=3)
    ax.set_xlabel(r"injected health-estimate noise $\sigma_\epsilon$")
    ax.set_ylabel("optimality gap")
    ax.set_title("Sensitivity to health-estimate noise")
    _save(fig, out_dir, "fig_health_sensitivity.pdf")


# ----------------------------------------------------- 5. gamma U-curve
def fig_gamma_sensitivity(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    gammas = [1, 5, 10, 15, 20, 30] if not quick else [1, 10, 30]
    n_steps = 300 if quick else 500
    util: list[float] = []
    util_std: list[float] = []
    retained: list[float] = []
    original = runner.gamma
    try:
        for gamma in gammas:
            runner.gamma = float(gamma)
            runs = []
            retained_runs = []
            for run_idx in range(2 if quick else 3):
                run_seed = seed + run_idx * 17
                profile, _ = runner.degradation_profile(n_steps, seed=run_seed)
                m, log = runner.run_proposed(profile, run_seed)
                _, oracle_log = runner.run_oracle(profile, run_seed)
                _, no_adapt_log = runner._run_no_adaptation(profile, run_seed)
                band = (
                    runner._per_tick_cost_at_true_health(oracle_log, profile),
                    runner._per_tick_cost_at_true_health(no_adapt_log, profile),
                )
                runs.append(runner._utilisation_timeseries(log, profile, band))
                # station-keeping fraction retained by the degraded agent:
                # 1 = still commanded to its own station (full use of its
                # remaining authority), 0 = parked at the nominal point.
                retained_runs.append(
                    float(
                        np.linalg.norm(log.optimiser_targets[-1][0])
                        / max(np.linalg.norm(runner.desired_positions[0]), 1e-9)
                    )
                )
            util.append(float(np.mean(runs)))
            util_std.append(float(np.std(runs)))
            retained.append(float(np.mean(retained_runs)))
    finally:
        runner.gamma = original

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    axes[0].errorbar(gammas, util, yerr=util_std, fmt="o-", capsize=3)
    axes[0].axvline(10, color="gray", ls=":", lw=0.8)
    axes[0].text(10.6, float(np.nanmax(util)) * 0.55, "operating\npoint",
                 fontsize=8, color="gray")
    axes[0].set_xlabel(r"regularisation strength $\gamma$")
    axes[0].set_ylabel("band utilisation")
    axes[0].set_title("Safety-performance trade-off")
    axes[1].plot(gammas, retained, "s-", color="C1")
    axes[1].axvline(10, color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel(r"regularisation strength $\gamma$")
    axes[1].set_ylabel("retained station-keeping fraction")
    axes[1].set_title("Capability use of the degraded agent")
    plt.tight_layout()
    _save(fig, out_dir, "fig_gamma_sensitivity.pdf")


# ----------------------------------------------------- 6 + 7. scenario 1 trajectories
def fig_scenario1(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    n_steps = 500 if not quick else 200
    profile, _ = actuator_degradation_profile(
        runner.n, n_steps, eta=Config.ETA_SINGLE, onset_time=80,
        rng=np.random.default_rng(seed),
    )

    methods = {
        "Proposed": runner.run_proposed,
        "Robust DO": runner.run_robust_do,
        "FDI-Reconf": runner.run_fdi,
        "D-S Fusion": runner.run_ds_fusion,
        "Byzantine": runner.run_byzantine,
        "Oracle": runner.run_oracle,
    }
    cost_traj: dict[str, list[float]] = {}
    constr_traj: dict[str, list[float]] = {}
    for name, fn in methods.items():
        _, log = fn(profile, seed)
        cost_traj[name] = [
            float(np.linalg.norm(p - runner.desired_positions) ** 2)
            for p in log.positions
        ]
        # Edge-level formation-keeping score per diagnosis tick (the
        # pair-level constraint rate saturates at the 2 m tolerance and
        # cannot separate the methods).
        constr_traj[name] = [
            runner._edge_constraint_rate(target)
            for target in log.optimiser_targets
        ]

    # Normalise by the Oracle's settled cost so the plot uses the same
    # Oracle = 1 convention as the manuscript's normalised index.
    oracle_ref = float(np.mean(cost_traj["Oracle"][-3:])) or 1.0
    cost_traj = {name: [v / oracle_ref for v in ys] for name, ys in cost_traj.items()}

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name, ys in cost_traj.items():
        ax.plot(log.diag_steps[: len(ys)], ys, "o-", label=name, lw=1.2)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("formation cost (Oracle = 1)")
    ax.set_title("Scenario 1: progressive actuator degradation")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_scenario1_cost.pdf")

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name, ys in constr_traj.items():
        ax.plot(log.diag_steps[: len(ys)], ys, "o-", label=name, lw=1.2)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("constraint satisfaction rate")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Scenario 1: constraint satisfaction over time")
    _save(fig, out_dir, "fig_scenario1_constraint.pdf")


# ----------------------------------------------------- 8 + 9. scenario 2
def fig_scenario2(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    n_steps = 500 if not quick else 200
    edges_to_drop = [(0, 1 % runner.n), (0, 2 % runner.n)]
    w_seq = communication_loss_w_sequence(runner.W, 10, edges_to_drop=edges_to_drop)
    n_runs = 3 if quick else 8

    methods = {
        "Proposed": runner.run_proposed,
        "Robust DO": runner.run_robust_do,
        "FDI-Reconf": runner.run_fdi,
    }
    traces: dict[str, np.ndarray] = {}
    for name, fn in methods.items():
        runs = []
        for run_idx in range(n_runs):
            run_seed = seed + run_idx * 17
            profile, _ = runner.degradation_profile(n_steps, seed=run_seed)
            _, log = fn(profile, run_seed, w_base_per_interval=w_seq)
            runs.append([
                float(np.linalg.norm(p - runner.desired_positions) ** 2)
                for p in log.positions
            ])
        traces[name] = np.stack(runs)  # (n_runs, n_ticks)
    xs = np.arange(traces["Proposed"].shape[1])

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name, arr in traces.items():
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(xs, mean, "o-", label=name, lw=1.2)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("diagnostic interval")
    ax.set_ylabel("formation tracking cost")
    ax.set_title("Scenario 2: communication-link degradation")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_scenario2_cost.pdf")

    # Across-run cost variance under the shared intermittent-loss
    # sequence (second half of the run). Reported as measured; the
    # manuscript's variance-reduction narrative comes from its synthetic
    # generator and is not reproduced by the closed loop (see
    # KNOWN_DISCREPANCIES.md).
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    names = list(traces.keys())
    values = []
    for name in names:
        arr = traces[name]
        half = arr.shape[1] // 2
        values.append(float(arr[:, half:].var(axis=0).mean()))
    ax.bar(names, values)
    ax.set_ylabel("mean across-run cost variance")
    ax.set_title("Scenario 2: cost variance under intermittent loss")
    _save(fig, out_dir, "fig_scenario2_variance.pdf")


# ----------------------------------------------------- 10. topology
def fig_topology(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    modes = ["random", "high_weight", "adjacent"]
    counts = list(range(0, 9)) if not quick else list(range(0, 5))
    n_steps = 300

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for mode in modes:
        ys: list[float] = []
        for n_rem in counts:
            n_run = 2 if quick else 3
            vals: list[float] = []
            for run_idx in range(n_run):
                run_seed = seed + run_idx * 17
                rng = np.random.default_rng(run_seed)
                agent_idx = int(rng.integers(0, runner.n))
                profile, _ = runner.degradation_profile(
                    n_steps, seed=run_seed, agent_idx=agent_idx
                )
                if n_rem == 0:
                    w_seq = None
                    edges_seq = None
                else:
                    w_seq, edges_seq = topology_removal_sequence(
                        runner.W, 10, mode=mode, n_removals=n_rem,
                        degraded_agent=agent_idx,
                        rng=np.random.default_rng(run_seed),
                    )
                m, log = runner.run_proposed(
                    profile, run_seed,
                    w_base_per_interval=w_seq,
                    edges_per_interval=edges_seq,
                )
                vals.append(runner._edge_constraint_rate(log.optimiser_targets[-1]))
            ys.append(float(np.mean(vals)))
        ax.plot(counts, ys, "o-", label=mode, lw=1.2)
    ax.axhline(0.94, color="gray", ls="--", lw=0.8, label="threshold 0.94")
    ax.set_xlabel("edges removed")
    ax.set_ylabel("formation-keeping error within bounds")
    ax.set_title("Topology robustness")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_topology_robustness.pdf")


# ----------------------------------------------------- 11. multi-fault
def fig_multi_fault(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    n_steps = 600 if not quick else 200
    onset = 80
    profile, _ = runner.concurrent_degradation_profile(
        n_steps, onset_time=onset, seed=seed
    )
    _, log = runner.run_proposed(profile, seed)
    h_est = np.stack(log.health_est)
    ts = log.diag_steps

    # analytic ground truth for a dense trace
    t = np.arange(0, n_steps)
    true_fast = np.exp(-Config.ETA_FAST * np.maximum(t - onset, 0))
    true_slow = np.exp(-Config.ETA_SLOW * np.maximum(t - onset, 0))
    bias_fast = float(abs(h_est[-1, 0] - profile[0, -1]))
    bias_slow = float(abs(h_est[-1, 1] - profile[1, -1]))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(t, true_fast, "k-", lw=1.2, label="true (fast)")
    ax.plot(ts, h_est[:, 0], "C0o-", lw=1.0, label="est. (fast)")
    ax.plot(t, true_slow, "k--", lw=1.2, label="true (slow)")
    ax.plot(ts, h_est[:, 1], "C1s-", lw=1.0, label="est. (slow)")
    ax.text(
        0.55, 0.9,
        f"final bias: fast {bias_fast:.3f}, slow {bias_slow:.3f}",
        transform=ax.transAxes, fontsize=8,
    )
    ax.set_xlabel("simulation step")
    ax.set_ylabel("health degree")
    ax.set_title("Concurrent degradation: estimate vs ground truth")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out_dir, "fig_multi_fault.pdf")


# ----------------------------------------------------- 12. scalability
def fig_scalability(out_dir: Path, seed: int, quick: bool):
    from math import perm

    sizes = [5, 8, 12, 16, 20, 30] if not quick else [5, 8, 12]
    walls: list[float] = []
    walls_std: list[float] = []
    n_diag = 10
    for size in sizes:
        sub = ExperimentRunner(n_satellites=size, seed=seed)
        ws = []
        for run_idx in range(2 if quick else 3):
            profile, _ = sub.degradation_profile(200, seed=seed + run_idx * 17)
            m, _ = sub.run_proposed(profile, seed + run_idx)
            ws.append(m["wall_time"])
        walls.append(float(np.mean(ws)))
        walls_std.append(float(np.std(ws)))

    # Per-permutation cost estimated from the smallest measured point, then
    # used to price the full (untruncated) permutation set: the analytical
    # complexity story of the manuscript's scalability panel. Both curves
    # are reported per diagnosis interval.
    unit_cost = walls[0] / n_diag / sum(
        perm(sizes[0], length) for length in range(1, Config.L_MAX + 1)
    )
    grid = list(range(5, 51))
    full_pes = [
        unit_cost * sum(perm(n, length) for length in range(1, n + 1))
        for n in grid
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.errorbar(
        sizes, np.array(walls) / n_diag, yerr=np.array(walls_std) / n_diag,
        fmt="o-", capsize=3, label="truncated PES (L=3, measured)",
    )
    ax.plot(grid, full_pes, "s--", lw=1.2, label="full PES (all lengths)")
    ax.axhline(1e4, color="gray", ls=":", lw=0.9)
    ax.text(
        50, 1.2e4, "practical ceiling ($10^{4}$ s)",
        fontsize=8, color="gray", ha="right", va="bottom",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 2e5)
    ax.set_xlabel(r"formation size $N$")
    ax.set_ylabel("wall time per diagnosis interval (s)")
    ax.set_title("Scalability of truncated RPS pipeline")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_scalability.pdf")


# ----------------------------------------------------- 13. high-fidelity track
def fig_high_fidelity(out_dir: Path, seed: int, quick: bool):
    """Diagnostic accuracy on the NASA 42 stand-in (8-sat GTO)."""
    hf = run_hf_diagnostic_experiment(
        n_steps=200 if quick else 600, seed=seed
    )
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    n = hf.true_per_agent.size
    idx = np.arange(n)
    width = 0.35
    ax.bar(idx - width / 2, hf.true_per_agent, width, label="true (mean)")
    ax.bar(
        idx + width / 2, hf.mean_estimate_per_agent, width, label="estimate (mean)"
    )
    ax.set_xticks(idx)
    ax.set_xticklabels([f"sat {i}" for i in range(n)])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("health degree")
    ax.set_title(
        f"HF stand-in: MAE={hf.health_mae:.3f}, "
        f"τ={hf.kendall_tau:.2f}, det={hf.detection_delay:.0f} steps"
    )
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out_dir, "fig_high_fidelity.pdf")


# ----------------------------------------------------- 14. step fault
def fig_step_fault(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    """Progressive decay vs abrupt step drop with the early trigger.

    Cost is the global formation cost at the *true* health evaluated on
    the per-step physical formation state, so the plot shows the real
    transient (rise, peak, recovery) rather than coarse diagnosis-tick
    samples. The measured overshoot and recovery are annotated on the
    step trace.
    """
    n_steps = 300 if quick else 500
    onset = min(Config.STEP_FAULT_ONSET, n_steps // 2)
    prog, _ = runner.degradation_profile(n_steps, seed=seed)
    step, _ = step_fault_profile(
        runner.n, n_steps,
        onset_time=onset,
        health_after=Config.STEP_FAULT_HEALTH_AFTER,
    )
    _, log_p = runner.run_proposed(prog, seed, trace_positions=True)
    _, log_s = runner.run_proposed(
        step, seed,
        early_trigger_threshold=Config.STEP_FAULT_TRIGGER,
        early_trigger_check_every=Config.STEP_FAULT_CHECK_EVERY,
        trace_positions=True,
    )

    def trace(log, profile):
        steps = log.position_step_trace
        return steps, [
            formation_cost_global(
                positions, profile[:, min(step, profile.shape[1] - 1)],
                runner.desired_positions, runner.edges,
                beta=runner.beta, gamma=runner.gamma,
            )
            for step, positions in zip(
                steps, log.position_trace, strict=False
            )
        ]

    xs_p, ys_p = trace(log_p, prog)
    xs_s, ys_s = trace(log_s, step)
    ys_s_arr = np.asarray(ys_s)
    steady = float(np.mean(ys_s_arr[-max(1, len(ys_s_arr) // 10):]))
    post = ys_s_arr[len(ys_s_arr) - (n_steps - onset):]
    peak_idx = int(np.argmax(post))
    peak = float(post[peak_idx])
    overshoot = 100.0 * (peak - steady) / steady if steady > 0 else float("nan")
    recovery = -1
    for idx in range(peak_idx, len(post)):
        if post[idx] <= 1.05 * steady:
            recovery = idx
            break

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(xs_p, ys_p, lw=1.4, label="Progressive degradation")
    ax.plot(xs_s, ys_s, lw=1.4, label="Step fault (early trigger)")
    ax.axvline(onset, color="gray", ls="--", lw=0.8)
    ax.annotate(
        f"overshoot {overshoot:.0f}%\nrecovery {recovery} steps",
        xy=(onset + peak_idx, peak),
        xytext=(onset + 0.25 * (n_steps - onset), peak + 0.15 * peak),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel("simulation step")
    ax.set_ylabel("global cost at true health")
    ax.set_title("Step-fault transient response")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_step_fault.pdf")


# ----------------------------------------------------- driver
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out = args.out
    seed = args.seed
    quick = args.quick

    runner = ExperimentRunner(n_satellites=Config.NUM_SATELLITES, seed=seed)
    print(f"Generating figures into {out.resolve()} (quick={quick})")

    fig_architecture(out)
    fig_convergence(out, runner, seed, quick)
    fig_health_sensitivity(out, runner, seed, quick)
    fig_gamma_sensitivity(out, runner, seed, quick)
    fig_scenario1(out, runner, seed, quick)
    fig_scenario2(out, runner, seed, quick)
    fig_topology(out, runner, seed, quick)
    fig_multi_fault(out, runner, seed, quick)
    fig_step_fault(out, runner, seed, quick)
    fig_scalability(out, seed, quick)
    fig_high_fidelity(out, seed, quick)

    print("done.")


if __name__ == "__main__":
    main()
