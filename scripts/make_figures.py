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
from ddsa.engine import run_closed_loop
from ddsa.hf_runner import run_hf_diagnostic_experiment
from ddsa.rho_max_calibration import calibrate_rho_max
from ddsa.runner import ExperimentRunner
from ddsa.scenarios import (
    actuator_degradation_profile,
    communication_loss_w_sequence,
    concurrent_degradation_profile,
    perturb_topology,
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
    rho_factors = np.array([0.1, 0.5, 1.0, 1.3, 2.0, 3.0]) if not quick else np.array([0.5, 1.0, 2.0])
    n_steps = 500 if not quick else 200
    histories: dict[float, list[float]] = {}
    convergence_times: list[float] = []

    for rho_factor in rho_factors:
        eta = rho_factor * Config.CHARACTERISTIC_HEALTH_RATE
        profile, _ = runner.degradation_profile(n_steps, eta=eta, seed=seed)
        log = _proposed_log(runner, profile, seed)
        hist = np.asarray(log.consensus_history, dtype=float)
        histories[float(rho_factor)] = hist.tolist()
        # convergence time: first iteration where error < 1.05 * final tail mean
        if hist.size >= 50:
            tail = np.mean(hist[-30:])
            crossings = np.where(hist <= 1.05 * tail)[0]
            convergence_times.append(float(crossings[0]) if crossings.size else float(hist.size))
        else:
            convergence_times.append(float(hist.size))

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for rho_factor, hist in histories.items():
        ax.plot(hist, label=rf"$\rho = {rho_factor:.1f}\rho_{{\max}}$", lw=1.2)
    ax.set_xlabel("DIGing iteration")
    ax.set_ylabel("consensus error")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Convergence under varying health-variation rate")
    _save(fig, out_dir, "fig_convergence_rate.pdf")

    # convergence time vs rho/rho_max with the calibrated rho_max
    cal = calibrate_rho_max(
        runner,
        rho_factors=rho_factors,
        seed=seed,
        n_steps=n_steps,
        n_repeats=1 if quick else 2,
    )
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(rho_factors, convergence_times, "o-", lw=1.2, label="empirical")
    ax.axvline(
        cal.rho_star_empirical / Config.CHARACTERISTIC_HEALTH_RATE,
        color="red",
        ls="--",
        lw=1.0,
        label=fr"$\rho^*$ (empirical, +{cal.margin_percent:.0f}% margin)",
    )
    ax.axvline(
        cal.rho_max_theoretical / Config.CHARACTERISTIC_HEALTH_RATE,
        color="black",
        ls=":",
        lw=1.0,
        label=r"$\rho_{\max}$ (Eq. 12)",
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

    for sigma in sigmas:
        gaps = []
        for run_idx in range(3 if not quick else 2):
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
                iters_per_diag=50,
                alpha=runner.alpha,
                gamma=runner.gamma,
                beta=runner.beta,
                seed=seed + run_idx,
                health_estimate_override=noisy,
            )
            metric = runner.metrics_from_log(log, profile)
            gaps.append(metric["global_cost"])
        means.append(float(np.mean(gaps)))
        stds.append(float(np.std(gaps)))

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.errorbar(sigmas, means, yerr=stds, fmt="o-", capsize=3)
    ax.set_xlabel(r"injected health-estimate noise $\sigma_\epsilon$")
    ax.set_ylabel("global cost (lower = better)")
    ax.set_title("Sensitivity to health-estimate noise")
    _save(fig, out_dir, "fig_health_sensitivity.pdf")


# ----------------------------------------------------- 5. gamma U-curve
def fig_gamma_sensitivity(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    gammas = [1, 5, 10, 15, 20, 30] if not quick else [1, 10, 30]
    util: list[float] = []
    util_std: list[float] = []
    original = Config.GAMMA_NUM
    try:
        for gamma in gammas:
            Config.GAMMA_NUM = float(gamma)
            runs = []
            for run_idx in range(2 if quick else 3):
                profile, _ = runner.degradation_profile(300, seed=seed + run_idx * 17)
                m, _ = runner.run_proposed(profile, seed + run_idx)
                runs.append(m["utilization"])
            util.append(float(np.mean(runs)))
            util_std.append(float(np.std(runs)))
    finally:
        Config.GAMMA_NUM = original

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.errorbar(gammas, util, yerr=util_std, fmt="o-", capsize=3)
    ax.set_xlabel(r"regularisation strength $\gamma$")
    ax.set_ylabel("residual capability utilisation")
    ax.set_title("Trade-off between safety and exploitation")
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
        constr_traj[name] = []
        for p in log.positions:
            ok = 0
            tot = 0
            for i in range(runner.n):
                for j in range(i + 1, runner.n):
                    e = np.linalg.norm(p[i] - p[j] - (runner.desired_positions[i] - runner.desired_positions[j]))
                    tot += 1
                    if e <= 2.0:
                        ok += 1
            constr_traj[name].append(ok / tot)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name, ys in cost_traj.items():
        ax.plot(log.diag_steps[: len(ys)], ys, "o-", label=name, lw=1.2)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("formation tracking cost")
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
    profile, _ = runner.degradation_profile(n_steps, seed=seed)
    edges_to_drop = [(0, 1 % runner.n), (0, 2 % runner.n)]
    w_seq = communication_loss_w_sequence(runner.W, 10, edges_to_drop=edges_to_drop)

    cost_curves: dict[str, list[float]] = {}

    _, log_p = runner.run_proposed(profile, seed, w_base_per_interval=w_seq)
    cost_curves["Proposed"] = [
        float(np.linalg.norm(p - runner.desired_positions) ** 2) for p in log_p.positions
    ]
    _, log_r = runner.run_robust_do(profile, seed)
    cost_curves["Robust DO"] = [
        float(np.linalg.norm(p - runner.desired_positions) ** 2) for p in log_r.positions
    ]
    _, log_f = runner.run_fdi(profile, seed)
    cost_curves["FDI-Reconf"] = [
        float(np.linalg.norm(p - runner.desired_positions) ** 2) for p in log_f.positions
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name, ys in cost_curves.items():
        ax.plot(log_p.diag_steps[: len(ys)], ys, "o-", label=name, lw=1.2)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("formation tracking cost")
    ax.set_title("Scenario 2: communication-link degradation")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_scenario2_cost.pdf")

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for name, ys in cost_curves.items():
        ys = np.asarray(ys, dtype=float)
        # rolling std across diagnostic windows
        window = 3
        rolling = np.array(
            [np.std(ys[max(0, i - window): i + 1]) for i in range(len(ys))]
        )
        ax.plot(log_p.diag_steps[: len(rolling)], rolling, "o-", label=name, lw=1.2)
    ax.set_xlabel("simulation step")
    ax.set_ylabel(r"rolling std of cost (window 3)")
    ax.set_title("Scenario 2: variance under intermittent loss")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_scenario2_variance.pdf")


# ----------------------------------------------------- 10. topology
def fig_topology(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    modes = ["random", "high_weight", "adjacent"]
    counts = list(range(0, 5)) if not quick else list(range(0, 4))
    n_steps = 300

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for mode in modes:
        ys: list[float] = []
        for n_rem in counts:
            n_run = 2 if quick else 3
            vals: list[float] = []
            for run_idx in range(n_run):
                profile, _ = runner.degradation_profile(n_steps, seed=seed + run_idx * 17)
                if n_rem == 0:
                    w_seq = None
                else:
                    w_seq = perturb_topology(
                        runner.W, 10, mode=mode, n_removals=n_rem,
                        rng=np.random.default_rng(seed + run_idx),
                    )
                m, _ = runner.run_proposed(profile, seed + run_idx, w_base_per_interval=w_seq)
                vals.append(m["constraint_rate"])
            ys.append(float(np.mean(vals)))
        ax.plot(counts, ys, "o-", label=mode, lw=1.2)
    ax.axhline(0.94, color="gray", ls="--", lw=0.8, label="threshold 0.94")
    ax.set_xlabel("edges removed")
    ax.set_ylabel("constraint satisfaction rate")
    ax.set_title("Topology robustness")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_topology_robustness.pdf")


# ----------------------------------------------------- 11. multi-fault
def fig_multi_fault(out_dir: Path, runner: ExperimentRunner, seed: int, quick: bool):
    n_steps = 600 if not quick else 200
    profile, _ = concurrent_degradation_profile(runner.n, n_steps, onset_time=80, rng=np.random.default_rng(seed))
    _, log = runner.run_proposed(profile, seed)
    h_true = np.stack(log.true_health)
    h_est = np.stack(log.health_est)
    ts = log.diag_steps

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(ts, h_true[:, 0], "k-", lw=1.2, label="true (fast)")
    ax.plot(ts, h_est[:, 0], "C0o-", lw=1.0, label="est. (fast)")
    ax.plot(ts, h_true[:, 1], "k--", lw=1.2, label="true (slow)")
    ax.plot(ts, h_est[:, 1], "C1s-", lw=1.0, label="est. (slow)")
    ax.set_xlabel("simulation step")
    ax.set_ylabel("health degree")
    ax.set_title("Concurrent degradation: estimate vs ground truth")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out_dir, "fig_multi_fault.pdf")


# ----------------------------------------------------- 12. scalability
def fig_scalability(out_dir: Path, seed: int, quick: bool):
    sizes = [5, 8, 12, 16, 20, 30] if not quick else [5, 8, 12]
    walls: list[float] = []
    walls_std: list[float] = []
    for size in sizes:
        sub = ExperimentRunner(n_satellites=size, seed=seed)
        ws = []
        for run_idx in range(2 if quick else 3):
            profile, _ = sub.degradation_profile(200, seed=seed + run_idx * 17)
            m, _ = sub.run_proposed(profile, seed + run_idx)
            ws.append(m["wall_time"])
        walls.append(float(np.mean(ws)))
        walls_std.append(float(np.std(ws)))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.errorbar(sizes, walls, yerr=walls_std, fmt="o-", capsize=3, label="truncated PES (L=3)")
    ax.set_xlabel(r"formation size $N$")
    ax.set_ylabel("wall time per diagnosis interval (s)")
    ax.set_title("Scalability of truncated RPS pipeline")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_scalability.pdf")


# ----------------------------------------------------- 13. high-fidelity track
def fig_high_fidelity(out_dir: Path, seed: int, quick: bool):
    """Diagnostic accuracy on the NASA 42 stand-in (3-sat GTO)."""
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
    fig_scalability(out, seed, quick)
    fig_high_fidelity(out, seed, quick)

    print("done.")


if __name__ == "__main__":
    main()
