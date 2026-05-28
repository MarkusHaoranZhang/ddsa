"""End-to-end reproduction driver.

Runs every study reported in the paper, dumps raw metrics to JSON, and
emits all 13 figures. Designed so a reviewer can do:

    pip install -e .[dev,learning,plot]
    python scripts/reproduce.py

and walk away with a ``results/`` directory and a ``figures/`` directory
that, together with the git commit hash recorded in ``results/meta.json``,
fully characterise the run.

Use ``--quick`` for a sub-minute smoke run before the full sweep.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dds_adapt.config import Config
from dds_adapt.runner import ExperimentRunner


# ----------------------------------------------------------- helpers
def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:  # noqa: BLE001 - reproduce.py must never fail here
        return "unknown"


def _python_packages() -> dict[str, str]:
    pkgs = {}
    # PyPI distribution name -> import module name. They differ for
    # scikit-learn (PyPI: ``scikit-learn``, import: ``sklearn``); the
    # earlier ``name.replace("-", "_")`` shortcut silently misreported
    # sklearn as "not installed" in meta.json.
    name_map = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
    }
    for dist_name, import_name in name_map.items():
        try:
            mod = __import__(import_name)
            pkgs[dist_name] = getattr(mod, "__version__", "?")
        except ImportError:
            pkgs[dist_name] = "not installed"
    return pkgs


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"  wrote {path}")


# ----------------------------------------------------------- driver
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=Config.N_RUNS)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--quick", action="store_true",
                        help="small sweep, suitable for a smoke check")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-learning", action="store_true",
                        help="skip the §5.5.4 learning baseline (no sklearn)")
    args = parser.parse_args()

    if args.quick:
        n_runs = max(2, min(args.n_runs, 3))
        n_steps = min(args.n_steps, 200)
    else:
        n_runs = args.n_runs
        n_steps = args.n_steps

    runner = ExperimentRunner(
        n_satellites=Config.NUM_SATELLITES, track="numerical",
        seed=args.seed, verbose=True,
    )

    out_dir = args.results_dir / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reproducing into {out_dir.resolve()}")

    # --------- meta -----------
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _python_packages(),
        "config": {
            "n_satellites": Config.NUM_SATELLITES,
            "n_runs": n_runs,
            "n_steps": n_steps,
            "seed": args.seed,
            "lambda2": runner.lambda2,
            "alpha": runner.alpha,
            "gamma": runner.gamma,
            "beta": runner.beta,
        },
    }
    _dump(out_dir / "meta.json", meta)

    # --------- studies -----------
    print("\n[1/6] Comparative study (Scenario 1)")
    comp = runner.run_comparative(n_runs=n_runs, n_steps=n_steps)
    _dump(out_dir / "comparative.json", comp)
    runner.print_results_table(comp, "Comparative")

    print("\n[2/6] Ablation study")
    abl = runner.run_ablation(n_runs=n_runs, n_steps=n_steps)
    _dump(out_dir / "ablation.json", abl)
    runner.print_results_table(abl, "Ablation")

    print("\n[3/6] Scenario 2: communication degradation")
    s2 = runner.run_communication_scenario(n_runs=n_runs, n_steps=n_steps)
    _dump(out_dir / "scenario2.json", s2)
    runner.print_results_table(s2, "Scenario 2")

    print("\n[4/6] Topology robustness")
    topo = runner.run_topology_robustness(
        n_runs=max(3, n_runs // 5), n_steps=n_steps
    )
    _dump(out_dir / "topology.json", topo)
    runner.print_results_table(topo, "Topology robustness")

    print("\n[5/6] Concurrent degradation")
    conc = runner.run_concurrent_degradation(n_runs=n_runs, n_steps=n_steps)
    _dump(out_dir / "concurrent.json", conc)
    runner.print_results_table(conc, "Concurrent")

    print("\n[6/6] Scalability sweep")
    sizes = [5, 8, 12, 16] if args.quick else [5, 8, 12, 16, 20, 30]
    scale = ExperimentRunner.run_scalability(
        sizes=sizes, n_steps=200, n_runs=2 if args.quick else 3
    )
    _dump(out_dir / "scale.json", scale)

    if not args.skip_learning:
        print("\n[+] Learning baseline (optional)")
        try:
            from dds_adapt.learning_baseline import evaluate_learning_baseline

            learn = evaluate_learning_baseline(
                n_agents=Config.NUM_SATELLITES,
                n_test=200 if args.quick else 500,
                seed=args.seed,
            )
            _dump(out_dir / "learning.json", learn)
        except ImportError as exc:
            print(f"  scikit-learn missing, skipping learning study: {exc}")

    print("\n[+] High-fidelity track (NASA 42 stand-in)")
    from dds_adapt.hf_runner import run_hf_diagnostic_experiment
    hf = run_hf_diagnostic_experiment(
        n_steps=200 if args.quick else 600, seed=args.seed
    )
    _dump(
        out_dir / "high_fidelity.json",
        {
            "health_mae": hf.health_mae,
            "kendall_tau": hf.kendall_tau,
            "detection_delay": hf.detection_delay,
            "mean_estimate_per_agent": hf.mean_estimate_per_agent.tolist(),
            "true_per_agent": hf.true_per_agent.tolist(),
        },
    )

    print("\n[+] rho_max calibration (Eq. 12)")
    from dds_adapt.rho_max_calibration import calibrate_rho_max
    cal = calibrate_rho_max(
        runner,
        seed=args.seed,
        n_steps=200 if args.quick else 400,
        n_repeats=1 if args.quick else 2,
    )
    _dump(
        out_dir / "rho_calibration.json",
        {
            "rhos": cal.rhos.tolist(),
            "steady_state_errors": cal.steady_state_errors.tolist(),
            "diverged": cal.diverged.tolist(),
            "rho_star_empirical": cal.rho_star_empirical,
            "rho_max_theoretical": cal.rho_max_theoretical,
            "margin_percent": cal.margin_percent,
            "constant_C": cal.constant_C,
        },
    )

    if not args.skip_figures:
        print("\n[+] Figures")
        try:
            # ``scripts`` may not be on the path when run as a script;
            # add the parent of this file so ``make_figures`` is importable.
            this_dir = Path(__file__).resolve().parent
            if str(this_dir) not in sys.path:
                sys.path.insert(0, str(this_dir))
            from make_figures import (  # type: ignore[import-not-found]
                fig_architecture,
                fig_convergence,
                fig_gamma_sensitivity,
                fig_health_sensitivity,
                fig_high_fidelity,
                fig_multi_fault,
                fig_scalability,
                fig_scenario1,
                fig_scenario2,
                fig_topology,
            )

            args.figures_dir.mkdir(parents=True, exist_ok=True)
            fig_architecture(args.figures_dir)
            fig_convergence(args.figures_dir, runner, args.seed, args.quick)
            fig_health_sensitivity(args.figures_dir, runner, args.seed, args.quick)
            fig_gamma_sensitivity(args.figures_dir, runner, args.seed, args.quick)
            fig_scenario1(args.figures_dir, runner, args.seed, args.quick)
            fig_scenario2(args.figures_dir, runner, args.seed, args.quick)
            fig_topology(args.figures_dir, runner, args.seed, args.quick)
            fig_multi_fault(args.figures_dir, runner, args.seed, args.quick)
            fig_scalability(args.figures_dir, args.seed, args.quick)
            fig_high_fidelity(args.figures_dir, args.seed, args.quick)
        except ImportError as exc:
            print(f"  matplotlib missing, skipping figures: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
