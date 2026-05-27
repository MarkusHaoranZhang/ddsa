"""Command line entry point for the experiment suite."""

from __future__ import annotations

import argparse

from dds_adapt.config import Config
from dds_adapt.runner import ExperimentRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dds-run",
        description="Diagnosis-Driven Structural Adaptation experiment suite.",
    )
    parser.add_argument(
        "--experiments",
        choices=[
            "comparative",
            "ablation",
            "scenario2",
            "topology",
            "concurrent",
            "scale",
            "learning",
            "hf",
            "rho-cal",
            "all",
        ],
        default="comparative",
    )
    parser.add_argument("--n-runs", type=int, default=Config.N_RUNS)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-satellites", type=int, default=Config.NUM_SATELLITES)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    Config.N_RUNS = args.n_runs

    print("=" * 80)
    print(" Diagnosis-Driven Structural Adaptation - Experiment Suite")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  N satellites:     {args.n_satellites}")
    print(f"  N runs:           {args.n_runs}")
    print(f"  Trajectory steps: {args.n_steps}")
    print(f"  Seed:             {args.seed}")
    runner = ExperimentRunner(
        n_satellites=args.n_satellites, track="numerical",
        seed=args.seed, verbose=True,
    )
    print(f"  lambda_2(W):      {runner.lambda2:.4f}")

    choice = args.experiments
    if choice in ("comparative", "all"):
        comp = runner.run_comparative(n_runs=args.n_runs, n_steps=args.n_steps)
        runner.print_results_table(comp, "Comparative study (Scenario 1)")

    if choice in ("ablation", "all"):
        abl = runner.run_ablation(n_runs=args.n_runs, n_steps=args.n_steps)
        runner.print_results_table(abl, "Ablation study")

    if choice in ("scenario2", "all"):
        s2 = runner.run_communication_scenario(
            n_runs=args.n_runs, n_steps=args.n_steps
        )
        runner.print_results_table(s2, "Scenario 2: communication degradation")

    if choice in ("topology", "all"):
        topo = runner.run_topology_robustness(
            n_runs=max(3, args.n_runs // 5), n_steps=args.n_steps
        )
        runner.print_results_table(topo, "Extended: topology robustness")

    if choice in ("concurrent", "all"):
        conc = runner.run_concurrent_degradation(
            n_runs=args.n_runs, n_steps=args.n_steps
        )
        runner.print_results_table(conc, "Extended: concurrent degradation")

    if choice in ("scale", "all"):
        scale = ExperimentRunner.run_scalability(
            sizes=[5, 10, 20, 30], n_steps=200, n_runs=2
        )
        print("\n" + "=" * 80)
        print(" Extended: scalability sweep")
        print("=" * 80)
        for size, metrics in scale.items():
            print(f"\nN = {size}:")
            for k, v in metrics.items():
                if isinstance(v, tuple):
                    print(f"  {k}: {v[0]:.4f} ± {v[1]:.4f}")

    if choice in ("learning", "all"):
        try:
            from dds_adapt.learning_baseline import evaluate_learning_baseline
            res = evaluate_learning_baseline(
                n_agents=args.n_satellites, n_test=300, seed=args.seed
            )
            runner.print_results_table(res, "Extended: learning baseline")
        except ImportError as exc:
            print(f"\nSkipping learning baseline: {exc}")

    if choice in ("hf", "all"):
        from dds_adapt.hf_runner import run_hf_diagnostic_experiment
        hf = run_hf_diagnostic_experiment(seed=args.seed)
        print("\n" + "=" * 80)
        print(" High-fidelity track (NASA 42 stand-in, 3 satellites GTO)")
        print("=" * 80)
        print(f"  health_mae:        {hf.health_mae:.4f}")
        print(f"  kendall_tau:       {hf.kendall_tau:.4f}")
        print(f"  detection_delay:   {hf.detection_delay:.1f} steps")
        print(f"  per-agent estimate: {hf.mean_estimate_per_agent}")
        print(f"  per-agent truth:    {hf.true_per_agent}")

    if choice in ("rho-cal", "all"):
        from dds_adapt.rho_max_calibration import calibrate_rho_max
        cal = calibrate_rho_max(runner, seed=args.seed)
        print("\n" + "=" * 80)
        print(" rho_max calibration (Eq. 12)")
        print("=" * 80)
        print(f"  rho_star (empirical, threshold = 1.5x baseline):  {cal.rho_star_empirical:.5f}")
        print(f"  rho_max (theoretical, Eq. 12 with C_prior):       {cal.rho_max_theoretical:.5f}")
        print(f"  ratio rho_star / rho_max:                         {cal.rho_star_empirical / max(cal.rho_max_theoretical, 1e-12):.2f}")
        print(f"  back-solved C from empirical rho_star:            {cal.constant_C:.4f}")
        print(
            "\n  Note: the absolute ratio depends heavily on the prior C used"
            "\n  for rho_max. We report the back-solved C so a reader can plug"
            "\n  it into Eq. (12) for their own design constraints."
        )
        for rho, err, div in zip(cal.rhos, cal.steady_state_errors, cal.diverged, strict=True):
            tag = ">1.5x baseline" if div else "ok"
            print(f"    rho = {rho:.5f}  tracking err = {err:.4e}  {tag}")

    print("\n" + "=" * 80)
    print(" DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
