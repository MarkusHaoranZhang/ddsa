"""CLI smoke tests for ``dds_adapt.cli.main``.

These tests call ``main`` in-process (faster, avoids subprocess
overhead and Windows console encoding gotchas) and verify that:

* every ``--experiments`` choice dispatches without raising;
* the parser's ``choices=`` constraint is enforced;
* the parser surfaces invalid integer arguments as a usage error.

We deliberately use very small ``--n-runs`` / ``--n-steps`` /
``--n-satellites`` so the suite finishes in a few seconds even when
the CLI dispatches into the full closed loop.
"""

from __future__ import annotations

import pytest

from dds_adapt.cli import main


# ----------------------------------------------------- happy paths
@pytest.mark.parametrize(
    "experiment",
    [
        "comparative",
        "ablation",
        "scenario2",
        "topology",
        "concurrent",
        "scale",
        "hf",
        "rho-cal",
    ],
)
def test_main_dispatches_each_experiment(experiment: str, capsys) -> None:
    """Every choice that does not require sklearn must run end-to-end."""
    main(
        [
            "--experiments", experiment,
            "--n-runs", "1",
            "--n-steps", "80",
            "--n-satellites", "4",
            "--seed", "0",
        ]
    )
    out = capsys.readouterr().out
    assert "DONE" in out


def test_main_default_runs_comparative(capsys) -> None:
    """Calling main with no choice falls back to the 'comparative' default."""
    main(
        [
            "--n-runs", "1",
            "--n-steps", "80",
            "--n-satellites", "4",
            "--seed", "0",
        ]
    )
    out = capsys.readouterr().out
    assert "Comparative study" in out
    assert "DONE" in out


# ----------------------------------------------------- error paths
def test_main_rejects_unknown_experiment() -> None:
    """argparse should refuse a value not in the ``choices=`` list."""
    with pytest.raises(SystemExit):
        main(["--experiments", "nonsense"])


def test_main_rejects_non_integer_n_runs() -> None:
    """Type coercion errors must surface as argparse usage errors."""
    with pytest.raises(SystemExit):
        main(["--n-runs", "not-a-number"])


def test_main_console_script_registered() -> None:
    """The ``dds-run`` console script in pyproject must resolve to ``main``."""
    from importlib.metadata import entry_points

    eps = entry_points(group="console_scripts")
    matching = [ep for ep in eps if ep.name == "dds-run"]
    assert matching, "dds-run console script not registered"
    assert matching[0].value == "dds_adapt.cli:main"
