"""Harness entry point. Runs OUTSIDE the sandbox, against the platform API."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from .adversarial import ALL_SCENARIOS, BY_NAME
from .contract import Verdict
from .orchestrator import Orchestrator, RunOptions
from .platform_api import PlatformClient
from .report import render_text, write_reports

app = typer.Typer(add_completion=False, help="AI Sandbox platform conformance harness")


@app.command("list")
def list_scenarios() -> None:
    """Show every scenario and what it asserts."""
    for scenario in ALL_SCENARIOS:
        gpu = "gpu" if scenario.needs_gpu else "cpu"
        victim = ", needs victim job" if scenario.needs_victim else ""
        typer.echo(f"\n{scenario.name}  [{gpu}{victim}]")
        typer.echo(f"  {scenario.summary}")
        for exp in scenario.expectations:
            typer.echo(f"    - {exp.key}  ({exp.spec_ref})")


@app.command()
def run(
    base_url: str = typer.Option(..., envvar="SANDBOX_API_URL", help="Platform API base URL"),
    email: str = typer.Option(None, envvar="SANDBOX_EMAIL"),
    password: str = typer.Option(None, envvar="SANDBOX_PASSWORD"),
    token: str = typer.Option(None, envvar="SANDBOX_TOKEN"),
    repo_url: str = typer.Option(..., envvar="CONFORMANCE_REPO", help="Git URL of THIS repo"),
    repo_ref: str = typer.Option("main"),
    template: str = typer.Option("pytorch", help="Template to launch sandboxes with"),
    gpu: str = typer.Option(None, help="Explicit GPU type, if the template requires one"),
    output_dir: str = typer.Option("reports", help="Where to write the scorecard"),
    hang_budget: float = typer.Option(
        1800.0, help="Seconds to wait on the hung-network scenario before calling it a failure"
    ),
    only: list[str] = typer.Option(None, "--only", help="Run only these adversarial scenarios"),
    skip_adversarial: bool = typer.Option(False, help="Routes and workloads only"),
    skip_workloads: bool = typer.Option(False, help="Routes and adversarial only"),
    dry_run: bool = typer.Option(False, help="Plan the matrix without launching anything"),
) -> None:
    """Run the conformance matrix and write a scorecard."""
    unknown = [name for name in (only or []) if name not in BY_NAME]
    if unknown:
        raise typer.BadParameter(f"unknown scenario(s): {unknown}. Try `conformance list`.")

    _warn_about_live_credentials()

    options = RunOptions(
        repo_url=repo_url,
        repo_ref=repo_ref,
        template=template,
        gpu=gpu,
        hang_budget_seconds=hang_budget,
        dry_run=dry_run,
        skip_adversarial=skip_adversarial,
        skip_workloads=skip_workloads,
        scenarios=tuple(only or ()),
    )

    with PlatformClient(base_url, email=email, password=password, token=token) as client:
        report = Orchestrator(client, options).run_all()

    paths = write_reports(report, Path(output_dir))
    typer.echo(render_text(report))
    typer.echo(f"\nscorecard: {paths['html']}")

    # Non-zero exit on any real failure, so CI can gate on this.
    if any(r.verdict in (Verdict.FAIL, Verdict.ERROR) for r in report.results):
        raise typer.Exit(1)


@app.command()
def preflight(
    base_url: str = typer.Option(..., envvar="SANDBOX_API_URL"),
    token: str = typer.Option(None, envvar="SANDBOX_TOKEN"),
) -> None:
    """Check the platform is reachable and the harness can authenticate."""
    with PlatformClient(base_url, token=token) as client:
        typer.echo(f"healthz: {client.healthz()}")
        try:
            templates = client.templates()
            count = len(templates) if isinstance(templates, list) else "unknown"
            typer.echo(f"templates: {count}")
        except Exception as exc:
            typer.echo(f"templates: unavailable ({exc})")


def _warn_about_live_credentials() -> None:
    """Refuse to run silently in the build environment described in spec section 30.

    Master Spec v27 documents five real, billable RunPod pods created accidentally
    in one session, because `celery_supervisor.py` auto-starts live Celery workers
    whenever the shared dev Postgres holds a non-terminal SandboxInstance row.
    This harness deliberately creates failing instances, which is exactly the
    trigger condition. Running it in that environment is how you get a surprise
    invoice, so it warns loudly and requires an explicit acknowledgement.
    """
    if os.environ.get("CONFORMANCE_ACK_LIVE_CREDENTIALS") == "yes":
        return
    typer.secho(
        "\n  WARNING — read before running against a live-credentialed environment.\n"
        "  This harness deliberately creates instances that fail, hang, and fill disks.\n"
        "  Master Spec section 30 records five accidental billable pods caused by a dev\n"
        "  script that auto-starts real Celery workers whenever a non-terminal\n"
        "  SandboxInstance row exists — which this harness creates constantly.\n\n"
        "  Before running: stop any Celery worker/beat/supervisor process, and confirm\n"
        "  you intend to spend real GPU money.\n\n"
        "  Then set CONFORMANCE_ACK_LIVE_CREDENTIALS=yes to proceed.\n",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(2)


if __name__ == "__main__":
    app()
