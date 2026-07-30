"""Structural guarantees about the harness itself.

These are cheap and they catch the drift that makes a harness quietly stop
covering what it claims to cover — a scenario silently losing its blast-radius
assertion, or a verdict rollup that reports PASS because one check errored.
"""

from __future__ import annotations

import json

import pytest

from conformance.adversarial import ALL_SCENARIOS, BY_NAME
from conformance.contract import (
    ConformanceReport,
    Observation,
    Route,
    ScenarioResult,
    Verdict,
)
from conformance.platform_api import pluck
from conformance.report import render_html, render_text
from conformance.routes import build_notebook
from conformance.workloads import run_command
from conformance.workloads.base import MANIFEST_MARKER, PROGRESS_MARKER, WorkloadManifest

REQUESTED_SCENARIOS = {
    "cuda_oom",
    "host_ram_exhaustion",
    "hung_network_read",
    "kill_at_60_resume_from_checkpoint",
    "disk_exhaustion",
}


def test_all_five_requested_scenarios_exist() -> None:
    assert set(BY_NAME) == REQUESTED_SCENARIOS


def test_every_scenario_declares_at_least_three_expectations() -> None:
    for scenario in ALL_SCENARIOS:
        assert len(scenario.expectations) >= 3, scenario.name


def test_expectation_keys_are_unique_within_a_scenario() -> None:
    for scenario in ALL_SCENARIOS:
        keys = [e.key for e in scenario.expectations]
        assert len(keys) == len(set(keys)), f"{scenario.name} has duplicate expectation keys"


def test_every_expectation_cites_the_spec() -> None:
    """A finding must be arguable against the document, not against my opinion."""
    for scenario in ALL_SCENARIOS:
        for exp in scenario.expectations:
            assert exp.spec_ref.strip(), f"{scenario.name}.{exp.key} has no spec_ref"


def test_node_level_faults_carry_a_blast_radius_assertion() -> None:
    """RAM and disk faults threaten neighbours; that must never be dropped."""
    for name in ("host_ram_exhaustion", "disk_exhaustion"):
        scenario = BY_NAME[name]
        assert scenario.needs_victim, f"{name} must run a victim job"
        assert any(e.blast_radius for e in scenario.expectations), name


def test_cuda_oom_does_not_waste_a_victim_instance() -> None:
    """A GPU OOM is container-scoped; a second billable instance would prove nothing."""
    assert BY_NAME["cuda_oom"].needs_victim is False


def test_scenario_commands_are_non_empty() -> None:
    for scenario in ALL_SCENARIOS:
        assert scenario.run_command.strip()


def test_unknown_expectation_key_raises() -> None:
    with pytest.raises(KeyError):
        BY_NAME["cuda_oom"].expectation("nope")


# --- verdict rollup ---------------------------------------------------------


def _result(*verdicts: Verdict) -> ScenarioResult:
    return ScenarioResult(
        scenario="s",
        route=Route.GIT_REPO,
        instance_id="i",
        workload_run_id=None,
        started_at_unix=0.0,
        duration_seconds=1.0,
        observations=[Observation(f"k{i}", v, "d") for i, v in enumerate(verdicts)],
    )


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        ((Verdict.PASS, Verdict.PASS), Verdict.PASS),
        ((Verdict.PASS, Verdict.UNOBSERVABLE), Verdict.UNOBSERVABLE),
        ((Verdict.PASS, Verdict.UNOBSERVABLE, Verdict.FAIL), Verdict.FAIL),
        ((Verdict.FAIL, Verdict.ERROR), Verdict.ERROR),
        ((), Verdict.SKIPPED),
    ],
)
def test_rollup_takes_the_worst_verdict(verdicts: tuple, expected: Verdict) -> None:
    assert _result(*verdicts).verdict is expected


def test_unobservable_never_silently_becomes_a_pass() -> None:
    """The whole point of the three-way verdict."""
    assert _result(Verdict.PASS, Verdict.UNOBSERVABLE).verdict is not Verdict.PASS


def test_report_separates_failures_from_unobservable() -> None:
    report = ConformanceReport(results=[_result(Verdict.FAIL), _result(Verdict.UNOBSERVABLE)])
    assert len(report.failures()) == 1
    assert len(report.unobservable()) == 1
    assert report.counts()["fail"] == 1
    assert report.counts()["unobservable"] == 1


def test_reports_render_without_a_platform() -> None:
    report = ConformanceReport(
        results=[_result(Verdict.FAIL), _result(Verdict.PASS)],
        platform_base_url="https://example.test",
    )
    text = render_text(report)
    assert "fail=1" in text
    html = render_html(report)
    assert "<!doctype html>" in html
    assert "unobservable" in html
    json.loads(json.dumps(report.to_dict()))  # must be JSON-serialisable


# --- field plucking ---------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expect_value", "expect_observable"),
    [
        ({"exit_code": 137}, 137, True),
        ({"exitcode": 137}, 137, True),
        ({"exit_code": None}, None, True),  # present but null: observable, and a defect
        ({"other": 1}, None, False),  # absent entirely: unobservable
        (None, None, False),
        ("not a dict", None, False),
    ],
)
def test_pluck_distinguishes_absent_from_null(payload, expect_value, expect_observable) -> None:
    found = pluck(payload, ("exit_code", "exitcode"))
    assert found.value == expect_value
    assert found.observable is expect_observable


def test_pluck_prefers_the_first_candidate() -> None:
    assert pluck({"exitcode": 2, "exit_code": 1}, ("exit_code", "exitcode")).value == 1


# --- workload contract ------------------------------------------------------


def test_manifest_marker_is_last_line(tmp_path, capsys) -> None:
    """Routes that can only scrape stdout depend on this."""
    WorkloadManifest(workload="w", started_at_unix=0.0).write(tmp_path)
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1].startswith(MANIFEST_MARKER)
    assert json.loads((tmp_path / "manifest.json").read_text())["workload"] == "w"


def test_manifest_is_json_serialisable_with_all_fields() -> None:
    manifest = WorkloadManifest(
        workload="lora_intent_json",
        started_at_unix=1.0,
        resumed_from_step=30,
        completed_steps=50,
        total_steps=50,
        metrics={"tuned": {"accuracy": 0.9}},
    )
    payload = json.loads(json.dumps(manifest.to_dict()))
    assert payload["resumed_from_step"] == 30


def test_run_command_covers_all_three_workloads() -> None:
    for workload in ("lora", "whisper", "faiss"):
        command = run_command(workload, canary=True)
        assert "--canary" in command
        assert workload in command or "embed_faiss" in command


def test_notebook_is_valid_ipynb_and_fails_on_nonzero_exit() -> None:
    """papermill reports success for a crashed subprocess unless the cell asserts."""
    notebook = json.loads(build_notebook("lora", canary=True))
    assert notebook["nbformat"] == 4
    source = "".join(notebook["cells"][0]["source"])
    assert "returncode == 0" in source, "notebook must fail when the workload fails"


def test_progress_marker_is_parseable(capsys) -> None:
    from conformance.workloads.base import emit_progress

    emit_progress(30, 50, epoch=0.6)
    line = capsys.readouterr().out.strip()
    assert line.startswith(PROGRESS_MARKER)
    payload = json.loads(line.split(PROGRESS_MARKER, 1)[1])
    assert payload["pct"] == 60.0  # the kill trigger depends on this exact field
