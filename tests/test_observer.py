"""The observer's verdict logic is the harness's own correctness surface.

If it reports FAIL when the platform was fine, engineers chase ghosts. If it
reports PASS when a field was simply missing, a real gap ships. So the three-way
distinction gets tested explicitly, in both directions.
"""

from __future__ import annotations

import pytest
from fake_platform import FakePlatformClient

from conformance.contract import Verdict
from conformance.observer import ENOSPC_PATTERNS, OOM_PATTERNS, Observer
from conformance.platform_api import F_RAM_USED, PlatformError


def obs(**kwargs) -> Observer:
    return Observer(FakePlatformClient(**kwargs))  # type: ignore[arg-type]


# --- terminal status --------------------------------------------------------


def test_missing_run_is_unobservable_not_fail() -> None:
    result = obs().run_reaches_terminal_status(None)
    assert result.verdict is Verdict.UNOBSERVABLE


def test_run_stuck_running_is_a_real_failure() -> None:
    result = obs().run_reaches_terminal_status({"status": "running"})
    assert result.verdict is Verdict.FAIL
    assert "stuck" in result.detail


def test_failed_run_passes_the_terminal_check() -> None:
    assert obs().run_reaches_terminal_status({"status": "failed"}).verdict is Verdict.PASS


def test_status_field_absent_is_unobservable() -> None:
    result = obs().run_reaches_terminal_status({"exit_code": 1})
    assert result.verdict is Verdict.UNOBSERVABLE


# --- exit codes -------------------------------------------------------------


def test_absent_exit_code_field_is_unobservable() -> None:
    assert obs().exit_code_surfaced({"status": "failed"}).verdict is Verdict.UNOBSERVABLE


def test_null_exit_code_on_finished_run_is_a_failure() -> None:
    """Present-but-null is different from absent, and it is a real defect."""
    result = obs().exit_code_surfaced({"status": "failed", "exit_code": None})
    assert result.verdict is Verdict.FAIL
    assert "null" in result.detail


def test_zero_exit_code_for_a_fault_scenario_is_a_failure() -> None:
    assert obs().exit_code_surfaced({"exit_code": 0}).verdict is Verdict.FAIL


def test_unexpected_nonzero_code_still_passes() -> None:
    result = obs().exit_code_surfaced({"exit_code": 1}, expect=137)
    assert result.verdict is Verdict.PASS


# --- message legibility -----------------------------------------------------


def test_oom_pattern_matches_real_torch_message() -> None:
    run = {
        "stderr": "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB"
    }
    result = obs().message_matches("k", run, OOM_PATTERNS, "OOM")
    assert result.verdict is Verdict.PASS


def test_generic_failure_message_is_illegible_and_fails() -> None:
    result = obs().message_matches("k", {"stderr": "Command failed"}, OOM_PATTERNS, "OOM")
    assert result.verdict is Verdict.FAIL


def test_no_output_at_all_is_unobservable() -> None:
    result = obs().message_matches("k", {"status": "failed"}, OOM_PATTERNS, "OOM")
    assert result.verdict is Verdict.UNOBSERVABLE


def test_enospc_pattern_matches_python_oserror() -> None:
    run = {"stderr": "OSError: [Errno 28] No space left on device"}
    assert obs().message_matches("k", run, ENOSPC_PATTERNS, "disk").verdict is Verdict.PASS


# --- instance state ---------------------------------------------------------


def test_instance_left_running_after_fault_is_a_failure() -> None:
    result = obs(instance_status="running").instance_not_left_nominally_active("i-1")
    assert result.verdict is Verdict.FAIL
    assert "nominally-active" in result.detail


def test_terminated_instance_passes() -> None:
    result = obs(instance_status="terminated").instance_not_left_nominally_active("i-1")
    assert result.verdict is Verdict.PASS


def test_status_read_error_is_error_not_fail() -> None:
    result = obs(status_raises=PlatformError("boom")).instance_not_left_nominally_active("i-1")
    assert result.verdict is Verdict.ERROR


def test_runtime_failed_billing_is_unobservable_via_api() -> None:
    """Correct behaviour we cannot confirm without the provider's own API."""
    result = obs(instance_status="runtime_failed").billing_stopped("i-1")
    assert result.verdict is Verdict.UNOBSERVABLE


def test_terminated_confirms_billing_stopped() -> None:
    assert obs(instance_status="terminated").billing_stopped("i-1").verdict is Verdict.PASS


def test_still_running_means_billing_not_stopped() -> None:
    assert obs(instance_status="running").billing_stopped("i-1").verdict is Verdict.FAIL


# --- metrics ----------------------------------------------------------------


def test_no_metrics_rows_is_unobservable() -> None:
    result = obs(metrics=[]).metrics_kept_reporting("i-1", F_RAM_USED, "ram")
    assert result.verdict is Verdict.UNOBSERVABLE


def test_metrics_that_stopped_mid_incident_is_a_failure() -> None:
    rows = [{"ram_used_mb": 1024}, {"gpu_utilization_pct_avg": 0}, {"gpu_utilization_pct_avg": 0}]
    result = obs(metrics=rows).metrics_kept_reporting("i-1", F_RAM_USED, "ram")
    assert result.verdict is Verdict.FAIL
    assert "died during the incident" in result.detail


def test_disk_ramp_detected() -> None:
    rows = [
        {"disk_used_gb": 5, "disk_total_gb": 100},
        {"disk_used_gb": 60, "disk_total_gb": 100},
        {"disk_used_gb": 99, "disk_total_gb": 100},
    ]
    assert obs(metrics=rows).disk_metrics_showed_ramp("i-1").verdict is Verdict.PASS


def test_flat_disk_metric_during_a_fill_is_a_failure() -> None:
    rows = [{"disk_used_gb": 5, "disk_total_gb": 100}, {"disk_used_gb": 6, "disk_total_gb": 100}]
    assert obs(metrics=rows).disk_metrics_showed_ramp("i-1").verdict is Verdict.FAIL


@pytest.mark.parametrize(
    ("utils", "expected"),
    [
        ([0.0, 0.0, 0.1], Verdict.FAIL),
        ([0.0, 96.0], Verdict.PASS),
        ([], Verdict.UNOBSERVABLE),
    ],
)
def test_idle_gpu_detection(utils: list[float], expected: Verdict) -> None:
    rows = [{"gpu_utilization_pct_avg": u} for u in utils]
    assert obs(metrics=rows).idle_gpu_detected("i-1").verdict is expected


def test_ssh_observability_lost_is_a_failure() -> None:
    result = obs(latest_metrics=None).platform_retained_ssh_observability("i-1")
    assert result.verdict is Verdict.FAIL


def test_ssh_observability_retained_passes() -> None:
    result = obs(latest_metrics={"ram_used_mb": 900}).platform_retained_ssh_observability("i-1")
    assert result.verdict is Verdict.PASS


# --- terminate / blast radius ----------------------------------------------


def test_terminate_rejected_during_fault_is_a_failure() -> None:
    result = obs(terminate_raises=PlatformError("409")).terminate_still_works("i-1")
    assert result.verdict is Verdict.FAIL
    assert "bills until" in result.detail


def test_terminate_succeeds_passes() -> None:
    assert obs(instance_status="running").terminate_still_works("i-1").verdict is Verdict.PASS


def test_victim_death_is_an_isolation_failure() -> None:
    result = obs().victim_survived({"status": "failed", "exit_code": 137})
    assert result.verdict is Verdict.FAIL
    assert "isolation failure" in result.detail


def test_victim_survival_passes() -> None:
    assert obs().victim_survived({"status": "succeeded", "exit_code": 0}).verdict is Verdict.PASS


def test_absent_victim_is_unobservable() -> None:
    assert obs().victim_survived(None).verdict is Verdict.UNOBSERVABLE
