"""Scorecard rendering.

The report deliberately gives UNOBSERVABLE its own column rather than folding it
into failures or passes. On the first run of a harness like this, the
unobservable count is usually the most actionable number on the page: it lists
exactly the questions your platform cannot currently answer about itself.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .contract import Verdict

if TYPE_CHECKING:
    from .contract import ConformanceReport

_COLORS = {
    "pass": "#0f7b4f",
    "fail": "#b3261e",
    "unobservable": "#8a6100",
    "skipped": "#5f6368",
    "error": "#7a1fa2",
}


def render_text(report: ConformanceReport) -> str:
    counts = report.counts()
    lines = [
        "AI Sandbox conformance report",
        f"platform: {report.platform_base_url}",
        f"duration: {report.duration_seconds:.0f}s",
        "",
        "  ".join(f"{k}={v}" for k, v in counts.items() if v),
        "",
    ]
    for result in report.results:
        lines.append(f"[{result.verdict.value.upper():>12}] {result.scenario}")
        for obs in result.observations:
            if obs.verdict in (Verdict.PASS, Verdict.SKIPPED):
                continue
            lines.append(f"    {obs.verdict.value:>12}  {obs.key}: {obs.detail}")
    if report.skipped:
        lines += ["", "Skipped: " + ", ".join(report.skipped)]

    unobservable = report.unobservable()
    if unobservable:
        lines += [
            "",
            "Questions this platform cannot currently answer about itself:",
            *(f"  - {r.scenario}" for r in unobservable),
        ]
    return "\n".join(lines)


def render_html(report: ConformanceReport) -> str:
    counts = report.counts()
    chips = "".join(
        f'<span class="chip" style="--c:{_COLORS.get(k, "#555")}">{html.escape(k)} '
        f"<b>{v}</b></span>"
        for k, v in counts.items()
        if v
    )

    rows = []
    for result in report.results:
        obs_rows = "".join(
            f'<tr class="obs"><td></td><td><code>{html.escape(o.key)}</code></td>'
            f'<td><span class="v" style="--c:{_COLORS.get(o.verdict.value, "#555")}">'
            f"{html.escape(o.verdict.value)}</span></td>"
            f"<td>{html.escape(o.detail)}</td></tr>"
            for o in result.observations
        )
        rows.append(
            f'<tr class="scenario"><td colspan="2"><b>{html.escape(result.scenario)}</b>'
            f'<div class="meta">route: {html.escape(result.route.value)} · '
            f"{result.duration_seconds:.0f}s</div></td>"
            f'<td><span class="v" style="--c:{_COLORS.get(result.verdict.value, "#555")}">'
            f"{html.escape(result.verdict.value)}</span></td><td></td></tr>{obs_rows}"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Sandbox conformance report</title>
<style>
  :root {{ color-scheme: light dark; --bg:#fff; --fg:#1a1c1e; --line:#e3e3e6; --muted:#5f6368; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#131416; --fg:#e6e1e5; --line:#2c2e30; --muted:#a5a5a8; }}
  }}
  body {{ margin:0; padding:2rem 1.25rem; background:var(--bg); color:var(--fg);
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
  main {{ max-width:1080px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
  .sub {{ color:var(--muted); margin-bottom:1.25rem; }}
  .chip {{ display:inline-block; padding:.25rem .7rem; margin:0 .4rem .4rem 0;
    border-radius:999px; background:color-mix(in srgb, var(--c) 14%, transparent);
    color:var(--c); font-size:.85rem; }}
  table {{ width:100%; border-collapse:collapse; margin-top:1.25rem; }}
  td {{ padding:.5rem .6rem; border-bottom:1px solid var(--line); vertical-align:top; }}
  tr.scenario td {{ background:color-mix(in srgb, var(--fg) 4%, transparent); }}
  tr.obs td {{ font-size:.88rem; color:var(--muted); }}
  tr.obs code {{ font-size:.85em; }}
  .v {{ color:var(--c); font-weight:600; font-size:.82rem; text-transform:uppercase;
    letter-spacing:.03em; white-space:nowrap; }}
  .meta {{ color:var(--muted); font-size:.8rem; font-weight:400; }}
  .note {{ margin-top:1.5rem; padding:.9rem 1rem; border-left:3px solid #8a6100;
    background:color-mix(in srgb, #8a6100 8%, transparent); font-size:.9rem; }}
</style></head>
<body><main>
  <h1>AI Sandbox conformance report</h1>
  <div class="sub">{html.escape(report.platform_base_url)} ·
    {report.duration_seconds:.0f}s · {len(report.results)} scenarios</div>
  <div>{chips}</div>
  <div class="note"><b>unobservable</b> is not a soft failure. It means the platform may
  have behaved correctly but exposes no way for anyone outside the sandbox to confirm it.
  On a multi-tenant GPU platform that is its own finding.</div>
  <table>{"".join(rows)}</table>
</main></body></html>"""


def write_reports(report: ConformanceReport, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "conformance.json",
        "html": output_dir / "conformance.html",
        "text": output_dir / "conformance.txt",
    }
    paths["json"].write_text(json.dumps(report.to_dict(), indent=2))
    paths["html"].write_text(render_html(report))
    paths["text"].write_text(render_text(report))
    return paths
