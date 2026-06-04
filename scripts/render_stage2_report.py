from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_percent(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.2f}%"


def _fmt_number(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def _status_label(ok: bool | None) -> tuple[str, str]:
    if ok is True:
        return "ok", "ready"
    if ok is False:
        return "fail", "blocked"
    return "warn", "unknown"


def _load_history(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_series(history: Sequence[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in history:
        if "loss" not in row or row.get("loss") in (None, ""):
            continue
        value = _as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _line_chart(values: Sequence[float], *, color: str, label: str) -> str:
    if not values:
        return f'<div class="empty-chart">No {html.escape(label)} values logged.</div>'
    width = 720
    height = 180
    pad = 18
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1.0e-9)
    if len(values) == 1:
        points = [(width / 2.0, height / 2.0)]
    else:
        points = [
            (
                pad + (index * (width - (2 * pad)) / (len(values) - 1)),
                height - pad - ((value - min_value) / span * (height - (2 * pad))),
            )
            for index, value in enumerate(values)
        ]
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    marker_text = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" />'
        for x, y in points[-8:]
    )
    return f"""
<svg class="line-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)} line chart">
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" />
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" />
  <polyline points="{point_text}" />
  <g>{marker_text}</g>
  <text x="{pad}" y="14">{html.escape(label)} min {min_value:.2f}</text>
  <text x="{width - pad}" y="14" text-anchor="end">max {max_value:.2f}</text>
</svg>
""".replace("<polyline", f'<polyline style="stroke:{color}"').replace("<circle", f'<circle style="fill:{color}"')


def _bar_row(label: str, value: Any, *, class_name: str = "") -> str:
    number = _as_float(value)
    width = 0.0 if number is None else max(0.0, min(100.0, number))
    return f"""
<div class="bar-row {html.escape(class_name)}">
  <div class="bar-label">{html.escape(label)}</div>
  <div class="bar-track"><div class="bar-fill" style="width: {width:.2f}%"></div></div>
  <div class="bar-value">{_fmt_percent(value)}</div>
</div>
"""


def _diagnostic_rows(text: str | None) -> str:
    if not text:
        return '<tr><td colspan="6">No diagnostics logged.</td></tr>'
    rows: list[str] = []
    for line in str(text).splitlines():
        parts: dict[str, str] = {}
        for item in line.split(" | "):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            parts[key.strip()] = value.strip()
        rows.append(
            "<tr>"
            f"<td>{html.escape(parts.get('target', ''))}</td>"
            f"<td>{html.escape(parts.get('predicted', ''))}</td>"
            f"<td>{html.escape(parts.get('candidate_predicted', ''))}</td>"
            f"<td>{html.escape(parts.get('probe_predicted', ''))}</td>"
            f"<td>{html.escape(parts.get('baseline_predicted', ''))}</td>"
            f"<td>{html.escape(parts.get('decoded', ''))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _requirements(items: Iterable[Any]) -> str:
    requirements = [str(item) for item in items if item]
    if not requirements:
        return '<li class="ok-text">No missing requirements logged.</li>'
    return "\n".join(f"<li>{html.escape(item)}</li>" for item in requirements)


def render_stage2_report(report_path: Path, history_path: Path, output_path: Path) -> Path:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    history = _load_history(history_path)
    smoke = report.get("training_smoke_report") or {}
    probe_class, probe_label = _status_label(bool(smoke.get("latent_probe_ready")) if "latent_probe_ready" in smoke else None)
    decode_class, decode_label = _status_label(bool(smoke.get("latent_training_ready")) if "latent_training_ready" in smoke else None)
    phase_class, phase_label = _status_label(bool(report.get("passed")) if "passed" in report else None)
    loss_values = _metric_series(history, "loss")
    probe_values = _metric_series(history, "answer_probe_accuracy")
    contrast_values = _metric_series(history, "answer_contrast_accuracy")
    adapter_updates = _metric_series(history, "handoff_adapter_update_norm")
    probe_updates = _metric_series(history, "latent_answer_probe_update_norm")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LXP Stage II Latent Transfer Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f5f1;
      --panel: #ffffff;
      --ink: #171b1f;
      --muted: #5e6670;
      --line: #d9d6cf;
      --ok: #0f766e;
      --warn: #b45309;
      --fail: #b91c1c;
      --blue: #2563eb;
      --purple: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 48px) 20px;
      border-bottom: 1px solid var(--line);
      background: #fbfaf7;
    }}
    main {{ padding: 24px clamp(18px, 4vw, 48px) 48px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(24px, 4vw, 38px); letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; }}
    .subhead {{ color: var(--muted); max-width: 980px; }}
    .grid {{ display: grid; gap: 14px; }}
    .cards {{ grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin-bottom: 20px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
    }}
    .card-value {{ font-size: 28px; font-weight: 720; margin: 4px 0; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      color: white;
    }}
    .ok .badge, .ok-badge {{ background: var(--ok); }}
    .warn .badge, .warn-badge {{ background: var(--warn); }}
    .fail .badge, .fail-badge {{ background: var(--fail); }}
    .status-row {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(120px, 1.1fr) minmax(140px, 3fr) 78px; gap: 10px; align-items: center; margin: 10px 0; }}
    .bar-label {{ color: var(--muted); }}
    .bar-track {{ height: 12px; background: #ece9e2; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--blue); border-radius: 999px; }}
    .probe .bar-fill {{ background: var(--ok); }}
    .decode .bar-fill {{ background: var(--fail); }}
    .baseline .bar-fill {{ background: var(--warn); }}
    .chart-grid {{ grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .line-chart {{ width: 100%; height: auto; min-height: 170px; }}
    .line-chart line {{ stroke: var(--line); stroke-width: 1; }}
    .line-chart polyline {{ fill: none; stroke-width: 3; }}
    .line-chart text {{ fill: var(--muted); font-size: 12px; }}
    .empty-chart {{ color: var(--muted); padding: 44px 12px; text-align: center; border: 1px dashed var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    ul {{ margin: 8px 0 0; padding-left: 20px; color: var(--muted); }}
    .ok-text {{ color: var(--ok); }}
    .two-col {{ grid-template-columns: minmax(0, 1.2fr) minmax(260px, 0.8fr); }}
    @media (max-width: 820px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
      .bar-value {{ font-weight: 700; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>LXP Stage II Latent Transfer Report</h1>
    <div class="subhead">
      {html.escape(str(report.get("model_pair", "unknown model pair")))} | device {html.escape(str(report.get("effective_device", "unknown")))} | dtype {html.escape(str(report.get("effective_torch_dtype", report.get("torch_dtype", "unknown"))))}
    </div>
  </header>
  <main class="grid">
    <section class="grid cards">
      <div class="panel {probe_class}">
        <div class="status-row"><h3>Latent Probe</h3><span class="badge">{probe_label}</span></div>
        <div class="card-value">{_fmt_percent(smoke.get("final_heldout_latent_probe_accuracy"))}</div>
        <div class="subhead">Trainable readout over compressed latent prefixes.</div>
      </div>
      <div class="panel {decode_class}">
        <div class="status-row"><h3>Free Decode</h3><span class="badge">{decode_label}</span></div>
        <div class="card-value">{_fmt_percent(smoke.get("final_heldout_exact_match_accuracy"))}</div>
        <div class="subhead">Open-ended actor generation after latent handoff.</div>
      </div>
      <div class="panel">
        <h3>Candidate NLL</h3>
        <div class="card-value">{_fmt_percent(smoke.get("final_heldout_latent_candidate_accuracy"))}</div>
        <div class="subhead">Frozen actor candidate ranking from the latent prefix.</div>
      </div>
      <div class="panel">
        <h3>First Token</h3>
        <div class="card-value">{_fmt_percent(smoke.get("final_heldout_latent_first_token_accuracy"))}</div>
        <div class="subhead">Greedy answer-boundary token accuracy, mean rank {_fmt_number(smoke.get("final_heldout_latent_first_token_rank_mean"))}.</div>
      </div>
      <div class="panel {phase_class}">
        <div class="status-row"><h3>Phase Gate</h3><span class="badge">{phase_label}</span></div>
        <div class="card-value">{_fmt_number(smoke.get("final_heldout_answer_perplexity"))}</div>
        <div class="subhead">Final heldout answer perplexity.</div>
      </div>
    </section>

    <section class="grid two-col">
      <div class="panel">
        <h2>Accuracy Surfaces</h2>
        {_bar_row("Open decode exact match", smoke.get("final_heldout_exact_match_accuracy"), class_name="decode")}
        {_bar_row("Latent first token", smoke.get("final_heldout_latent_first_token_accuracy"))}
        {_bar_row("Latent candidate NLL", smoke.get("final_heldout_latent_candidate_accuracy"))}
        {_bar_row("Latent answer probe", smoke.get("final_heldout_latent_probe_accuracy"), class_name="probe")}
        {_bar_row("Actor text baseline", smoke.get("final_heldout_actor_text_baseline_accuracy"), class_name="baseline")}
      </div>
      <div class="panel">
        <h2>Recommendation</h2>
        <p>
          Present the current result as a latent-readout workbench: the probe verifies that the compressed latent prefix carries answer information, while the free-form decoder remains a known interface failure.
        </p>
        <p>
          The next production step is to replace the fixed probe with a candidate-conditioned readout or train the actor-side decoder objective until open decode stops collapsing.
        </p>
      </div>
    </section>

    <section class="grid chart-grid">
      <div class="panel">
        <h2>Training Loss</h2>
        {_line_chart(loss_values, color="#2563eb", label="loss")}
      </div>
      <div class="panel">
        <h2>Probe Accuracy By Step</h2>
        {_line_chart(probe_values, color="#0f766e", label="answer probe accuracy")}
      </div>
      <div class="panel">
        <h2>Candidate Contrast Accuracy</h2>
        {_line_chart(contrast_values, color="#7c3aed", label="candidate contrast accuracy")}
      </div>
      <div class="panel">
        <h2>Module Updates</h2>
        {_line_chart(adapter_updates, color="#b45309", label="adapter update norm")}
        {_line_chart(probe_updates, color="#0f766e", label="probe update norm")}
      </div>
    </section>

    <section class="panel">
      <h2>Final Diagnostics</h2>
      <table>
        <thead>
          <tr><th>Target</th><th>Decode</th><th>Candidate</th><th>Probe</th><th>Actor Baseline</th><th>Decoded Text</th></tr>
        </thead>
        <tbody>
          {_diagnostic_rows(smoke.get("heldout_eval_diagnostics"))}
        </tbody>
      </table>
    </section>

    <section class="grid two-col">
      <div class="panel">
        <h2>Smoke Gate Gaps</h2>
        <ul>{_requirements(smoke.get("missing_requirements") or [])}</ul>
      </div>
      <div class="panel">
        <h2>Phase Gate Gaps</h2>
        <ul>{_requirements(report.get("missing_requirements") or [])}</ul>
      </div>
    </section>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a static Stage II training report.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = render_stage2_report(args.report, args.history, args.output)
    print(f"Wrote visual Stage II report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
