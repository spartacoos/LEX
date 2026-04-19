"""
Pytest configuration for LEX tests.

Key responsibility: collect per-question eval metrics as tests run, then
emit:
  - tests/reports/eval-YYYYMMDD-HHMMSS.csv   (one row per question)
  - tests/reports/eval-YYYYMMDD-HHMMSS.md    (human-readable summary)

The CSV is the authoritative artefact — append-friendly for plotting
or diffing across runs. The Markdown is a convenience view.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from statistics import mean

import pytest


# Where per-run reports live.
_REPORTS_DIR = Path(__file__).parent / "reports"
_REPORTS_DIR.mkdir(exist_ok=True)

# Module-level collector. The eval test appends one dict per question.
_metric_rows: list[dict] = []


def record_eval_row(row: dict) -> None:
    """Called from inside the eval test for each question."""
    _metric_rows.append(row)


def pytest_sessionfinish(session, exitstatus):
    """After all tests finish, write reports — but only if we have data."""
    if not _metric_rows:
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = _REPORTS_DIR / f"eval-{timestamp}.csv"
    md_path = _REPORTS_DIR / f"eval-{timestamp}.md"

    # --- CSV: one row per question ------------------------------------
    # Column order kept stable so diffs across runs are easy to read.
    fieldnames = [
        "id", "category", "question",
        "context_precision", "context_recall",
        "faithfulness", "answer_relevancy",
        "citation_correctness",
        "retrieved_articles", "expected_articles",
        "answer_chars", "citations_count",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in _metric_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # --- Markdown: grouped by category with averages ------------------
    lines: list[str] = []
    lines.append(f"# LEX eval report — {timestamp}")
    lines.append("")
    lines.append(f"Questions: **{len(_metric_rows)}**")
    lines.append("")

    # Overall summary.
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Mean | Target |")
    lines.append("|---|---:|---:|")
    for key, target in [
        ("context_precision", 0.80),
        ("context_recall", 0.80),
        ("faithfulness", 0.90),
        ("answer_relevancy", 0.85),
        ("citation_correctness", 0.90),
    ]:
        values = [r[key] for r in _metric_rows if r.get(key) is not None]
        if not values:
            continue
        avg = mean(values)
        mark = "✅" if avg >= target else "❌"
        lines.append(f"| {key} | {avg:.3f} | {target:.2f} {mark} |")
    lines.append("")

    # Per-category breakdown.
    categories = sorted({r["category"] for r in _metric_rows})
    for cat in categories:
        rows = [r for r in _metric_rows if r["category"] == cat]
        lines.append(f"## Category: {cat} ({len(rows)} questions)")
        lines.append("")
        lines.append("| id | citation ✓ | precision | recall | faith | rel | question |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for r in rows:
            q = r["question"][:60] + ("…" if len(r["question"]) > 60 else "")
            lines.append(
                f"| {r['id']} "
                f"| {r.get('citation_correctness', 0):.2f} "
                f"| {r.get('context_precision', 0):.2f} "
                f"| {r.get('context_recall', 0):.2f} "
                f"| {r.get('faithfulness', 0):.2f} "
                f"| {r.get('answer_relevancy', 0):.2f} "
                f"| {q} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines))

    # Print paths to terminal so it's obvious where the reports landed.
    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter:
        terminalreporter.write_line("")
        terminalreporter.write_line(f"eval report (csv): {csv_path}")
        terminalreporter.write_line(f"eval report (md):  {md_path}")
