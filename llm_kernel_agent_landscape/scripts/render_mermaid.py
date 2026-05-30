#!/usr/bin/env python3
"""Generate Mermaid diagrams from landscape YAML data.

Usage:
    python scripts/render_mermaid.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT / "scripts"))

from mermaid_lib import (  # noqa: E402
    build_entries_by_id,
    entry_label,
    group_labels_by_date,
    load_landscape_data,
    render_timeline,
    resolve_entry_date,
    resolve_item_label,
    sanitize_mermaid_text,
)


def build_report_timeline(data: dict[str, Any]) -> str:
    entries_by_id = build_entries_by_id(data["entries"])
    timeline_cfg = data["timeline"]
    buckets: list[tuple[str, list[str]]] = []

    for bucket in timeline_cfg.get("buckets", []):
        date = bucket["date"]
        labels = [
            resolve_item_label(item, entries_by_id)
            for item in bucket.get("items", [])
        ]
        buckets.append((date, labels))

    title = data["config"]["timelines"]["report"]["title"]
    return render_timeline(title, buckets)


def build_full_timeline(data: dict[str, Any]) -> str:
    entries_by_id = build_entries_by_id(data["entries"])
    dates_cfg = data["timeline_dates"]
    period_defaults = dates_cfg.get("period_defaults", {})
    overrides = dates_cfg.get("overrides", {})

    items: list[tuple[str, str]] = []
    for entry in data["entries"]:
        if not entry.get("show_in_full", True):
            continue
        date = resolve_entry_date(entry, period_defaults, overrides)
        items.append((date, entry_label(entry)))

    if data["config"]["timelines"]["full"].get("include_external", True):
        for ext in dates_cfg.get("external", []):
            items.append((ext["date"], sanitize_mermaid_text(ext["label"])))

    buckets = group_labels_by_date(items)
    title = data["config"]["timelines"]["full"]["title"]
    return render_timeline(title, buckets)


def build_category_map(data: dict[str, Any]) -> str:
    cfg = data["config"]["category_map"]
    categories = data["categories"]["categories"]
    include = cfg.get("include_categories") or list(categories.keys())

    lines = ["flowchart TB"]

    for category in include:
        cat_cfg = categories[category]
        title_en = sanitize_mermaid_text(cat_cfg.get("title_en", category))
        node_id = category.replace(" ", "_").replace("/", "_").lower()
        lines.append(f'    subgraph {node_id}["{title_en}"]')
        lines.append("        direction TB")

        cat_entries = [
            entry for entry in data["entries"]
            if entry["category"] == category and entry.get("show_in_full", True)
        ]
        cat_entries.sort(key=lambda e: (e.get("year", 0), e.get("name", "")))

        for entry in cat_entries:
            safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", entry["id"])
            label = entry_label(entry)
            sub = sanitize_mermaid_text(entry.get("subcategory", ""))
            if sub:
                lines.append(f'        {safe_id}["{label}<br/><i>{sub}</i>"]')
            else:
                lines.append(f'        {safe_id}["{label}"]')

        lines.append("    end")

    return "\n".join(lines) + "\n"


def main() -> int:
    data = load_landscape_data()
    OUT.mkdir(parents=True, exist_ok=True)

    report = build_report_timeline(data)
    full = build_full_timeline(data)
    category = build_category_map(data)

    (OUT / "landscape_timeline_report.mmd").write_text(report, encoding="utf-8")
    (OUT / "landscape_timeline_full.mmd").write_text(full, encoding="utf-8")
    (OUT / "landscape_category_map.mmd").write_text(category, encoding="utf-8")

    report_count = sum(len(b.get("items", [])) for b in data["timeline"].get("buckets", []))
    full_count = len([e for e in data["entries"] if e.get("show_in_full", True)])
    full_count += len(data["timeline_dates"].get("external", []))

    print(f"Wrote {OUT / 'landscape_timeline_report.mmd'} ({report_count} items)")
    print(f"Wrote {OUT / 'landscape_timeline_full.mmd'} ({full_count} items)")
    print(f"Wrote {OUT / 'landscape_category_map.mmd'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
