#!/usr/bin/env python3
"""Inject generated Mermaid blocks into the root README.

Usage:
    python scripts/sync_readme.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mermaid_lib import load_landscape_data  # noqa: E402


def wrap_mermaid(content: str) -> str:
    body = content.rstrip("\n")
    return f"```mermaid\n{body}\n```"


def replace_between(text: str, begin: str, end: str, replacement: str) -> str:
    start = text.find(begin)
    stop = text.find(end)
    if start == -1 or stop == -1 or stop < start:
        raise ValueError(f"Markers not found or out of order: {begin!r} ... {end!r}")
    stop += len(end)
    return text[:start] + begin + "\n\n" + replacement + "\n\n" + end + text[stop:]


def main() -> int:
    data = load_landscape_data()
    cfg = data["config"]
    readme_path = (ROOT / cfg["readme_path"]).resolve()

    report_mmd = (ROOT / "figures" / "landscape_timeline_report.mmd").read_text(encoding="utf-8")
    full_mmd = (ROOT / "figures" / "landscape_timeline_full.mmd").read_text(encoding="utf-8")
    category_mmd = (ROOT / "figures" / "landscape_category_map.mmd").read_text(encoding="utf-8")

    text = readme_path.read_text(encoding="utf-8")
    markers = cfg["markers"]

    text = replace_between(
        text,
        markers["report_timeline"]["begin"],
        markers["report_timeline"]["end"],
        wrap_mermaid(report_mmd),
    )
    text = replace_between(
        text,
        markers["full_timeline"]["begin"],
        markers["full_timeline"]["end"],
        wrap_mermaid(full_mmd),
    )
    text = replace_between(
        text,
        markers["category_map"]["begin"],
        markers["category_map"]["end"],
        wrap_mermaid(category_mmd),
    )

    readme_path.write_text(text, encoding="utf-8")
    print(f"Updated {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
