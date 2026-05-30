#!/usr/bin/env python3
"""Fetch candidate lines from upstream awesome repos.

This script does NOT modify data/entries.yaml. It only writes a rough candidates file
for manual review.

Usage:
    python scripts/fetch_candidates.py
"""
from __future__ import annotations

from pathlib import Path
import re
import urllib.request
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "candidates.yaml"

README_URLS = [
    "https://raw.githubusercontent.com/qhy991/Awesome-LLM-Kernel-Agent/main/README.md",
    "https://raw.githubusercontent.com/flagos-ai/awesome-LLM-driven-kernel-generation/main/README.md",
]

LINK_RE = re.compile(r"\[([^\]]{2,80})\]\((https?://[^)]+)\)")


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> None:
    candidates = []
    for url in README_URLS:
        try:
            text = fetch(url)
        except Exception as exc:
            print(f"Failed to fetch {url}: {exc}")
            continue
        for name, link in LINK_RE.findall(text):
            # Keep likely paper/project links and ignore badges/images/noise.
            if len(name) < 3 or name.lower() in {"paper", "code", "github", "arxiv"}:
                continue
            if "img.shields" in link or "badge" in link:
                continue
            candidates.append({"name": name.strip(), "url": link.strip(), "source_readme": url})
    OUT.write_text(yaml.safe_dump(candidates, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(candidates)} candidates to {OUT}")


if __name__ == "__main__":
    main()
