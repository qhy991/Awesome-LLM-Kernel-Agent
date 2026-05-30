#!/usr/bin/env python3
"""Render the LLM-driven Kernel Engineering Landscape as SVG/PNG/PDF.

Usage:
    python scripts/render_landscape.py --view report
    python scripts/render_landscape.py --view full
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any
import textwrap

import yaml
import svgwrite

try:
    import cairosvg  # type: ignore
except Exception:  # pragma: no cover
    cairosvg = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"

FONT = "Inter, Helvetica Neue, Arial, sans-serif"
FONT_MONO = "SF Mono, Menlo, Consolas, monospace"

BACKEND_ABBR = {
    "CUDA": "CU",
    "Triton": "Tri",
    "HIP": "HIP",
    "Metal": "MTL",
    "NPU": "NPU",
    "TPU": "TPU",
    "CANN": "CANN",
    "Ascend": "ASC",
    "MetaX": "MX",
    "HPC": "HPC",
}

PERIOD_ORDER = ["2026+", "2025 H2", "2025 H1", "2024"]

VIEW_LAYOUT = {
    "report": {
        "width": 1920,
        "content_top": 230,
        "bottom_margin": 120,
        "node_w": 192,
        "node_h": 78,
        "node_gap": 14,  # gap between node boxes (center-to-center = node_h + node_gap)
        "period_band_pad": 44,
        "font_title": 14,
        "font_sub": 10,
        "title_size": 32,
    },
    "full": {
        "width": 2100,
        "content_top": 230,
        "bottom_margin": 120,
        "node_w": 178,
        "node_h": 72,
        "node_gap": 12,
        "period_band_pad": 40,
        "font_title": 13,
        "font_sub": 9,
        "title_size": 30,
    },
}


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_text(
    dwg: svgwrite.Drawing,
    text: str,
    x: float,
    y: float,
    *,
    size: int = 16,
    fill: str = "#0B1F3A",
    weight: str = "normal",
    anchor: str = "middle",
    line_height: int | None = None,
    font_family: str = FONT,
) -> None:
    lines = str(text).split("\n")
    if line_height is None:
        line_height = int(size * 1.22)
    base_y = y - (len(lines) - 1) * line_height / 2
    node = dwg.text(
        "",
        insert=(x, base_y),
        text_anchor=anchor,
        font_size=size,
        fill=fill,
        font_weight=weight,
        font_family=font_family,
    )
    for i, line in enumerate(lines):
        node.add(dwg.tspan(line, x=[x], dy=[0 if i == 0 else line_height]))
    dwg.add(node)


def wrapped_label(label: str, width: int = 16) -> str:
    if "\n" in label:
        return label
    if len(label) <= width:
        return label
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=True))


def draw_header(dwg: svgwrite.Drawing, x: float, label: str, color: str, subtitle: str) -> None:
    dwg.add(
        dwg.rect(
            insert=(x - 128, 128),
            size=(256, 64),
            rx=12,
            ry=12,
            fill="#FFFFFF",
            stroke=color,
            stroke_width=2.0,
        )
    )
    # soft header tint
    dwg.add(
        dwg.rect(
            insert=(x - 128, 128),
            size=(256, 64),
            rx=12,
            ry=12,
            fill=color,
            opacity=0.06,
        )
    )
    add_text(dwg, label, x, 158, size=22, fill=color, weight="bold")
    add_text(dwg, subtitle, x, 182, size=10, fill=color, weight="bold")


def draw_column_band(
    dwg: svgwrite.Drawing, x: float, y_top: float, y_bot: float, color: str, soft_fill: str
) -> None:
    dwg.add(
        dwg.rect(
            insert=(x - 108, y_top),
            size=(216, y_bot - y_top),
            rx=14,
            ry=14,
            fill=soft_fill,
            opacity=0.35,
        )
    )


def draw_backend_pills(
    dwg: svgwrite.Drawing,
    x: float,
    y: float,
    backends: list[str],
    color: str,
    *,
    compact: bool = False,
) -> None:
    shown = backends[:4]
    if not shown:
        return
    pill_h = 14 if compact else 15
    pill_w = 28
    gap = 3
    total_w = len(shown) * pill_w + (len(shown) - 1) * gap
    start_x = x - total_w / 2 + pill_w / 2
    for i, b in enumerate(shown):
        px = start_x + i * (pill_w + gap)
        abbr = BACKEND_ABBR.get(b, b[:3].upper())
        dwg.add(
            dwg.rect(
                insert=(px - pill_w / 2, y - pill_h / 2),
                size=(pill_w, pill_h),
                rx=4,
                ry=4,
                fill="#FFFFFF",
                stroke=color,
                stroke_width=0.9,
                opacity=0.95,
            )
        )
        add_text(
            dwg,
            abbr,
            px,
            y + 1,
            size=8,
            fill=color,
            weight="bold",
            font_family=FONT_MONO,
        )


def draw_node(
    dwg: svgwrite.Drawing,
    x: float,
    y: float,
    entry: dict[str, Any],
    color: str,
    fill: str,
    layout: dict[str, Any],
) -> None:
    w = layout["node_w"]
    h = layout["node_h"]
    highlight = entry.get("highlight", False)
    stroke = "#E8A317" if highlight else color
    sw = 3.2 if highlight else 1.4

    dwg.add(
        dwg.rect(
            insert=(x - w / 2, y - h / 2),
            size=(w, h),
            rx=11,
            ry=11,
            fill=fill,
            stroke=stroke,
            stroke_width=sw,
        )
    )
    if highlight:
        dwg.add(
            dwg.rect(
                insert=(x - w / 2 - 4, y - h / 2 - 4),
                size=(w + 8, h + 8),
                rx=14,
                ry=14,
                fill="none",
                stroke="#E8A317",
                stroke_width=1.0,
                opacity=0.5,
            )
        )

    name_y = y - 10
    title_color = "#0B1F3A" if highlight else color
    add_text(
        dwg,
        wrapped_label(entry["name"], 15),
        x,
        name_y,
        size=layout["font_title"],
        fill=title_color,
        weight="bold",
    )

    aff = entry.get("affiliation_short", "")
    if aff:
        add_text(
            dwg,
            wrapped_label(aff, 22),
            x,
            y + 14,
            size=layout["font_sub"],
            fill="#64748B",
            weight="normal",
        )

    backends = entry.get("backends") or []
    draw_backend_pills(dwg, x, y + h / 2 - 11, backends, color, compact=True)


def compute_layout(
    entries: list[dict[str, Any]], cfg: dict[str, Any], view: str
) -> tuple[dict[str, tuple[float, float]], dict[str, Any], int]:
    """Return positions, layout dict, and computed canvas height.

    Spacing is **not** fixed globally. For each time band we:
    1. Use center-to-center spacing of ``node_h + node_gap`` (no overlap within a cell).
    2. Stack time bands vertically with padding based on the densest column in that band.
    """
    categories = cfg["categories"]
    layout = dict(VIEW_LAYOUT[view])
    visible_flag = "show_in_full" if view == "full" else "show_in_report"
    visible = [e for e in entries if e.get(visible_flag, True)]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in visible:
        groups[(e["category"], e["period"])].append(e)

    node_h = layout["node_h"]
    row_spacing = node_h + layout["node_gap"]
    band_pad = layout["period_band_pad"]

    period_max_n: dict[str, int] = {p: 0 for p in PERIOD_ORDER}
    for (cat, period), group in groups.items():
        if period in period_max_n:
            period_max_n[period] = max(period_max_n[period], len(group))

    period_y: dict[str, float] = {}
    y_cursor = layout["content_top"]
    for period in PERIOD_ORDER:
        n = period_max_n[period]
        if n == 0:
            period_y[period] = y_cursor + 36
            y_cursor += 72
            continue
        stack_h = (n - 1) * row_spacing if n > 1 else 0
        period_y[period] = y_cursor + stack_h / 2.0
        y_cursor += stack_h + band_pad

    height = int(y_cursor + layout["bottom_margin"])
    layout["row_spacing"] = row_spacing
    layout["period_y"] = period_y
    layout["height"] = height

    positions: dict[str, tuple[float, float]] = {}
    for (cat, period), group in groups.items():
        x = categories[cat]["x"]
        y0 = period_y.get(period, layout["content_top"])
        group = sorted(group, key=lambda e: (e.get("display_priority", 99), e["name"]))
        n = len(group)
        for i, e in enumerate(group):
            y = y0 + (i - (n - 1) / 2.0) * row_spacing + e.get("y_offset", 0)
            x_final = x + 24 + e.get("x_offset", 0)
            positions[e["id"]] = (x_final, y)

    return positions, layout, height


def render(view: str = "report") -> Path:
    cfg = load_yaml(DATA / "categories.yaml")
    entries = load_yaml(DATA / "entries.yaml")
    categories = cfg["categories"]

    positions, layout, height = compute_layout(entries, cfg, view)
    width = layout["width"]
    visible_flag = "show_in_full" if view == "full" else "show_in_report"
    visible_entries = [e for e in entries if e.get(visible_flag, True)]
    n_visible = len(visible_entries)

    OUT.mkdir(exist_ok=True)
    svg_path = OUT / f"landscape_{view}.svg"
    dwg = svgwrite.Drawing(str(svg_path), size=(width, height), profile="full")

    # Background gradient
    defs = dwg.defs
    grad = dwg.linearGradient(start=("0%", "0%"), end=("100%", "100%"), id="bg_grad")
    grad.add_stop_color(offset="0%", color="#FAFCFF")
    grad.add_stop_color(offset="100%", color="#F4F7FB")
    defs.add(grad)
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="url(#bg_grad)"))

    # Title block
    title = "LLM-driven Kernel Engineering Landscape"
    if view == "full":
        title += " (Complete)"
    add_text(dwg, title, width / 2, 52, size=layout["title_size"], fill="#0B1F3A", weight="bold")
    subtitle = (
        "Model capability · Agent workflow · Data assets · Evaluation · Integrated systems"
    )
    add_text(dwg, subtitle, width / 2, 88, size=15, fill="#475569", weight="bold")
    stats = f"{n_visible} entries · {view} view · May 2026"
    add_text(dwg, stats, width / 2, 112, size=11, fill="#94A3B8", weight="normal")

    period_y = layout["period_y"]
    y_top, y_bot = layout["content_top"] - 12, height - layout["bottom_margin"] + 20
    for cat, meta in categories.items():
        draw_column_band(dwg, meta["x"], y_top, y_bot, meta["color"], meta.get("soft_fill", "#F8FAFC"))

    add_text(dwg, "Time / maturity →", 82, 168, size=14, fill="#475569", weight="bold")

    for period in PERIOD_ORDER:
        y = period_y[period]
        dwg.add(
            dwg.line(
                start=(128, y),
                end=(width - 80, y),
                stroke="#CBD5E1",
                stroke_width=1.0,
                stroke_dasharray="6,10",
            )
        )
        dwg.add(
            dwg.rect(
                insert=(22, y - 24),
                size=(96, 48),
                rx=8,
                ry=8,
                fill="#FFFFFF",
                stroke="#E2E8F0",
                stroke_width=1.2,
            )
        )
        add_text(dwg, period, 70, y + 5, size=17, fill="#334155", weight="bold")

    for cat, meta in categories.items():
        x = meta["x"]
        color = meta["color"]
        draw_header(dwg, x, cat, color, meta.get("title_en", meta.get("title_cn", "")))
        dwg.add(
            dwg.line(
                start=(x - 96, 208),
                end=(x - 96, y_bot),
                stroke=color,
                stroke_width=2.2,
                opacity=0.85,
            )
        )
        for p in PERIOD_ORDER:
            py = period_y[p]
            dwg.add(
                dwg.circle(
                    center=(x - 96, py),
                    r=7,
                    fill="#FFFFFF",
                    stroke=color,
                    stroke_width=1.6,
                )
            )

    for e in visible_entries:
        x, y = positions[e["id"]]
        meta = categories[e["category"]]
        draw_node(dwg, x, y, e, meta["color"], meta.get("soft_fill", "#F8FAFC"), layout)

    # Legend
    legend_w, legend_h = 280, 132
    legend_x = width - legend_w - 36
    legend_y = height - 150
    dwg.add(
        dwg.rect(
            insert=(legend_x, legend_y),
            size=(legend_w, legend_h),
            rx=12,
            ry=12,
            fill="#FFFFFF",
            stroke="#E2E8F0",
            stroke_width=1.2,
            opacity=0.97,
        )
    )
    lx = legend_x + 20
    ly = legend_y + 22
    for i, cat in enumerate(categories.keys()):
        dwg.add(
            dwg.rect(
                insert=(lx + i * 16, ly - 8),
                size=(10, 18),
                fill=categories[cat]["color"],
                stroke="none",
            )
        )
    add_text(dwg, "Column = area", legend_x + 148, ly + 2, size=12, fill="#334155", anchor="start")
    dwg.add(
        dwg.line(
            start=(lx, ly + 26),
            end=(lx + 50, ly + 26),
            stroke="#94A3B8",
            stroke_width=1.4,
            stroke_dasharray="5,5",
        )
    )
    add_text(dwg, "Line = time band", legend_x + 58, ly + 31, size=12, fill="#334155", anchor="start")
    add_text(dwg, "Subtitle = affiliation", legend_x + 20, ly + 52, size=12, fill="#64748B", anchor="start")
    draw_backend_pills(dwg, legend_x + 52, ly + 68, ["CUDA", "Triton"], "#1976D2")
    add_text(dwg, "Badge = backend", legend_x + 88, ly + 69, size=12, fill="#64748B", anchor="start")
    dwg.add(
        dwg.rect(
            insert=(lx, ly + 84),
            size=(48, 20),
            rx=5,
            ry=5,
            fill="#FFFFFF",
            stroke="#E8A317",
            stroke_width=2.2,
        )
    )
    add_text(dwg, "Gold border = highlight", legend_x + 58, ly + 98, size=12, fill="#334155", anchor="start")

    footer = (
        "Sources: Awesome-LLM-Kernel-Agent · awesome-LLM-driven-kernel-generation · "
        f"Curated {len(entries)} works, showing {n_visible} in {view} view"
    )
    add_text(dwg, footer, 40, height - 28, size=9, fill="#94A3B8", anchor="start")

    dwg.save()

    if cairosvg is not None:
        try:
            cairosvg.svg2png(
                url=str(svg_path),
                write_to=str(OUT / f"landscape_{view}.png"),
                output_width=width,
            )
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(OUT / f"landscape_{view}.pdf"))
        except Exception as exc:  # pragma: no cover
            print(f"Warning: SVG generated but PNG/PDF conversion failed: {exc}")
    else:
        print("Warning: cairosvg not installed; generated SVG only.")

    print(f"Wrote {svg_path} ({width}×{height}, {n_visible} nodes)")
    return svg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["report", "full"], default="report")
    args = parser.parse_args()
    render(args.view)


if __name__ == "__main__":
    main()
