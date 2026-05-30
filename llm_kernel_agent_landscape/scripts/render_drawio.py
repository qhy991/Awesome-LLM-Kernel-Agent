#!/usr/bin/env python3
"""Generate an editable draw.io / diagrams.net version of the report landscape.

Usage:
    python scripts/render_drawio.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import itertools
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"

WIDTH = 1800
HEIGHT = 1050


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_positions(entries: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    periods = cfg["periods"]
    categories = cfg["categories"]
    visible = [e for e in entries if e.get("show_in_report", True)]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in visible:
        groups[(e["category"], e["period"])].append(e)
    positions: dict[str, tuple[float, float]] = {}
    for (cat, period), group in groups.items():
        x = categories[cat]["x"]
        y0 = periods[period]["y"]
        group = sorted(group, key=lambda e: (e.get("display_priority", 99), e["name"]))
        n = len(group)
        for i, e in enumerate(group):
            y = y0 + (i - (n - 1) / 2.0) * 54 + e.get("y_offset", 0)
            positions[e["id"]] = (x + 22 + e.get("x_offset", 0), y)
    return positions


def style_rect(fill: str, stroke: str, font: str, bold: bool = False, dashed: bool = False) -> str:
    s = [
        "rounded=1", "whiteSpace=wrap", "html=1", "arcSize=12",
        f"fillColor={fill}", f"strokeColor={stroke}", f"fontColor={font}",
        "fontFamily=Helvetica", "align=center", "verticalAlign=middle",
    ]
    if bold:
        s.append("fontStyle=1")
    if dashed:
        s.append("dashed=1")
    return ";".join(s) + ";"


def style_text(color: str, bold: bool = False, size: int = 16) -> str:
    s = [
        "text", "html=1", "strokeColor=none", "fillColor=none",
        "align=center", "verticalAlign=middle", "whiteSpace=wrap",
        f"fontColor={color}", "fontFamily=Microsoft YaHei", f"fontSize={size}",
    ]
    if bold:
        s.append("fontStyle=1")
    return ";".join(s) + ";"


def style_line(color: str, width: float = 1.0, dashed: bool = False) -> str:
    s = [
        "endArrow=none", "html=1", "rounded=0", f"strokeColor={color}", f"strokeWidth={width}",
    ]
    if dashed:
        s.append("dashed=1")
    return ";".join(s) + ";"


def add_vertex(root, id_gen, value: str, x: float, y: float, w: float, h: float, style: str, parent: str = "1") -> str:
    cid = f"n{id_gen()}"
    cell = ET.SubElement(root, "mxCell", {"id": cid, "value": value, "style": style, "vertex": "1", "parent": parent})
    ET.SubElement(cell, "mxGeometry", {"x": str(round(x, 2)), "y": str(round(y, 2)), "width": str(round(w, 2)), "height": str(round(h, 2)), "as": "geometry"})
    return cid


def add_line(root, id_gen, x1: float, y1: float, x2: float, y2: float, style: str, parent: str = "1") -> str:
    cid = f"e{id_gen()}"
    cell = ET.SubElement(root, "mxCell", {"id": cid, "style": style, "edge": "1", "parent": parent})
    geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.SubElement(geo, "mxPoint", {"x": str(round(x1, 2)), "y": str(round(y1, 2)), "as": "sourcePoint"})
    ET.SubElement(geo, "mxPoint", {"x": str(round(x2, 2)), "y": str(round(y2, 2)), "as": "targetPoint"})
    return cid


def main() -> None:
    cfg = load_yaml(DATA / "categories.yaml")
    entries = load_yaml(DATA / "entries.yaml")
    categories = cfg["categories"]
    periods = cfg["periods"]
    positions = compute_positions(entries, cfg)

    OUT.mkdir(exist_ok=True)
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "modified": "2026-05-24T00:00:00.000Z", "agent": "KernelOwl landscape renderer", "version": "24.0.0"})
    diagram = ET.SubElement(mxfile, "diagram", {"name": "Landscape"})
    model = ET.SubElement(diagram, "mxGraphModel", {"dx": "1200", "dy": "800", "grid": "1", "gridSize": "10", "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1", "pageScale": "1", "pageWidth": str(WIDTH), "pageHeight": str(HEIGHT), "math": "0", "shadow": "0"})
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    counter = itertools.count(2)
    id_gen = lambda: next(counter)

    # Title and subtitle
    add_vertex(root, id_gen, "LLM-driven Kernel Engineering Landscape", 0, 28, WIDTH, 50, style_text("#0B1F3A", bold=True, size=34))
    add_vertex(
        root,
        id_gen,
        "Model capability · Agent workflow · Data · Evaluation · Systems",
        0,
        78,
        WIDTH,
        32,
        style_text("#334155", bold=True, size=16),
    )
    add_vertex(root, id_gen, "Time / maturity", 20, 150, 120, 40, style_text("#334155", bold=True, size=16))

    # Year bands
    for period, meta in periods.items():
        y = meta["y"]
        add_line(root, id_gen, 130, y, 1660, y, style_line("#A9B1BD", 1.0, dashed=True))
        add_vertex(root, id_gen, period, 24, y - 22, 88, 44, style_rect("#F8FAFC", "#CBD5E1", "#334155", bold=True))

    # Columns
    for cat, meta in categories.items():
        x = meta["x"]
        color = meta["color"]
        add_vertex(root, id_gen, cat, x - 120, 135, 240, 58, style_rect("#FFFFFF", color, color, bold=True))
        add_vertex(root, id_gen, meta.get("title_en", meta.get("title_cn", "")), x - 120, 176, 240, 20, style_text(color, bold=True, size=8))
        add_line(root, id_gen, x - 90, 208, x - 90, 805, style_line(color, 2.0))
        for pmeta in periods.values():
            y = pmeta["y"]
            add_vertex(root, id_gen, "", x - 98, y - 8, 16, 16, style_rect("#FFFFFF", color, color))

    # Nodes
    for e in entries:
        if not e.get("show_in_report", True):
            continue
        x, y = positions[e["id"]]
        meta = categories[e["category"]]
        color = meta["color"]
        fill = meta.get("soft_fill", "#FFFFFF")
        stroke = "#F6A800" if e.get("highlight", False) else color
        add_vertex(root, id_gen, e["name"].replace("\n", "<br>"), x - 90, y - 30, 180, 60, style_rect(fill, stroke, color, bold=True))

    # Legend
    add_vertex(
        root,
        id_gen,
        "Column = research area<br>Line = time band<br>Gold border = highlight",
        1460,
        850,
        250,
        110,
        style_rect("#FFFFFF", "#CBD5E1", "#334155", bold=False),
    )
    add_vertex(root, id_gen, "Sources: qhy991/Awesome-LLM-Kernel-Agent; flagos-ai/awesome-LLM-driven-kernel-generation. Representative works only.", 90, 1000, 700, 24, style_text("#64748B", bold=False, size=8))

    tree = ET.ElementTree(mxfile)
    path = OUT / "landscape_report.drawio"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
